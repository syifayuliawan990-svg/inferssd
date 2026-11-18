# SPDX-License-Identifier: Apache-2.0
"""Model identity, portable marginals and unchanged default profiling behavior."""

import copy
import json
from dataclasses import replace

import pytest
from test_decode_profiling import GIB, MODEL, estimate, setup
from test_machine_command import execute, machine
from test_machine_probes import FakeRunner

from inferssd.app import catalog
from inferssd.machine import command
from inferssd.strategy.heat import fingerprint, heat_from_counts, heat_from_json


def hot(model=MODEL):
    f = model.facts
    counts = [[1 if e < f.experts_used else 0 for e in range(f.experts)] for _ in range(f.layers)]
    return heat_from_counts(f, model.revision, [counts], "Synthetic decode; one complete step")


def test_equal_request_weight_and_portable_round_trip():
    model = replace(MODEL.facts, layers=2, experts=4, experts_used=2)
    a = [[1, 1, 0, 0]] * 2
    b = [[0, 0, 100, 100]] * 2
    heat = heat_from_counts(model, "checkpoint", [a, b], "two decode requests")
    assert heat.probabilities == ((0.25, 0.25, 0.25, 0.25),) * 2
    assert heat.coverage(3) == 0.375
    assert heat.coverage(100) == 1
    assert heat_from_json(json.loads(json.dumps(heat.to_json()))) == heat
    assert fingerprint(replace(model, paths=("different-storage.gguf",))) == fingerprint(model)
    assert not any(key in heat.to_json() for key in ("tokens_per_second", "device", "hit_rate"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("probabilities", [[float("nan")]]),
        ("probabilities", [[float("inf")]]),
        ("probabilities", [[-0.1, 1.1]]),
        ("probabilities", [[0.4, 0.4]]),
        ("probabilities", [[True]]),
        ("requests", 0),
        ("requests", True),
        ("workload", ""),
        ("model_revision", ""),
        ("model_fingerprint", "invalid"),
    ],
)
def test_reject_malformed_artifacts(field, value):
    data = hot().to_json()
    data[field] = value
    with pytest.raises(ValueError):
        heat_from_json(data)


def test_no_silent_hardware_or_timing_calibration_fields():
    data = hot().to_json()
    data["residual_ms"] = 10
    with pytest.raises(ValueError, match="fields"):
        heat_from_json(data)


def test_binding_rejects_revision_quantization_dimensions_and_impossible_frequency():
    h = hot()
    with pytest.raises(ValueError, match="revision"):
        h.check(MODEL.facts, "other-checkpoint")
    with pytest.raises(ValueError, match="quantization"):
        h.check(replace(MODEL.facts, expert_bytes=1), MODEL.revision)
    with pytest.raises(ValueError, match="dimensions"):
        replace(h, probabilities=h.probabilities[:1]).check(MODEL.facts, MODEL.revision)
    impossible = ((1.0,) + (0.0,) * (MODEL.facts.experts - 1),) * MODEL.facts.layers
    with pytest.raises(ValueError, match="top-k"):
        replace(h, probabilities=impossible).check(MODEL.facts, MODEL.revision)


def test_reject_incomplete_or_invalid_counts():
    m = replace(MODEL.facts, layers=2, experts=4, experts_used=2)
    for requests in (
        [],
        [[[1, 1, 0, 0]]],
        [[[1, 0, 0, 0]] * 2],
        [[[2, 0, 0, 0]] * 2],
        [[[1, 1, 0, 0], [2, 2, 0, 0]]],
        [[[True, 1, 0, 0]] * 2],
        [[[0, 0, 0, 0]] * 2],
    ):
        with pytest.raises(ValueError):
            heat_from_counts(m, "revision", requests, "synthetic")


@pytest.mark.parametrize("gpu", [True, False])
def test_optional_heat_preserves_baseline_and_uses_current_device_capacity(gpu):
    c, rt = setup()
    if not gpu:
        rt = replace(rt, gpus=())
    rt = replace(rt, memory_budget_bytes=32 * GIB)
    base = estimate(c, rt)
    h = hot()
    value = estimate(c, rt, expert_heat=h, model_revision=MODEL.revision)
    assert value.tokens_per_second == base.tokens_per_second
    assert value.detail == base.detail and value.scenario_range == base.scenario_range
    assert value.scenario_details == base.scenario_details
    assert all(value.scenarios[k] == v for k, v in base.scenarios.items())
    assert all(value.components[k] == v for k, v in base.components.items())
    assert value.scenarios["expert_heat_static_serial"] > base.tokens_per_second
    empty = estimate(
        c, replace(rt, gpus=(), memory_budget_bytes=0), expert_heat=h, model_revision=MODEL.revision
    )
    assert empty.components["expert_heat_gpu_hit_fraction"] == 0
    assert empty.components["expert_heat_storage_miss_fraction"] == 1
    assert empty.scenarios["expert_heat_static_serial"] == empty.tokens_per_second
    # Holding access frequencies fixed, slower storage still changes the forecast.
    slower = copy.deepcopy(c)
    for drive in slower["storage"]:
        for point in drive["random_read"]:
            point["Bps"] /= 2
    low = estimate(
        slower,
        replace(rt, gpus=(), memory_budget_bytes=0),
        expert_heat=h,
        model_revision=MODEL.revision,
    )
    assert low.scenarios["expert_heat_static_serial"] < empty.scenarios["expert_heat_static_serial"]


def test_heat_requires_revision_even_when_device_profile_is_incomplete():
    c, rt = setup()
    c["memory"]["dram_read_Bps"] = []
    with pytest.raises(ValueError, match="explicit model revision"):
        estimate(c, rt, expert_heat=hot())


def test_catalog_binding_json_and_text_output():
    c, rt = setup()
    rows = command.estimates(
        c, c["storage"][0]["drive_key"], "linux", lambda: rt, expert_heats=(hot(),)
    )
    first = rows[0][1]
    assert "expert_heat_static_serial" in first.scenarios
    assert all("expert_heat_static_serial" not in e.scenarios for _, e in rows[1:])
    assert "warm hottest-first" in command.estimate_lines(rows)[0]
    from pathlib import Path

    output = command.report(c, Path("profile.json"), c["storage"][0]["drive_key"], rows)
    assert output["estimates"][0]["reference_scenario"] == "uniform_serial"
    with pytest.raises(ValueError, match="duplicate"):
        command.matched_heats((hot(), hot()))
    with pytest.raises(ValueError, match="pinned catalog"):
        command.matched_heats((replace(hot(), model_revision="another-checkpoint"),))
    # Catalog Muse has a different quantization: never silently reuse Q4_K_S heat.
    assert catalog.MODELS[1].quantization != MODEL.quantization


def test_bad_cli_profile_fails_before_any_probe_or_file_write(tmp_path):
    host, files, build = machine(tmp_path)
    profile = tmp_path / "heat.json"
    profile.write_text(json.dumps(replace(hot(), model_revision="wrong").to_json()))
    runner = FakeRunner({})
    with pytest.raises(ValueError, match="pinned catalog"):
        execute(
            ["--model", files[0], "--build", str(build), "--expert-heat", str(profile)],
            host,
            runner,
            tmp_path,
        )
    assert runner.calls == []
    assert not (tmp_path / "state/inferssd/machine.json").exists()
