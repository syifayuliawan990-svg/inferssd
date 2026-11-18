# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete routing traces rather than silently skewing model heat."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_expert_heat import selection_counts
from inferssd.app import catalog


def events():
    return [
        dict(
            ev="ssd.place.layer",
            decode=True,
            has_token=True,
            token=token,
            layer=layer,
            selected_experts=[0, 3],
            wait_io_us=9999,
            vram_hits=2,
        )
        for token in (10, 11)
        for layer in (0, 1)
    ]


def run(tmp_path, rows):
    path = tmp_path / "native.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    model = replace(catalog.MODELS[0].facts, layers=2, experts=4, experts_used=2)
    return selection_counts(path, model, 2)


def test_counts_ignore_prefill_and_timing(tmp_path):
    rows = events()
    assert run(tmp_path, [{**rows[0], "decode": False}, *rows]) == [[2, 0, 0, 2]] * 2
    assert (
        run(
            tmp_path,
            [
                {k: v for k, v in r.items() if k not in ("wait_io_us", "vram_hits")}
                for r in rows
            ],
        )
        == [[2, 0, 0, 2]] * 2
    )


@pytest.mark.parametrize(
    "kind", ["duplicate", "missing", "short", "gap", "expert", "repeated-expert"]
)
def test_reject_unusable_trace(tmp_path, kind):
    rows = events()
    if kind == "duplicate":
        rows.append(rows[0])
    if kind == "missing":
        rows.pop()
    if kind == "short":
        rows = rows[:2]
    if kind == "gap":
        for r in rows[2:]:
            r["token"] += 1
    if kind == "expert":
        rows[0]["selected_experts"] = [0, 4]
    if kind == "repeated-expert":
        rows[0]["selected_experts"] = [0, 0]
    with pytest.raises(ValueError):
        run(tmp_path, rows)
