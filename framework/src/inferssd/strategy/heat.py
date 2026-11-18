# SPDX-License-Identifier: Apache-2.0
"""Portable expert-access marginals, not device hit rates or timing calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from inferssd.strategy.inputs import ModelFacts, model_facts_json


def fingerprint(model: ModelFacts) -> str:
    """Bind the quantized tensor geometry, excluding machine-local file paths.

    Geometry alone does not identify weights; ExpertHeat also requires an exact
    caller-supplied checkpoint revision (the catalog uses pinned HF revisions).
    """
    value = model_facts_json(model)
    del value["paths"]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ExpertHeat:
    model_revision: str
    model_fingerprint: str
    probabilities: tuple[tuple[float, ...], ...]
    requests: int
    workload: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise ValueError("expert heat needs a pinned model revision")
        if (
            not isinstance(self.model_fingerprint, str)
            or len(self.model_fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in self.model_fingerprint)
        ):
            raise ValueError("expert heat needs a SHA256 model fingerprint")
        if type(self.requests) is not int or self.requests < 1:
            raise ValueError("expert heat requests must be a positive integer")
        if not isinstance(self.workload, str) or not self.workload.strip():
            raise ValueError("expert heat needs a workload description")
        if not self.probabilities or not self.probabilities[0]:
            raise ValueError("expert heat needs a nonempty layer/expert matrix")
        width = len(self.probabilities[0])
        for row in self.probabilities:
            if len(row) != width or any(
                type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in row
            ):
                raise ValueError("expert heat probabilities must be finite and nonnegative")
            if not math.isclose(math.fsum(row), 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError("each expert heat layer must sum to one")

    def check(self, model: ModelFacts, revision: str) -> None:
        if self.model_revision != revision or self.model_fingerprint != fingerprint(model):
            raise ValueError("expert heat does not match the model revision and quantization")
        if len(self.probabilities) != model.layers or len(self.probabilities[0]) != model.experts:
            raise ValueError("expert heat dimensions do not match the model")
        if not model.experts_used or any(
            p > 1 / model.experts_used + 1e-9 for row in self.probabilities for p in row
        ):
            raise ValueError("expert heat exceeds distinct top-k selection frequency")

    def coverage(self, slots: int) -> float:
        """Traffic covered by a warm static hottest set; equal layer weight.

        This is not a cache simulation: no ordering, admissions, prefetch or
        warm-up is modeled. Capacity is supplied by the current device model.
        """
        if slots < 0:
            raise ValueError("expert heat slot count must be nonnegative")
        ranked = sorted((p for row in self.probabilities for p in row), reverse=True)
        return min(1.0, math.fsum(ranked[:slots]) / len(self.probabilities))

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "inferssd-expert-heat/1",
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
            "probabilities": [list(row) for row in self.probabilities],
            "requests": self.requests,
            "workload": self.workload,
        }


def heat_from_json(value: object) -> ExpertHeat:
    """Read a strict artifact; reject timing/hardware fields rather than ignore them."""
    keys = {
        "schema",
        "model_revision",
        "model_fingerprint",
        "probabilities",
        "requests",
        "workload",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("invalid expert heat fields")
    if value["schema"] != "inferssd-expert-heat/1":
        raise ValueError("unsupported expert heat schema")
    rows = value["probabilities"]
    if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
        raise ValueError("expert heat probabilities must be an array of arrays")
    return ExpertHeat(
        value["model_revision"],
        value["model_fingerprint"],
        tuple(tuple(row) for row in rows),
        value["requests"],
        value["workload"],
    )


def heat_from_counts(
    model: ModelFacts,
    revision: str,
    requests: Iterable[Sequence[Sequence[int]]],
    workload: str,
) -> ExpertHeat:
    """Average normalized decode selection counts, giving each request equal weight.

    Supply one [layer][expert] count matrix per request, over the same declared
    output-token window. Prompt/prefill and padding events must be excluded by
    the caller. Every layer must cover the same number of routed token steps.
    """
    sums = [[0.0] * model.experts for _ in range(model.layers)]
    n = 0
    if not model.experts_used:
        raise ValueError("expert heat requires routed experts")
    for counts in requests:
        if len(counts) != model.layers or any(len(row) != model.experts for row in counts):
            raise ValueError("selection count dimensions do not match the model")
        if any(type(c) is not int or c < 0 for row in counts for c in row):
            raise ValueError("selection counts must be nonnegative integers")
        totals = [sum(row) for row in counts]
        if not totals[0] or len(set(totals)) != 1 or totals[0] % model.experts_used:
            raise ValueError("selection counts need complete, equally covered decode layers")
        steps = totals[0] // model.experts_used
        if any(c > steps for row in counts for c in row):
            raise ValueError("an expert can be selected at most once per decode step")
        for total, row, acc in zip(totals, counts, sums):
            for e, count in enumerate(row):
                acc[e] += count / total
        n += 1
    if not n:
        raise ValueError("expert heat needs at least one request")
    result = ExpertHeat(
        revision, fingerprint(model), tuple(tuple(p / n for p in row) for row in sums), n, workload
    )
    result.check(model, revision)
    return result
