# SPDX-License-Identifier: Apache-2.0
"""Export model-side decode access marginals from single-request native traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferssd.strategy.heat import heat_from_counts
from inferssd.strategy.inputs import ModelFacts, model_facts_from


def selection_counts(path: Path, model: ModelFacts, steps: int) -> list[list[int]]:
    """Require exactly STEPS contiguous token steps with every layer represented.

    Each input file must cover one request. Only decode placement events with
    a token ID are counted. No timing, measured cache hits or device fields are
    read into the artifact. Duplicate or incomplete events are errors.
    """
    if steps < 1:
        raise ValueError("decode steps must be positive")
    counts = [[0] * model.experts for _ in range(model.layers)]
    seen: set[tuple[int, int]] = set()
    tokens: set[int] = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("ev") != "ssd.place.layer"
                or not row.get("decode")
                or not row.get("has_token")
            ):
                continue
            token, layer, experts = (
                row.get("token"),
                row.get("layer"),
                row.get("selected_experts"),
            )
            if (
                type(token) is not int
                or type(layer) is not int
                or not 0 <= layer < model.layers
            ):
                raise ValueError("invalid token or routed layer index")
            if (token, layer) in seen:
                raise ValueError(
                    "duplicate token/layer event; provide one request per file"
                )
            if (
                not isinstance(experts, list)
                or len(experts) != model.experts_used
                or any(
                    type(e) is not int or not 0 <= e < model.experts for e in experts
                )
                or len(set(experts)) != len(experts)
            ):
                raise ValueError("invalid distinct top-k expert selections")
            seen.add((token, layer))
            tokens.add(token)
            for expert in experts:
                counts[layer][expert] += 1
    if (
        len(tokens) != steps
        or len(seen) != steps * model.layers
        or max(tokens, default=0) - min(tokens, default=0) + 1 != steps
    ):
        raise ValueError(
            "trace must contain the requested complete contiguous decode window"
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-facts",
        type=Path,
        required=True,
        help="model_facts_json from the exact quantized model headers",
    )
    parser.add_argument(
        "--model-revision", required=True, help="pinned checkpoint revision"
    )
    parser.add_argument(
        "--workload", required=True, help="corpus, prompts and generation settings"
    )
    parser.add_argument("--decode-steps", type=int, default=255)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "traces", nargs="+", type=Path, help="one native-events JSONL per request"
    )
    args = parser.parse_args()
    if len({path.resolve() for path in args.traces}) != len(args.traces):
        parser.error("duplicate trace input")
    model = model_facts_from(json.loads(args.model_facts.read_text()))
    heat = heat_from_counts(
        model,
        args.model_revision,
        (selection_counts(p, model, args.decode_steps) for p in args.traces),
        f"{args.workload}; {args.decode_steps} decode steps per request",
    )
    args.output.write_text(json.dumps(heat.to_json(), indent=2) + "\n")


if __name__ == "__main__":
    main()
