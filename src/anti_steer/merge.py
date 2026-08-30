#!/usr/bin/env python3
"""Merge anti-steering evaluation shards and recompute aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMPATIBILITY_KEYS = (
    "model",
    "dataset",
    "condition",
    "max_new_tokens",
    "dtype",
    "layer",
    "suppressed_features",
    "suppression_mode",
    "suppression_strength",
    "inference_engine",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    reference = payloads[0]
    for path, payload in zip(args.inputs[1:], payloads[1:]):
        mismatches = [
            key
            for key in COMPATIBILITY_KEYS
            if payload.get(key) != reference.get(key)
        ]
        if mismatches:
            raise ValueError(f"Incompatible shard {path}: {', '.join(mismatches)}")

    results = sorted(
        [row for payload in payloads for row in payload["results"]],
        key=lambda row: row["question_idx"],
    )
    if not results:
        raise ValueError("No results to merge")
    indices = [row["question_idx"] for row in results]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate question indices across shards")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Question indices do not form a contiguous range")

    merged = {key: value for key, value in reference.items() if key != "results"}
    merged["samples"] = len(results)
    merged["target_samples"] = len(results)
    merged["dataset_range"] = [indices[0], indices[-1] + 1]
    merged["correct_count"] = sum(row["correct"] for row in results)
    merged["accuracy"] = merged["correct_count"] / len(results)
    merged["mean_num_generated_tokens"] = (
        sum(row["num_generated_tokens"] for row in results) / len(results)
    )
    batch_sizes = {
        payload.get("batch_size") for payload in payloads if payload.get("batch_size") is not None
    }
    if batch_sizes:
        merged["batch_sizes"] = sorted(batch_sizes)
    for key in (
        "hook_calls",
        "hook_processed_rows",
        "hook_rows_with_any_selected_feature_active",
        "hook_rows_with_target_in_sae_topk",
    ):
        merged[key] = sum(payload.get(key, 0) for payload in payloads)
    merged["results"] = results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({k: v for k, v in merged.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
