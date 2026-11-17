"""Fit and validate a contextual action policy from saved Arkanoid rollouts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parent
MIN_ACTION_SAMPLES = 64
TARGET_POINTS_PER_SECOND = 16.0


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def _context_keys(level: str, density: str, landing: str) -> tuple[str, ...]:
    return (
        f"{level}|{density}|{landing}",
        f"{level}|{density}|*",
        f"{level}|*|{landing}",
        f"{level}|*|*",
        f"*|{density}|*",
        "*|*|*",
    )


def _rate(values: list[float]) -> float:
    return values[1] / max(0.05, values[2])


def train_policy(
    rollouts_paths: list[Path] | Path,
    profile_path: Path,
) -> dict[str, Any]:
    """Learn action values on 80% of samples and report holdout performance."""
    training: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0, 0.0])
    )
    validation: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0, 0.0])
    )
    if isinstance(rollouts_paths, Path):
        paths = [rollouts_paths]
    else:
        paths = list(rollouts_paths)
    total_rows = train_rows = validation_rows = 0
    level_one_rows = level_one_loop_labeled_rows = level_one_loops = 0

    for path in paths:
        with _open_csv(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                total_rows += 1
                level = row["level"]
                density = row["brick_bucket"]
                landing = row["landing_bucket"]
                action = row["action"]
                points = float(row["points"])
                duration = max(
                    0.05,
                    float(row["inbound_seconds"]) + float(row["flight_seconds"]),
                )
                if level == "1":
                    level_one_rows += 1
                    if row.get("scoring_upper_loop") not in (None, ""):
                        level_one_loop_labeled_rows += 1
                        level_one_loops += int(row["scoring_upper_loop"] or 0)
                target = training if total_rows % 5 else validation
                if target is training:
                    train_rows += 1
                else:
                    validation_rows += 1
                for key in _context_keys(level, density, landing):
                    values = target[key][action]
                    values[0] += 1
                    values[1] += points
                    values[2] += duration

    cells: dict[str, dict[str, float | int]] = {}
    for context, actions in training.items():
        for action, values in actions.items():
            cells[f"{context}|{action}"] = {
                "samples": int(values[0]),
                "mean_points_per_second": _rate(values),
            }

    policy: dict[str, dict[str, Any]] = {}
    for context, actions in training.items():
        qualified = {
            action: values
            for action, values in actions.items()
            if values[0] >= MIN_ACTION_SAMPLES
        }
        if not qualified:
            continue
        best_action, best_values = max(
            qualified.items(),
            key=lambda item: (_rate(item[1]), item[1][0], item[0] == "CENTER"),
        )
        policy[context] = {
            "action": best_action,
            "samples": int(best_values[0]),
            "mean_points_per_second": _rate(best_values),
            "action_values": {
                action: {
                    "samples": int(values[0]),
                    "mean_points_per_second": _rate(values),
                }
                for action, values in qualified.items()
            },
        }

    validation_total = [0.0, 0.0, 0.0]
    validation_policy = [0.0, 0.0, 0.0]
    covered_contexts = 0
    target_contexts = 0
    for context, actions in validation.items():
        if "|*|*" in context:
            continue
        keys = context.split("|")
        if len(keys) != 3 or "*" in keys:
            continue
        selected_policy: dict[str, Any] | None = None
        for key in _context_keys(keys[0], keys[1], keys[2]):
            selected_policy = policy.get(key)
            if selected_policy is not None:
                break
        if selected_policy is None:
            continue
        covered_contexts += 1
        for values in actions.values():
            validation_total[1] += values[1]
            validation_total[2] += values[2]
        selected_values = actions.get(str(selected_policy["action"]))
        if selected_values is not None:
            validation_policy[0] += selected_values[0]
            validation_policy[1] += selected_values[1]
            validation_policy[2] += selected_values[2]
            if _rate(selected_values) >= TARGET_POINTS_PER_SECOND:
                target_contexts += 1

    random_rate = validation_total[1] / max(0.05, validation_total[2])
    learned_rate = validation_policy[1] / max(0.05, validation_policy[2])
    lift = (learned_rate / random_rate - 1.0) if random_rate > 0 else 0.0
    profile: dict[str, Any] = {
        "version": 3,
        "simulations": total_rows,
        "rollout_files": [str(path) for path in paths],
        "training_samples": train_rows,
        "validation_samples": validation_rows,
        "level_one_samples": level_one_rows,
        "level_one_loop_labeled_samples": level_one_loop_labeled_rows,
        "level_one_scoring_loops": level_one_loops,
        "minimum_action_samples": MIN_ACTION_SAMPLES,
        "objective": "total_points / total_elapsed_seconds",
        "features": ["level", "remaining_brick_bucket", "landing_third"],
        "validation": {
            "covered_contexts": covered_contexts,
            "random_points_per_second": random_rate,
            "learned_points_per_second": learned_rate,
            "relative_lift": lift,
            "target_points_per_second": TARGET_POINTS_PER_SECOND,
            "contexts_meeting_target": target_contexts,
            "context_target_rate": (
                target_contexts / covered_contexts if covered_contexts else 0.0
            ),
        },
        "cells": cells,
        "policy": policy,
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile["validation"] | {
        "simulations": total_rows,
        "training_samples": train_rows,
        "validation_samples": validation_rows,
        "level_one_samples": level_one_rows,
        "level_one_loop_labeled_samples": level_one_loop_labeled_rows,
        "level_one_scoring_loops": level_one_loops,
        "policy_contexts": len(policy),
        "target_points_per_second": TARGET_POINTS_PER_SECOND,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a score-rate policy from saved rollouts")
    parser.add_argument(
        "--rollouts", type=Path, nargs="+",
        default=[ROOT / "training_data" / "rollouts_1000000.csv.gz"],
    )
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "training_data" / "learned_policy.json",
    )
    args = parser.parse_args()
    print(json.dumps(train_policy(args.rollouts, args.profile), indent=2))


if __name__ == "__main__":
    main()
