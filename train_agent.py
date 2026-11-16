"""Generate reusable Arkanoid action-value priors from simulated rebounds.

Run with: python train_agent.py --simulations 1000000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from arkanoid_domain import ArkanoidSearchDomain, ArkanoidState, BrickState
from learn_policy import train_policy
from server.config import load_config


ROOT = Path(__file__).resolve().parent
DEFAULT_SIMULATIONS = 1_000_000
RNG_SEED = 20261116


def _brick_bucket(count: int) -> str:
    if count <= 10:
        return "0-10"
    if count <= 20:
        return "11-20"
    if count <= 30:
        return "21-30"
    return "31+"


def _landing_bucket(x: float, width: float) -> str:
    return str(min(2, max(0, int(3.0 * x / width))))


def _load_maps(domain: ArkanoidSearchDomain) -> list[tuple[int, tuple[BrickState, ...]]]:
    config = load_config(str(ROOT / "config.json"))
    points_by_type = {name: item.points for name, item in config.bricks.types.items()}
    maps: list[tuple[int, tuple[BrickState, ...]]] = []
    for level, filename in enumerate(config.game.level_sequence, start=1):
        path = ROOT / config.game.maps_dir / filename
        raw = json.loads(path.read_text(encoding="utf-8"))
        bricks = tuple(
            BrickState(
                x=float(item["x"]),
                y=float(item["y"]),
                width=float(item.get("width", raw.get("brick_width", config.bricks.width))),
                height=float(item.get("height", raw.get("brick_height", config.bricks.height))),
                health=int(item.get("health", config.bricks.types[item["type"]].health)),
                points=0 if bool(item.get("indestructible", False))
                else int(points_by_type.get(item["type"], 0)),
                indestructible=bool(item.get("indestructible", False)),
            )
            for item in raw.get("bricks", [])
        )
        maps.append((level, bricks))
    return maps


def _make_domain() -> ArkanoidSearchDomain:
    config = load_config(str(ROOT / "config.json"))
    return ArkanoidSearchDomain(
        width=float(config.screen.width),
        height=float(config.screen.height),
        ceiling_y=config.screen.ceiling_y,
        paddle_y=config.paddle.y,
        paddle_width=config.paddle.width,
        paddle_speed=config.paddle.speed,
        ball_radius=config.ball.radius,
        ball_speed=config.ball.speed,
        fps=config.game.fps,
        paddle_step_distance=config.paddle.step_distance,
        paddle_left_ratio=config.paddle.left_ratio,
        paddle_right_ratio=config.paddle.right_ratio,
        center_angle_half_width=config.paddle.angles.alpha_deg,
        side_angle=config.paddle.angles.side_deg,
        side_angle_half_width=config.paddle.angles.beta_deg,
    )


def _sample_board(
    bricks: tuple[BrickState, ...], rng: random.Random
) -> tuple[BrickState, ...]:
    """Create a plausible mid-game board by simulating earlier brick damage."""
    progress = rng.betavariate(1.35, 2.2)
    remaining: list[BrickState] = []
    for brick in bricks:
        if brick.indestructible:
            remaining.append(brick)
            continue
        if rng.random() < progress:
            continue
        health = brick.health
        if health > 1 and rng.random() < progress * 0.55:
            health = max(1, health - 1)
        remaining.append(BrickState(
            brick.x, brick.y, brick.width, brick.height,
            health, brick.points, brick.indestructible,
        ))
    return tuple(remaining)


def _brick_rows(state: ArkanoidState, radius: float) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = {}
    for index, brick in enumerate(state.bricks):
        if brick.health <= 0 and not brick.indestructible:
            continue
        first = int((brick.y - radius) // 24.0)
        last = int((brick.y + brick.height + radius) // 24.0)
        for row in range(first, last + 1):
            rows.setdefault(row, []).append(index)
    return rows


def _contact_interval(domain: ArkanoidSearchDomain, action: str) -> tuple[float, float]:
    left = domain.paddle_width * domain.paddle_left_ratio
    right = domain.paddle_width * (1.0 - domain.paddle_right_ratio)
    pad = domain.ball_radius + 1.0
    if action == domain.LEFT:
        return pad, max(pad, left - 1.0)
    if action == domain.CENTER:
        return left + 1.0, max(left + 1.0, right - 1.0)
    return right + 1.0, max(right + 1.0, domain.paddle_width - pad)


def _profile_keys(
    level: int,
    brick_bucket: str,
    landing_bucket: str,
    action: str,
) -> tuple[str, ...]:
    return (
        f"{level}|{brick_bucket}|{landing_bucket}|{action}",
        f"{level}|{brick_bucket}|*|{action}",
        f"{level}|*|{landing_bucket}|{action}",
        f"{level}|*|*|{action}",
        f"*|{brick_bucket}|*|{action}",
        f"*|*|*|{action}",
    )


def train(simulations: int, output_dir: Path) -> dict[str, Any]:
    if simulations < 1_000_000:
        raise ValueError("At least 1,000,000 simulations are required")

    rng = random.Random(RNG_SEED)
    domain = _make_domain()
    maps = _load_maps(domain)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"rollouts_{simulations}.csv.gz"
    profile_path = output_dir / "learned_policy.json"

    fields = (
        "simulation", "level", "remaining_bricks", "brick_bucket",
        "landing_x", "landing_bucket", "paddle_x", "contact_x", "action",
        "angle_deg", "inbound_seconds", "points", "flight_seconds",
        "points_per_second",
    )
    aggregates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    total_points = 0
    total_seconds = 0.0

    with gzip.open(csv_path, "wt", newline="", encoding="utf-8", compresslevel=1) as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for simulation in range(simulations):
            level, original_bricks = rng.choice(maps)
            bricks = _sample_board(original_bricks, rng)
            action = rng.choice(domain.ACTIONS)
            contact_low, contact_high = _contact_interval(domain, action)
            contact_x = rng.uniform(contact_low, contact_high)
            max_paddle_x = domain.width - domain.paddle_width
            paddle_low = max(0.0, domain.ball_radius - contact_x)
            paddle_high = min(
                max_paddle_x,
                domain.width - domain.ball_radius - contact_x,
            )
            paddle_x = rng.uniform(paddle_low, max(paddle_low, paddle_high))
            landing_x = paddle_x + contact_x
            inbound_seconds = rng.uniform(0.15, 2.0)
            state = ArkanoidState(
                ball_x=landing_x,
                ball_y=domain.paddle_y - domain.ball_radius,
                ball_vx=0.0,
                ball_vy=domain.ball_speed,
                paddle_x=paddle_x,
                bricks=bricks,
                lives=3,
            )
            rows = _brick_rows(state, domain.ball_radius)
            mode, half_width = domain.angle_parameters(action)
            angle = rng.triangular(mode - half_width, mode + half_width, mode)
            points, flight_seconds = domain._simulate_shot_details(
                state, landing_x, angle, rows, 24.0
            )
            duration = max(0.05, inbound_seconds + flight_seconds)
            points_per_second = points / duration
            count = state.remaining_bricks
            bucket = _brick_bucket(count)
            landing_group = _landing_bucket(landing_x, domain.width)
            for key in _profile_keys(level, bucket, landing_group, action):
                aggregate = aggregates[key]
                aggregate[0] += 1
                aggregate[1] += points
                aggregate[2] += duration

            total_points += points
            total_seconds += duration
            writer.writerow((
                simulation + 1,
                level,
                count,
                bucket,
                round(landing_x, 3),
                landing_group,
                round(paddle_x, 3),
                round(contact_x, 3),
                action,
                round(angle, 6),
                round(inbound_seconds, 6),
                points,
                round(flight_seconds, 6),
                round(points_per_second, 6),
            ))
            if (simulation + 1) % 100_000 == 0:
                print(f"Simulated {simulation + 1:,}/{simulations:,} rebounds", flush=True)

    cells = {
        key: {
            "samples": int(values[0]),
            "mean_points_per_second": values[1] / max(0.05, values[2]),
        }
        for key, values in aggregates.items()
    }
    overall_points_per_second = total_points / max(0.05, total_seconds)
    profile: dict[str, Any] = {
        "version": 2,
        "simulations": simulations,
        "seed": RNG_SEED,
        "average_points_per_second": overall_points_per_second,
        "objective": "total_points / total_elapsed_seconds",
        "features": ["level", "remaining_brick_bucket", "landing_third", "action"],
        "cells": cells,
    }
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return {
        "simulations": simulations,
        "average_points_per_second": overall_points_per_second,
        "profile_cells": len(cells),
        "csv_path": str(csv_path),
        "profile_path": str(profile_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Arkanoid action priors by simulation")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "training_data"
    )
    args = parser.parse_args()
    summary = train(args.simulations, args.output_dir)
    summary["learned_policy"] = train_policy(
        Path(summary["csv_path"]), Path(summary["profile_path"])
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
