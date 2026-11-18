"""Tune the ExpectimaxAgent's scalar parameters on the exact engine.

Every configuration plays the same seeds (paired comparison), so the noise of
the random rebound angles mostly cancels between configurations. The search is
coordinate descent: each parameter in turn tries a few values around the
current best and keeps one only if it beats the incumbent on the screening
seeds. Results are cached in the log, so an interrupted run resumes.

    python tune_expectimax.py                       # screen and tune
    python tune_expectimax.py --validate RHO_SCALE=0.4 GAMMA=0.8
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from agents.expectimax_agent import ExpectimaxAgent
from server.config import load_config
from server.highscores import HighScoreManager
from server.logic import Arkanoid

DATA_DIR = Path(__file__).resolve().parent / "training_data"  # this folder's data

# Screening plays with an unlimited planning budget. ENDGAME_HITS and NEXT_BOARD_LIMIT change
# the search cost, so the final validation also runs under the live per-tick budget (--budgeted).
GRID = {
    "ENDGAME_HITS": [4, 6, 8],
    "RHO_SCALE": [0.2, 0.25, 0.3, 0.35, 0.45, 0.55],
    "GAMMA": [0.5, 0.6, 0.7, 0.8, 0.9],
    "HIT_CONTINUATION": [0.6, 0.8, 1.0, 1.2],
    "RHO_PRIOR": [0.2, 0.3, 0.45],
    "NEXT_BOARD_LIMIT": [6, 20],
}
LEVELS = (1, 2, 3)


def play(level: int, seed: int, params: dict[str, float], budgeted: bool = False) -> float:
    """Normalized score of one game on the server engine.

    ``budgeted`` plans under the live per-tick time limit, as on a 30 Hz server.
    """
    random.seed(seed)
    config = load_config()
    config.game.level_sequence = [f"level{level}.json"]
    config.game.level_time_limit = 0.0
    game = Arkanoid(config, HighScoreManager(None))
    agent = ExpectimaxAgent("ws://localhost:1/ws", "Tune", "tune", unlimited_budget=not budgeted)
    for name, value in params.items():
        setattr(agent, name, value)
    dt = 1.0 / config.game.fps
    while not game.game_over and not game.game_won:
        agent.current_state = game.get_state()
        agent._last_wall = None  # the server ticks every 1/30 s, however fast this loop runs
        decision = agent.deliberate()
        try:
            decision.send(None)
        except StopIteration as done:
            action = done.value
        else:  # pragma: no cover - deliberate never awaits
            raise RuntimeError("deliberate() must not suspend")
        if action:
            game.move_paddle(action["direction"])
        game.update(dt)
    return float(game.get_state().get("normalized_score", 0.0))


def _key(params: dict[str, float], budgeted: bool) -> str:
    changed = {k: v for k, v in sorted(params.items()) if v != getattr(ExpectimaxAgent, k)}
    return ("budgeted " if budgeted else "") + json.dumps(changed)


def _job(args: tuple[int, int, dict[str, float], bool]) -> tuple[int, int, float]:
    level, seed, params, budgeted = args
    return level, seed, play(level, seed, params, budgeted)


class Evaluator:
    def __init__(self, pool: ProcessPoolExecutor, log: Path, budgeted: bool = False) -> None:
        self.pool = pool
        self.budgeted = budgeted
        self.log = log
        self.cache: dict[str, dict[str, float]] = json.loads(log.read_text()) if log.exists() else {}

    def scores(self, params: dict[str, float], seeds: list[int]) -> dict[int, list[float]]:
        """Per-level scores of ``params`` on ``seeds``, playing only uncached games."""
        cached = self.cache.setdefault(_key(params, self.budgeted), {})
        todo = [(level, seed, params, self.budgeted) for level in LEVELS for seed in seeds if f"{level}:{seed}" not in cached]
        for level, seed, score in self.pool.map(_job, todo):
            cached[f"{level}:{seed}"] = score
        if todo:
            self.log.write_text(json.dumps(self.cache, indent=1))
        return {level: [cached[f"{level}:{seed}"] for seed in seeds] for level in LEVELS}


def mean_score(per_level: dict[int, list[float]]) -> float:
    return statistics.fmean(statistics.fmean(v) for v in per_level.values())


def paired_diff(a: dict[int, list[float]], b: dict[int, list[float]]) -> tuple[float, float]:
    """Mean and standard error of the level-averaged score difference a - b."""
    diffs = [statistics.fmean(a[lv][i] - b[lv][i] for lv in LEVELS) for i in range(len(a[LEVELS[0]]))]
    se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("inf")
    return statistics.fmean(diffs), se


def report(name: str, per_level: dict[int, list[float]]) -> None:
    levels = "  ".join(f"L{lv} {statistics.fmean(v):6.2f}" for lv, v in per_level.items())
    print(f"  {name:<52} {levels}  mean {mean_score(per_level):6.3f}", flush=True)


def defaults() -> dict[str, float]:
    return {name: getattr(ExpectimaxAgent, name) for name in GRID}


def tune(ev: Evaluator, seeds: list[int], rounds: int) -> dict[str, float]:
    best = defaults()
    best_scores = ev.scores(best, seeds)
    report(f"defaults {best}", best_scores)
    for round_ in range(rounds):
        changed = False
        for name, values in GRID.items():
            print(f"round {round_ + 1}, {name}", flush=True)
            for value in values:
                candidate = {**best, name: value}
                if candidate == best:
                    continue
                scores = ev.scores(candidate, seeds)
                diff, se = paired_diff(scores, best_scores)
                report(f"{name}={value}  diff {diff:+.3f} +- {se:.3f}", scores)
                # Require the gain to be clear of the paired noise, not just positive.
                if diff > se:
                    best, best_scores, changed = candidate, scores, True
            print(f"  -> {name}={best[name]}", flush=True)
        if not changed:
            break
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=16, help="screening seeds per level")
    parser.add_argument("--seed-start", type=int, default=80000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--validate", nargs="*", metavar="NAME=VALUE",
                        help="compare these overrides against the defaults on held-out seeds")
    parser.add_argument("--validate-seeds", type=int, default=44)
    parser.add_argument("--validate-start", type=int, default=90000)
    parser.add_argument("--budgeted", action="store_true",
                        help="validate under the live per-tick planning budget (use fewer workers than cores)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--log", type=Path, default=DATA_DIR / "expectimax_tuning.json")
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        ev = Evaluator(pool, args.log)
        if args.validate is not None:
            best = {**defaults(), **{k: json.loads(v) for k, v in (item.split("=") for item in args.validate)}}
        else:
            best = tune(ev, list(range(args.seed_start, args.seed_start + args.seeds)), args.rounds)
            print(f"best on screening seeds: {best}", flush=True)
        seeds = list(range(args.validate_start, args.validate_start + args.validate_seeds))
        ev.budgeted = args.budgeted
        print(f"validation on seeds {seeds[0]}-{seeds[-1]}" + (" (live budget)" if args.budgeted else ""), flush=True)
        base = ev.scores(defaults(), seeds)
        report("defaults", base)
        if best != defaults():
            tuned = ev.scores(best, seeds)
            report(f"tuned {best}", tuned)
            diff, se = paired_diff(tuned, base)
            print(f"  paired difference {diff:+.3f} +- {se:.3f} points/s", flush=True)


if __name__ == "__main__":
    main()
