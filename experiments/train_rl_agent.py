"""Train and evaluate the online Arkanoid Q-learning policy on official physics."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from experiments.rl_agent import DEFAULT_Q_PATH, ReinforcementAgent
from experiments.strategic_agent import StrategicAgent
from server.config import load_config
from server.highscores import HighScoreManager
from server.logic import Arkanoid


ROOT = Path(__file__).resolve().parents[1]  # repository root (config.json, maps/)
DATA_DIR = Path(__file__).resolve().parent / "training_data"  # this folder's data
DEFAULT_RESULTS_PATH = DATA_DIR / "rl_training_results.json"


def _run_episode(
    level: int,
    seed: int,
    *,
    q_path: Path,
    agent_kind: str,
    epsilon: float = 0.0,
    learn: bool = False,
) -> dict[str, Any]:
    random.seed(seed)
    config = load_config(str(ROOT / "config.json"))
    config.game.level_sequence = [f"level{level}.json"]
    game = Arkanoid(config, HighScoreManager(None))

    if agent_kind == "rl":
        agent: ReinforcementAgent | StrategicAgent = ReinforcementAgent(
            "ws://localhost:8765/ws",
            "Arkanoid RL Training Agent",
            f"rl-{level}-{seed}",
            epsilon=epsilon,
            q_path=q_path,
            learn=learn,
        )
        agent._rng.seed(seed)
    elif agent_kind == "strategic":
        agent = StrategicAgent(
            "ws://localhost:8765/ws", "Strategic Reference Agent", f"st-{level}-{seed}"
        )
        agent._rng.seed(seed)
    else:
        raise ValueError(f"unknown agent kind: {agent_kind}")

    dt = 1.0 / config.game.fps
    while not game.game_over and not game.game_won:
        agent.current_state = game.get_state()
        decision = agent.deliberate()
        try:
            action = decision.send(None)
        except StopIteration as completed:
            action = completed.value
        else:
            decision.close()
            raise RuntimeError("the official game logic requires a non-blocking agent")
        if action and isinstance(action.get("direction"), str):
            game.move_paddle(action["direction"])
        game.update(dt)

    agent.current_state = game.get_state()
    if learn:
        # report() closes the last rebound transition and flushes the Q table.
        with redirect_stdout(StringIO()):
            agent.report()

    state = agent.current_state
    return {
        "level": level,
        "seed": seed,
        "score": int(state.get("score", 0)),
        "normalized_score": float(state.get("normalized_score", 0.0)),
        "elapsed_seconds": float(state.get("level_time_elapsed", 0.0)),
        "lives_lost": max(0, 3 - int(state.get("lives", 3))),
        "cleared": bool(state.get("game_won")),
        "timed_out": bool(state.get("timed_out")),
    }


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [float(row["normalized_score"]) for row in records]
    return {
        "games": len(records),
        "mean_normalized_score": round(statistics.fmean(rates), 4) if rates else 0.0,
        "clear_rate": round(
            sum(bool(row["cleared"]) for row in records) / len(records), 4
        ) if records else 0.0,
        "mean_lives_lost": round(
            statistics.fmean(int(row["lives_lost"]) for row in records), 4
        ) if records else 0.0,
        "timeouts": sum(bool(row["timed_out"]) for row in records),
    }


def _checkpoint_stats(path: Path) -> dict[str, int]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"states": 0, "state_action_visits": 0}
    q_values = checkpoint.get("q_values", {})
    visits = checkpoint.get("visits", {})
    state_action_visits = 0
    if isinstance(visits, dict):
        for actions in visits.values():
            if isinstance(actions, dict):
                state_action_visits += sum(
                    max(0, int(count)) for count in actions.values()
                    if isinstance(count, (int, float))
                )
    return {
        "states": len(q_values) if isinstance(q_values, dict) else 0,
        "state_action_visits": state_action_visits,
    }


def _paired_gain(
    rl_records: list[dict[str, Any]],
    strategic_records: list[dict[str, Any]],
) -> dict[str, Any]:
    strategic_by_seed = {int(row["seed"]): float(row["normalized_score"])
                         for row in strategic_records}
    differences = [
        float(row["normalized_score"]) - strategic_by_seed[int(row["seed"])]
        for row in rl_records
        if int(row["seed"]) in strategic_by_seed
    ]
    mean = statistics.fmean(differences) if differences else 0.0
    margin = (
        2.1 * statistics.stdev(differences) / len(differences) ** 0.5
        if len(differences) >= 20 else None
    )
    return {
        "paired_games": len(differences),
        "mean_rl_minus_strategic": round(mean, 4),
        "ci95": (
            [round(mean - margin, 4), round(mean + margin, 4)]
            if margin is not None else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Q-learning on the official Arkanoid physics and evaluate it"
    )
    parser.add_argument("--episodes", type=int, default=180,
                        help="total training games distributed across levels")
    parser.add_argument("--training-epsilon", type=float, default=0.15,
                        help="safe-action exploration rate while training")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="held-out evaluation games per level")
    parser.add_argument("--seed-start", type=int, default=20261116)
    parser.add_argument("--validation-seed-start", type=int, default=20262116)
    parser.add_argument("--q-table", type=Path, default=DEFAULT_Q_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_games < 1:
        parser.error("--episodes and --eval-games must be positive")
    if not 0.0 <= args.training_epsilon <= 1.0:
        parser.error("--training-epsilon must be between 0 and 1")

    initial_checkpoint = _checkpoint_stats(args.q_table)
    training: list[dict[str, Any]] = []
    for index in range(args.episodes):
        level = index % 3 + 1
        seed = args.seed_start + index
        record = _run_episode(
            level,
            seed,
            q_path=args.q_table,
            agent_kind="rl",
            epsilon=args.training_epsilon,
            learn=True,
        )
        training.append(record)
        print(
            f"train {index + 1}/{args.episodes} level={level} seed={seed} "
            f"rate={record['normalized_score']:.2f} cleared={record['cleared']} "
            f"lives_lost={record['lives_lost']}",
            flush=True,
        )

    checkpoint = json.loads(args.q_table.read_text(encoding="utf-8"))
    q_values = checkpoint.get("q_values", {})
    visits = checkpoint.get("visits", {})
    q_states = len(q_values) if isinstance(q_values, dict) else 0
    state_action_visits = sum(
        sum(actions.values())
        for actions in visits.values()
        if isinstance(actions, dict)
    ) if isinstance(visits, dict) else 0
    final_checkpoint = {
        "states": q_states,
        "state_action_visits": state_action_visits,
    }

    evaluation: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for level in (1, 2, 3):
        rl_records: list[dict[str, Any]] = []
        strategic_records: list[dict[str, Any]] = []
        for index in range(args.eval_games):
            seed = args.validation_seed_start + index
            rl_record = _run_episode(
                level,
                seed,
                q_path=args.q_table,
                agent_kind="rl",
                epsilon=0.0,
                learn=False,
            )
            strategic_record = _run_episode(
                level,
                seed,
                q_path=args.q_table,
                agent_kind="strategic",
            )
            rl_records.append(rl_record)
            strategic_records.append(strategic_record)
            evaluation.extend((rl_record | {"agent": "rl"},
                               strategic_record | {"agent": "strategic"}))
            print(
                f"eval level={level} seed={seed} "
                f"rl={rl_record['normalized_score']:.2f} "
                f"strategic={strategic_record['normalized_score']:.2f}",
                flush=True,
            )
        summaries.append({
            "level": level,
            "rl": _metrics(rl_records),
            "strategic": _metrics(strategic_records),
            "paired_gain": _paired_gain(rl_records, strategic_records),
        })

    result = {
        "training": {
            "episodes": args.episodes,
            "epsilon": args.training_epsilon,
            "seed_start": args.seed_start,
            "checkpoint_before": initial_checkpoint,
            "checkpoint_after": final_checkpoint,
            "games": training,
        },
        "checkpoint": str(args.q_table),
        "checkpoint_states": q_states,
        "state_action_visits": state_action_visits,
        "evaluation": {
            "games_per_level": args.eval_games,
            "seed_start": args.validation_seed_start,
            "exploration": 0.0,
            "summary": summaries,
            "games": evaluation,
        },
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint_states": q_states,
                      "state_action_visits": state_action_visits,
                      "evaluation": summaries}, indent=2))
    print(f"Checkpoint: {args.q_table}")
    print(f"Results: {args.results}")


if __name__ == "__main__":
    main()
