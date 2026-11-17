"""Run paired, seeded benchmarks against the official Arkanoid server.

Each episode uses the server's ``Arkanoid`` engine for one official map. Live
mode starts a fresh server with ``--levels N``; logic mode runs the same engine
in-process. The baseline follows the observed ball; the student agent uses
sampled, angle-aware trajectory planning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from agents.dummy_agent import DummyAgent
from agents.strategic_agent import StrategicAgent
from arkanoid_domain import ArkanoidSearchDomain, ArkanoidState
from server.config import load_config
from server.highscores import HighScoreManager
from server.logic import Arkanoid


ROOT = Path(__file__).resolve().parent
SERVER_WORKER = r"""
import asyncio, random, sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
random.seed(int(sys.argv[1]))
sys.argv = ["server.server", *sys.argv[2:]]
from server.server import main
main()
"""


class ProfiledStudentAgent(StrategicAgent):
    """Record synchronous decision cost without changing agent actions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decision_times_ms: list[float] = []
        self.planning_times_ms: list[float] = []
        self.worst_planning_context: dict[str, Any] = {}

    async def deliberate(self) -> dict[str, Any] | None:
        started = time.perf_counter_ns()
        action = await super().deliberate()
        self.decision_times_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        return action

    def _choose_target(
        self,
        domain: ArkanoidSearchDomain,
        state: ArkanoidState,
        observation: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        started = time.perf_counter_ns()
        target = super()._choose_target(domain, state, observation)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self.planning_times_ms.append(elapsed_ms)
        if elapsed_ms > float(self.worst_planning_context.get("elapsed_ms", 0.0)):
            self.worst_planning_context = {
                "elapsed_ms": round(elapsed_ms, 4),
                "remaining_bricks": state.remaining_bricks,
                "ball_x": round(state.ball_x, 2),
                "ball_y": round(state.ball_y, 2),
                "ball_vx": round(state.ball_vx, 2),
                "ball_vy": round(state.ball_vy, 2),
                "level_time_elapsed": float((observation or {}).get("level_time_elapsed", 0.0)),
            }
        return target

    def report(self) -> None:
        """Keep one-line server logs; episode metrics are printed by this script."""


class QuietDummyAgent(DummyAgent):
    """Baseline agent without the normal end-of-game console report."""

    def report(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until_listening(process: subprocess.Popen[bytes], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before accepting connections (code {process.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"server did not listen on port {port} within {timeout:.1f}s")


def _server_log_excerpt(logs: Any) -> str:
    logs.flush()
    logs.seek(0)
    return logs.read().decode("utf-8", errors="replace")[-4000:]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


async def _run_agent(agent: BaseAgent, timeout: float) -> None:
    await asyncio.wait_for(agent.run(), timeout=timeout)


def run_episode(
    agent_name: str,
    level: int,
    seed: int,
    acceleration: float,
    level_time_limit: float,
    startup_timeout: float,
    episode_timeout: float,
) -> dict[str, Any]:
    """Start the real server for one seeded level and collect terminal metrics."""
    port = _free_port()
    logs = tempfile.TemporaryFile()
    server_args = [
        sys.executable,
        "-c",
        SERVER_WORKER,
        str(seed),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--levels", str(level),
        "--acceleration", str(acceleration),
        "--level-time-limit", str(level_time_limit),
    ]
    server = subprocess.Popen(
        server_args,
        cwd=ROOT,
        stdout=logs,
        stderr=subprocess.STDOUT,
    )
    agent: BaseAgent | None = None
    try:
        _wait_until_listening(server, port, startup_timeout)
        uri = f"ws://127.0.0.1:{port}/ws"
        if agent_name == "student":
            agent = ProfiledStudentAgent(uri, "Student Search Agent", f"eval-{level}-{seed}")
            agent._rng.seed(seed)
        elif agent_name == "baseline":
            agent = QuietDummyAgent(uri, "Follow Ball Baseline", f"base-{level}-{seed}")
        else:
            raise ValueError(f"unknown agent: {agent_name}")

        try:
            asyncio.run(_run_agent(agent, episode_timeout))
        except asyncio.TimeoutError as error:
            state = agent.current_state or {}
            decision_times = getattr(agent, "decision_times_ms", [])
            planning_times = getattr(agent, "planning_times_ms", [])
            partial_elapsed = float(state.get("level_time_elapsed", state.get("time_elapsed", 0.0)))
            partial_score = int(state.get("score", 0))
            return {
                "agent": agent_name,
                "level": level,
                "seed": seed,
                "score": partial_score,
                "elapsed_seconds": round(partial_elapsed, 3),
                # An unfinished runner-timeout is not a cleared level for the
                # official metric; retain partial raw rate separately below.
                "normalized_score": 0.0,
                "raw_points_per_second": round(partial_score / partial_elapsed, 4)
                if partial_elapsed > 0.0 else 0.0,
                "lives_lost": max(0, 3 - int(state.get("lives", 3))),
                "cleared": False,
                "timed_out": bool(state.get("timed_out")),
                "runner_timeout": True,
                "error": str(error) or f"episode exceeded {episode_timeout:.1f}s without a terminal game state",
                "agent_stop_reason": agent.stop_reason,
                "active_bricks": len(state.get("bricks", [])),
                "decision_count": len(decision_times),
                "decision_p95_ms": round(_percentile(decision_times, 0.95), 4),
                "decision_max_ms": round(max(decision_times, default=0.0), 4),
                "planning_p95_ms": round(_percentile(planning_times, 0.95), 4),
                "planning_max_ms": round(max(planning_times, default=0.0), 4),
                "_decision_times_ms": decision_times,
                "_planning_times_ms": planning_times,
                "planning_worst_context": getattr(agent, "worst_planning_context", {}),
                "server_log": _server_log_excerpt(logs),
            }
        state = agent.current_state or {}
        if not state.get("game_over") and not state.get("game_won"):
            raise RuntimeError(f"agent disconnected before a terminal game state: {agent.stop_reason}")

        elapsed = float(state.get("level_time_elapsed", state.get("time_elapsed", 0.0)))
        score = int(state.get("score", 0))
        cleared = bool(state.get("game_won"))
        raw_rate = score / elapsed if elapsed > 0.0 else 0.0
        graded_rate = float(state.get("normalized_score", 0.0))
        decision_times = getattr(agent, "decision_times_ms", [])
        return {
            "agent": agent_name,
            "level": level,
            "seed": seed,
            "score": score,
            "elapsed_seconds": round(elapsed, 3),
            "normalized_score": round(graded_rate, 4),
            "raw_points_per_second": round(raw_rate, 4),
            "lives_lost": max(0, 3 - int(state.get("lives", 3))),
            "cleared": cleared,
            "timed_out": bool(state.get("timed_out")),
            "decision_count": len(decision_times),
            "decision_p95_ms": round(_percentile(decision_times, 0.95), 4),
            "decision_max_ms": round(max(decision_times, default=0.0), 4),
            "planning_p95_ms": round(_percentile(agent.planning_times_ms, 0.95), 4)
            if isinstance(agent, ProfiledStudentAgent) else 0.0,
            "planning_max_ms": round(max(agent.planning_times_ms, default=0.0), 4)
            if isinstance(agent, ProfiledStudentAgent) else 0.0,
            "_decision_times_ms": decision_times,
            "_planning_times_ms": getattr(agent, "planning_times_ms", []),
            "planning_worst_context": getattr(agent, "worst_planning_context", {}),
        }
    except Exception as error:
        excerpt = _server_log_excerpt(logs)
        raise RuntimeError(f"{error}\nServer log:\n{excerpt}") from error
    finally:
        server.terminate()
        try:
            server.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        logs.close()


def run_logic_episode(
    agent_name: str,
    level: int,
    seed: int,
    level_time_limit: float,
) -> dict[str, Any]:
    """Benchmark against the server's own physics in-process, without sockets."""
    random.seed(seed)
    config = load_config()
    config.game.level_sequence = [f"level{level}.json"]
    config.game.level_time_limit = level_time_limit
    game = Arkanoid(config, HighScoreManager(None))
    if agent_name == "student":
        student_agent = ProfiledStudentAgent(
            "ws://localhost:8765/ws", "Student Search Agent", f"eval-{level}-{seed}"
        )
        student_agent._rng.seed(seed)
        agent: BaseAgent = student_agent
    elif agent_name == "baseline":
        agent = QuietDummyAgent("ws://localhost:8765/ws", "Follow Ball Baseline", f"base-{level}-{seed}")
    else:
        raise ValueError(f"unknown agent: {agent_name}")

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
            raise RuntimeError("the local game logic requires a non-blocking deliberate() method")
        if action:
            direction = action.get("direction")
            if isinstance(direction, str):
                game.move_paddle(direction)
        game.update(dt)

    state = game.get_state()
    elapsed = float(state.get("level_time_elapsed", 0.0))
    score = int(state.get("score", 0))
    raw_rate = score / elapsed if elapsed > 0.0 else 0.0
    decision_times = getattr(agent, "decision_times_ms", [])
    return {
        "agent": agent_name,
        "level": level,
        "seed": seed,
        "score": score,
        "elapsed_seconds": round(elapsed, 3),
        "normalized_score": round(float(state.get("normalized_score", 0.0)), 4),
        "raw_points_per_second": round(raw_rate, 4),
        "lives_lost": max(0, 3 - int(state.get("lives", 3))),
        "cleared": bool(state.get("game_won")),
        "timed_out": bool(state.get("timed_out")),
        "decision_count": len(decision_times),
        "decision_p95_ms": round(_percentile(decision_times, 0.95), 4),
        "decision_max_ms": round(max(decision_times, default=0.0), 4),
        "planning_p95_ms": round(_percentile(agent.planning_times_ms, 0.95), 4)
        if isinstance(agent, ProfiledStudentAgent) else 0.0,
        "planning_max_ms": round(max(agent.planning_times_ms, default=0.0), 4)
        if isinstance(agent, ProfiledStudentAgent) else 0.0,
        "_decision_times_ms": decision_times,
        "_planning_times_ms": getattr(agent, "planning_times_ms", []),
        "planning_worst_context": getattr(agent, "worst_planning_context", {}),
    }


def _summary(
    records: list[dict[str, Any]],
    agent_name: str,
    level: int,
    decision_latencies: list[float],
    planning_latencies: list[float],
) -> dict[str, Any]:
    rows = [row for row in records if row["agent"] == agent_name and row["level"] == level]
    rates = [float(row["normalized_score"]) for row in rows]
    lives = [int(row["lives_lost"]) for row in rows]
    return {
        "agent": agent_name,
        "level": level,
        "games": len(rows),
        "mean_normalized_score": round(statistics.fmean(rates), 4) if rates else 0.0,
        "std_normalized_score": round(statistics.stdev(rates), 4) if len(rates) > 1 else 0.0,
        "clear_rate": round(sum(bool(row["cleared"]) for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_lives_lost": round(statistics.fmean(lives), 4) if lives else 0.0,
        "timed_out_games": sum(bool(row["timed_out"]) for row in rows),
        "runner_timeouts": sum(bool(row.get("runner_timeout")) for row in rows),
        "tick_decision_p95_ms": round(_percentile(decision_latencies, 0.95), 4),
        "tick_decision_max_ms": round(max(decision_latencies, default=0.0), 4),
        "planning_p95_ms": round(_percentile(planning_latencies, 0.95), 4),
        "planning_max_ms": round(max(planning_latencies, default=0.0), 4),
    }


def _paired_comparison(
    records: list[dict[str, Any]], level: int, games: int, seed_start: int
) -> dict[str, Any]:
    by_pair = {
        (row["level"], row["seed"], row["agent"]): float(row["normalized_score"])
        for row in records
    }
    differences = [
        by_pair[(level, seed, "student")] - by_pair[(level, seed, "baseline")]
        for seed in range(seed_start, seed_start + games)
        if (level, seed, "student") in by_pair and (level, seed, "baseline") in by_pair
    ]
    if not differences:
        return {"level": level, "paired_games": 0, "mean_gain": 0.0, "ci95": None,
                "credible_improvement": False}
    mean_gain = statistics.fmean(differences)
    if len(differences) < 20:
        return {
            "level": level,
            "paired_games": len(differences),
            "mean_gain": round(mean_gain, 4),
            "ci95": None,
            "credible_improvement": False,
        }
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    # 2.1 conservatively covers the two-sided 95% t interval at n >= 20.
    margin = 2.1 * standard_error
    return {
        "level": level,
        "paired_games": len(differences),
        "mean_gain": round(mean_gain, 4),
        "ci95": [round(mean_gain - margin, 4), round(mean_gain + margin, 4)],
        "credible_improvement": len(differences) >= 20 and mean_gain - margin > 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired benchmark on the official Arkanoid server")
    parser.add_argument("--games", type=int, default=20, help="episodes per agent and level (minimum 20 for the requested protocol)")
    parser.add_argument("--levels", type=int, nargs="+", choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--agent", choices=("student", "baseline", "both"), default="both")
    parser.add_argument("--transport", choices=("live", "logic"), default="live",
                        help="live starts a WebSocket server; logic uses the same server physics in-process")
    parser.add_argument("--seed-start", type=int, default=20261116)
    parser.add_argument("--acceleration", type=float, default=5.0)
    parser.add_argument("--level-time-limit", type=float, default=600.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--episode-timeout", type=float, default=240.0)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_results.json")
    args = parser.parse_args()
    if args.games < 1 or args.acceleration <= 0:
        parser.error("--games and --acceleration must be positive")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    agents = ("baseline", "student") if args.agent == "both" else (args.agent,)
    records: list[dict[str, Any]] = []
    decision_latencies: dict[tuple[str, int], list[float]] = {}
    planning_latencies: dict[tuple[str, int], list[float]] = {}
    for level in args.levels:
        for index in range(args.games):
            seed = args.seed_start + index
            for agent_name in agents:
                print(f"level={level} seed={seed} agent={agent_name}", flush=True)
                if args.transport == "live":
                    record = run_episode(
                        agent_name,
                        level,
                        seed,
                        args.acceleration,
                        args.level_time_limit,
                        args.startup_timeout,
                        args.episode_timeout,
                    )
                else:
                    record = run_logic_episode(
                        agent_name, level, seed, args.level_time_limit
                    )
                key = (agent_name, level)
                decision_latencies.setdefault(key, []).extend(record.pop("_decision_times_ms", []))
                planning_latencies.setdefault(key, []).extend(record.pop("_planning_times_ms", []))
                records.append(record)

    summary = [
        _summary(
            records,
            agent_name,
            level,
            decision_latencies.get((agent_name, level), []),
            planning_latencies.get((agent_name, level), []),
        )
        for level in args.levels
        for agent_name in agents
    ]
    comparisons = (
        [_paired_comparison(records, level, args.games, args.seed_start) for level in args.levels]
        if args.agent == "both" else []
    )
    result = {
        "protocol": {
            "transport": args.transport,
            "engine": (
                "server/logic.py Arkanoid, invoked in-process"
                if args.transport == "logic" else "server.server WebSocket"
            ),
            "server_command": "server.server --levels N" if args.transport == "live" else None,
            "acceleration": args.acceleration if args.transport == "live" else None,
            "games_per_agent_per_level": args.games,
            "seed_start": args.seed_start,
        },
        "summary": summary,
        "paired_student_minus_baseline": comparisons,
        "games": records,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_student_minus_baseline": comparisons}, indent=2))
    print(f"Full results: {args.output}")


if __name__ == "__main__":
    main()
