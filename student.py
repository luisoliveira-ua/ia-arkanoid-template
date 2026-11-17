"""Entry point for the Arkanoid agents (expectimax planner by default, strategic and RL alternatives)."""

import argparse
import asyncio
from pathlib import Path

from agents.expectimax_agent import ExpectimaxAgent
from agents.rl_agent import DEFAULT_Q_PATH, ReinforcementAgent
from agents.strategic_agent import StrategicAgent


class StudentAgent(ExpectimaxAgent):
    """Submission-facing agent: exact-physics expectimax planning of every paddle rebound."""


class StudentStrategicAgent(StrategicAgent):
    """Previous submission agent using sampled strategic planning on approximate physics."""


AGENT_ID: str | None = None  # Set this to a group member's student number.


def main() -> None:
    parser = argparse.ArgumentParser(description="Arkanoid strategic or RL agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Student Arkanoid Agent")
    parser.add_argument("--id", default=None)
    parser.add_argument(
        "--strategy",
        choices=("expectimax", "strategic", "rl"),
        default="expectimax",
        help="expectimax plans on exact physics (fastest clears); strategic is the previous baseline; "
        "rl learns a rebound policy online",
    )
    parser.add_argument(
        "--rl-epsilon",
        type=float,
        default=ReinforcementAgent.DEFAULT_EPSILON,
        help="safe-action exploration probability for the RL strategy (0 to 1)",
    )
    parser.add_argument(
        "--rl-q-table",
        type=Path,
        default=DEFAULT_Q_PATH,
        help="persistent JSON checkpoint for the RL strategy",
    )
    args = parser.parse_args()
    if args.strategy == "rl":
        agent = ReinforcementAgent(
            args.server,
            args.name,
            args.id or AGENT_ID,
            epsilon=args.rl_epsilon,
            q_path=args.rl_q_table,
        )
    elif args.strategy == "strategic":
        agent = StudentStrategicAgent(args.server, args.name, args.id or AGENT_ID)
    else:
        agent = StudentAgent(args.server, args.name, args.id or AGENT_ID)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
