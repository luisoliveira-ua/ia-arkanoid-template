"""Submission agent for the Arkanoid assignment: exact-physics expectimax planning of every rebound.

The agent itself lives in ``agents/expectimax_agent.py`` (planner and paddle control) and
``agents/fast_physics.py`` (exact replica of the server's ball physics). This module only names
the submission class and runs it against a game server.
"""

import argparse
import asyncio

from agents.expectimax_agent import ExpectimaxAgent

AGENT_ID: str | None = None  # Student number of a group member (e.g. "a12345"); --id overrides it.


class StudentAgent(ExpectimaxAgent):
    """Submission-facing agent: exact-physics expectimax planning of every paddle rebound."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Arkanoid expectimax agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Student Arkanoid Agent")
    parser.add_argument("--id", default=None)
    args = parser.parse_args()
    asyncio.run(StudentAgent(args.server, args.name, args.id or AGENT_ID).run())


if __name__ == "__main__":
    main()
