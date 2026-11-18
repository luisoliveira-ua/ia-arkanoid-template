"""Submission agent for the Arkanoid assignment: exact-physics expectimax planning of every rebound.

The agent itself lives in ``agents/expectimax_agent.py`` (planner and paddle control) and
``agents/fast_physics.py`` (exact replica of the server's ball physics). This module only names
the submission class and runs it against a game server.
"""

import argparse
import asyncio

from agents.expectimax_agent import ExpectimaxAgent

AGENT_NAME = "Student Arkanoid Agent"  # fixed name of the group: the dashboard identifies and ranks the agent by it


class StudentAgent(ExpectimaxAgent):
    """Submission-facing agent: exact-physics expectimax planning of every paddle rebound."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Arkanoid expectimax agent")
    parser.add_argument("-s", "--server", default="ws://localhost:8765/ws", help="WebSocket URI")
    parser.add_argument("-n", "--name", default=AGENT_NAME, help="Agent name shown in the dashboard")
    args = parser.parse_args()
    asyncio.run(StudentAgent(args.server, args.name).run())


if __name__ == "__main__":
    main()
