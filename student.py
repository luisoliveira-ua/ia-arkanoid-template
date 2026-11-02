"""Entry point for the Arkanoid search agent."""

import argparse
import asyncio

from agents.search_agent import SearchAgent

# Keep the expected student module/class name while using the tree-search agent.
StudentAgent = SearchAgent
AGENT_ID: str | None = None  # Set this to a group member's student number.


def main() -> None:
    parser = argparse.ArgumentParser(description="Arkanoid points-per-second search agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Student Search Agent")
    parser.add_argument("--id", default=None)
    args = parser.parse_args()
    asyncio.run(StudentAgent(args.server, args.name, args.id or AGENT_ID).run())


if __name__ == "__main__":
    main()
