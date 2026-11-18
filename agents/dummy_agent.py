"""Dummy autonomous agent that keeps the paddle centre under the ball."""

import argparse
import asyncio
from typing import Any

from agents.base_agent import BaseAgent


class DummyAgent(BaseAgent):
    """Agent that follows the ball horizontally, or idles when the state has no positions."""

    async def raquete(self) -> dict[str, Any] | None:
        s = self.current_state or {}
        if "paddle_x" not in s or "ball_x" not in s:
            return None
        centro_raquete = s["paddle_x"] + s.get("paddle_width", 84.0) / 2
        diff = s["ball_x"] - centro_raquete  # >0: bola à direita

        margem = 14
        if diff > margem:
            return {"action": "move", "direction": "EAST"}
        if diff < -margem:
            return {"action": "move", "direction": "WEST"}
        return None

    async def deliberate(self) -> dict[str, Any] | None:
        """Move towards the ball when there are valid moves."""
        if not self.current_state or self.current_state.get("game_over"):
            return None

        valid_actions: list[dict[str, Any]] = (
            self.current_state.get("valid_actions")
            or self.current_state.get("actions")
            or []
        )
        if not valid_actions:
            return None
        return await self.raquete()


def main() -> None:
    """Run Dummy Agent against a game server."""
    parser = argparse.ArgumentParser(description="Arkanoid dummy agent")
    parser.add_argument("-s", "--server", default="ws://localhost:8765/ws", help="WebSocket URI")
    parser.add_argument("-n", "--name", default="Dummy Agent", help="Agent name shown in the high score table")
    args = parser.parse_args()
    asyncio.run(DummyAgent(args.server, args.name).run())


if __name__ == "__main__":
    main()
