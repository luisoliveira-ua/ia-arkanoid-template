"""Base agent client for communicating with AI Game Framework servers."""

import json
import logging
import uuid
from typing import Any

import websockets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - AGENT - %(levelname)s - %(message)s"
)


def resolve_agent_id(agent_id: str | None = None) -> str:
    """Return the trimmed agent id, or a random one (uuid4 hex) when it is None or empty."""
    agent_id = (agent_id or "").strip()
    return agent_id or uuid.uuid4().hex


class BaseAgent:
    """Abstract base agent handling WebSocket communication with the game server."""

    def __init__(
        self,
        server_uri: str = "ws://localhost:8765/ws",
        name: str = "Agent",
        agent_id: str | None = None,
    ) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Agent name must not be empty")
        self.server_uri = server_uri
        self.name = name
        self.agent_id = resolve_agent_id(agent_id)
        self.current_state: dict[str, Any] | None = None
        self.player_id: int | None = None
        self.setup_data: dict[str, Any] | None = None
        self.stop_reason: str = "connection closed"

    async def run(self) -> None:
        """Connect to game server and run the observation-action loop."""
        try:
            async with websockets.connect(self.server_uri) as websocket:
                await websocket.send(json.dumps({"client": "agent", "name": self.name, "id": self.agent_id}))
                logging.info("Connected to %s as %s (id %s)", self.server_uri, self.name, self.agent_id)

                async for message in websocket:
                    data = json.loads(message)

                    msg_type = data.get("type")
                    if msg_type == "setup":
                        self.player_id = data.get("player_id")
                        self.setup_data = data
                        logging.info("Assigned player ID: %s", self.player_id)
                        continue

                    if msg_type in ("state", "update"):
                        self.current_state = data
                        if data.get("game_won"):
                            logging.info("Game won! Victory achieved. Stopping agent.")
                            self.stop_reason = "game won"
                            break
                        if data.get("game_over"):
                            logging.info("Game over received. Stopping agent.")
                            self.stop_reason = "time limit reached" if data.get("timed_out") else "game over"
                            break

                        action = await self.deliberate()
                        if action is not None:
                            await websocket.send(json.dumps(action))

        except Exception as e:
            logging.error("Connection error: %s", e)
            self.stop_reason = f"connection error: {e}"
        self.report()

    def summary(self) -> str:
        """Plain-text result of the session from the last state received (the numbers the server grades on)."""
        s = self.current_state or {}
        limit = float(s.get("level_time_limit", 0.0))
        level_time = f"{float(s.get('level_time_elapsed', 0.0)):.1f} s" + (f" of {limit:.0f} s" if limit > 0.0 else "")
        stage, stages = s.get("stage", s.get("level", "?")), s.get("max_stages", s.get("max_levels", "?"))
        if isinstance(stage, int) and isinstance(stages, int):
            stage = min(stage, stages)  # after the last level is cleared the server counts one past it
        rows = [
            ("Agent", f"{self.name} (id {self.agent_id})"),
            ("Stop reason", self.stop_reason),
            ("Level", f"{stage} of {stages}: {s.get('stage_name', s.get('level_name', '?'))}"),
            ("Score", str(s.get("score", 0))),
            ("Avg normalized score", f"{float(s.get('avg_norm_score', 0.0)):.2f} points/s (the grade)"),
            ("Level normalized score", f"{float(s.get('normalized_score', 0.0)):.2f} points/s"),
            ("Game time", f"{float(s.get('time_elapsed', 0.0)):.1f} s"),
            ("Time on this level", level_time),
            ("Lives left", str(s.get("lives", "?"))),
            ("High score", f"{s.get('high_score', 0)} ({float(s.get('high_score_normalized', 0.0)):.2f} points/s)"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(["=" * 56, "AGENT STOPPED", *(f"{k:<{width}}  {v}" for k, v in rows), "=" * 56])

    def report(self) -> None:
        """Print the session summary on the terminal when the agent stops."""
        print(self.summary(), flush=True)

    async def deliberate(self) -> dict[str, Any] | None:
        """Analyze current state and decide on the next action."""
        raise NotImplementedError("Subclasses must implement deliberate()")
