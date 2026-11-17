"""Arkanoid game server integrating with AI Game Framework."""

import argparse
import logging
import re
import uuid
from typing import Any

import aigf.interface as interface
from aigf.main import run_app

from server.config import AppConfig, load_config, parse_levels
from server.highscores import HighScoreManager
from server.logic import Arkanoid
from server.remote import HIGHSCORE_URL, RemoteScoreSubmitter

_MAX_NAME_LEN = 32
_MAX_ID_LEN = 64
_MAX_SUBSTEP = 0.05  # seconds of simulated time per physics call

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - ARKANOID - %(levelname)s - %(message)s"
)


def sanitize_name(raw: object) -> str:
    """Clean an agent-supplied name: printable ASCII only, trimmed, max 32 chars, else "PLAYER"."""
    text = "".join(ch for ch in str(raw) if " " <= ch <= "~").strip()
    return text[:_MAX_NAME_LEN].strip() or "PLAYER"


def sanitize_id(raw: object) -> str:
    """Clean an agent-supplied id: [A-Za-z0-9._-] only, max 64 chars, else a random uuid4 hex."""
    text = re.sub(r"[^A-Za-z0-9._-]", "", str(raw or ""))[:_MAX_ID_LEN]
    return text or uuid.uuid4().hex


class ArkanoidGameServer(interface.GameInterface):
    """Arkanoid game server implementation using the AI Game Framework."""

    def __init__(
        self,
        config: AppConfig | None = None,
        fps: int | None = None,
        acceleration: float = 1.0,
        levels: list[str] | None = None,
        share_highscore: bool = False,
        level_time_limit: float | None = None,
    ) -> None:
        if acceleration <= 0.0:
            raise ValueError("acceleration must be > 0")
        if level_time_limit is not None and level_time_limit < 0.0:
            raise ValueError("level_time_limit must be >= 0")
        self.app_config = config or load_config()
        if fps is not None:
            self.app_config.game.fps = fps
        if levels:
            self.app_config.game.level_sequence = levels
        if level_time_limit is not None:
            self.app_config.game.level_time_limit = level_time_limit
        self.acceleration = acceleration
        self.sim_dt = 1.0 / self.app_config.game.fps
        # The framework loop sleeps 1/fps between ticks: raising its rate shortens wall-clock time,
        # while every tick still advances the simulation by exactly sim_dt (same dynamics for agents).
        super().__init__(
            is_real_time=True,
            fps=max(1, round(self.app_config.game.fps * acceleration)),
            maps_dir=self.app_config.game.maps_dir,
        )
        scores = HighScoreManager(self.app_config.game.highscores_file if share_highscore else None)
        self.game = Arkanoid(self.app_config, high_score_manager=scores)
        self.remote: RemoteScoreSubmitter | None = None
        if share_highscore:
            self.remote = RemoteScoreSubmitter(HIGHSCORE_URL)
            self.game.score_listener = self.remote.submit
        self.player_id: int | None = None

    async def on_player_connect(self, player_id: int) -> None:
        """Handle agent connection."""
        logging.info("Player %d connected.", player_id)
        if self.player_id is None or self.game.game_over:
            self.player_id = player_id
            self.game.reset_game()
            self.state = interface.GameState.RUNNING
            logging.info("Started game session for player %d.", player_id)
        else:
            logging.warning(
                "Extra player %d connected. Single-player mode active.", player_id
            )

    async def on_handshake(self, player_id: int, data: dict[str, Any]) -> None:
        """Handle agent handshake metadata."""
        await super().on_handshake(player_id, data)
        name = data.get("name")
        if name and isinstance(name, str):
            self.game.player_name = sanitize_name(name)
        self.game.agent_id = sanitize_id(data.get("id"))

    async def on_player_disconnect(self, player_id: int) -> None:
        """Handle agent disconnection."""
        logging.info("Player %d disconnected.", player_id)
        if self.player_id == player_id:
            self.player_id = None
            self.game.player_name = "PLAYER"
            self.game.agent_id = None
            self.state = interface.GameState.LOBBY
            self.game.reset_game()
            logging.info("Server reset to LOBBY awaiting a new agent.")

    async def on_start_sim(self) -> None:
        """Start or resume simulation loop."""
        if self.game.game_over:
            self.game.reset_game()
        self.state = interface.GameState.RUNNING

    async def on_reset_sim(self) -> None:
        """Reset simulation environment."""
        self.game.reset_game()
        self.state = interface.GameState.LOBBY

    async def process_action(self, player_id: int, action: dict[str, Any]) -> None:
        """Process movement actions sent by player agent or frontend."""
        act_type = action.get("action")
        if act_type == "start_sim":
            await self.on_start_sim()
            return

        if self.state == interface.GameState.RUNNING and (
            player_id == self.player_id or (self.player_id is None and player_id == 0)
        ):
            if act_type == "move":
                direction = action.get("direction")
                if isinstance(direction, str):
                    self.game.move_paddle(direction)
            elif isinstance(act_type, str):
                self.game.move_paddle(act_type)

    async def tick(self, dt: float) -> None:
        """Advance game physics by one fixed simulation step (dt from the framework is ignored)."""
        if self.state == interface.GameState.RUNNING:
            remaining = self.sim_dt
            while remaining > 1e-9 and not self.game.game_over:
                step = min(remaining, _MAX_SUBSTEP)
                self.game.update(step)
                remaining -= step
            if self.game.game_over:
                logging.info("Game Over or Game Won!")
                self.state = interface.GameState.LOBBY

    def get_state(self) -> dict[str, Any]:
        """Serialize current state for agents and viewer."""
        state = self.game.get_state()
        state["player_id"] = self.player_id
        return state

    def get_setup_payload(self) -> dict[str, Any]:
        """Return setup configuration payload."""
        return {
            "width": self.game.width,
            "height": self.game.height,
            "ceiling_y": self.game.ceiling_y,
            "paddle_width": self.game.paddle_width,
            "paddle_height": self.game.paddle_height,
            "stage": self.game.level_idx + 1,
            "max_stages": len(self.game.level_sequence),
            "level": self.game.level_idx + 1,
            "max_levels": len(self.game.level_sequence),
            "brick_width": self.app_config.bricks.width,
            "brick_height": self.app_config.bricks.height,
        }


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser for the game server."""
    parser = argparse.ArgumentParser(description="Arkanoid AI Game Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--fps", type=int, default=30, help="Simulation ticks per simulated second (default: 30)")
    parser.add_argument(
        "--acceleration",
        type=float,
        default=1.0,
        help="Wall-clock speed-up factor, e.g. 10 runs 10x faster for training (default: 1.0)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Levels to play: N, N,M,... or A-B (e.g. 1 | 1,2,3 | 1-3 | 1,3-5). Default: all in config.json",
    )
    parser.add_argument(
        "--level-time-limit",
        type=float,
        default=None,
        help="Seconds of game time allowed per level, 0 disables; game over if a level is not cleared in time "
        "(default: game.level_time_limit in config.json, 600)",
    )
    parser.add_argument(
        "--share-highscore",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist scores to the local CSV file (highscores.csv) and upload the normalized score (default: off)",
    )
    return parser


def main() -> None:
    """CLI entry point for running Arkanoid Game Server."""
    parser = build_parser()
    args = parser.parse_args()
    if args.fps < 1 or args.acceleration <= 0:
        parser.error("--fps must be >= 1 and --acceleration must be > 0")
    if args.level_time_limit is not None and args.level_time_limit < 0:
        parser.error("--level-time-limit must be >= 0")

    cfg = load_config(args.config)
    try:
        levels = parse_levels(args.levels, cfg.game.maps_dir) if args.levels else None
    except ValueError as e:
        parser.error(str(e))
    server = ArkanoidGameServer(
        config=cfg,
        fps=args.fps,
        acceleration=args.acceleration,
        levels=levels,
        share_highscore=args.share_highscore,
        level_time_limit=args.level_time_limit,
    )
    logging.info(
        "fps=%d acceleration=%.2fx levels=%s level_time_limit=%.0fs share_highscore=%s",
        args.fps,
        args.acceleration,
        cfg.game.level_sequence,
        cfg.game.level_time_limit,
        args.share_highscore,
    )
    run_app(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
