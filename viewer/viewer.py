"""Pygame graphical viewer for IA Arkanoid."""

import argparse
import asyncio
import json
import logging
import math
import threading
from typing import Any

import pygame
import websockets

from viewer.sprites import SpriteManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - VIEWER - %(levelname)s - %(message)s"
)


class ArkanoidViewer:
    """Pygame visualizer connecting as a spectator/client to the Arkanoid server."""

    def __init__(
        self,
        server_uri: str = "ws://localhost:8765/ws",
        width: int = 600,
        height: int = 800,
    ) -> None:
        self.server_uri = server_uri
        self.width = width
        self.height = height
        self.running = False
        self.connected = False

        self.state_lock = threading.Lock()
        self.latest_state: dict[str, Any] | None = None
        self.sprite_manager: SpriteManager | None = None

        self.font: pygame.font.Font | None = None
        self.font_large: pygame.font.Font | None = None
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self._background: pygame.Surface | None = None

        # Queue for interactive actions if player controls via viewer window
        self.action_queue: list[dict[str, Any]] = []

    def init_pygame(self) -> None:
        """Initialize Pygame display, fonts, and sprite manager."""
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("IA Arkanoid - Pygame Viewer")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        self.sprite_manager = SpriteManager()

    async def _network_loop(self) -> None:
        """WebSocket communication loop with server."""
        while self.running:
            try:
                async with websockets.connect(self.server_uri) as ws:
                    await ws.send(json.dumps({"client": "frontend"}))
                    self.connected = True
                    logging.info("Viewer connected to %s", self.server_uri)

                    while self.running:
                        # Send any queued actions from viewer input
                        while self.action_queue:
                            act = self.action_queue.pop(0)
                            await ws.send(json.dumps(act))

                        # Receive updates with short timeout
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.03)
                            data = json.loads(msg)
                            msg_type = data.get("type")
                            if msg_type in ("setup", "state", "update"):
                                w = data.get("width")
                                h = data.get("height")
                                if (
                                    w
                                    and h
                                    and (w != self.width or h != self.height)
                                    and self.screen is not None
                                ):
                                    self.width = int(w)
                                    self.height = int(h)
                                    self.screen = pygame.display.set_mode(
                                        (self.width, self.height)
                                    )
                                with self.state_lock:
                                    self.latest_state = data
                        except asyncio.TimeoutError:
                            continue

            except Exception as e:
                self.connected = False
                logging.debug("Network loop notice: %s. Reconnecting...", e)
                await asyncio.sleep(1.0)

    def _start_network_thread(self) -> None:
        """Launch the async network loop in a background thread."""

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._network_loop())

        net_thread = threading.Thread(target=runner, daemon=True)
        net_thread.start()

    def handle_input(self) -> None:
        """Handle keyboard input from the viewer window."""
        if not self.connected:
            self.action_queue.clear()
            return
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.action_queue.append({"action": "move", "direction": "WEST"})
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.action_queue.append({"action": "move", "direction": "EAST"})
        elif keys[pygame.K_SPACE] or keys[pygame.K_RETURN]:
            self.action_queue.append({"action": "start_sim"})

    def draw_hud(self, surface: pygame.Surface, state: dict[str, Any]) -> None:
        """Draw score, stage, lives, and high score header using Nord palette."""
        if self.font is None or self.sprite_manager is None:
            return

        score = state.get("score", 0)
        high_score = state.get("high_score", 0)
        timer = float(state.get("timer", state.get("time_elapsed", 0.0)))
        norm_score = float(state.get("normalized_score", 0.0))
        avg_norm = float(state.get("avg_norm_score", 0.0))
        high_score_norm = float(state.get("high_score_normalized", 0.0))
        stage = state.get("stage", state.get("level", 1))
        stage_name = state.get("stage_name", state.get("level_name", f"Stage {stage}"))
        lives = state.get("lives", 3)

        # Draw HUD backdrop (height 82px) using Polar Night nord1
        pygame.draw.rect(surface, (59, 66, 82), (0, 0, self.width, 82))
        # Virtual ceiling boundary line at y=82 using Polar Night nord3
        pygame.draw.line(surface, (76, 86, 106), (0, 82), (self.width, 82), 2)

        # === ROW 1: Stage badge, Timer & Lives ===
        badge_text = self.font.render(f"STAGE {stage:02d}", True, (136, 192, 208))
        surface.blit(badge_text, (20, 8))

        limit = float(state.get("level_time_limit", 0.0))
        if limit > 0.0:  # the level deadline: count down to it (resets with every level)
            left = max(0.0, limit - float(state.get("level_time_elapsed", 0.0)))
            shown, label = math.ceil(left), "TIME LEFT"
            time_color = (191, 97, 106) if left <= 30.0 else (216, 222, 233)  # Nord red for the last 30 s
        else:  # no deadline configured: elapsed game time
            shown, label, time_color = int(timer), "TIME", (216, 222, 233)
        time_text = self.font.render(f"{label}: {shown // 60:02d}:{shown % 60:02d}", True, time_color)
        time_rect = time_text.get_rect(center=(self.width // 2, 16))
        surface.blit(time_text, time_rect)

        lives_lbl = self.font.render("LIVES:", True, (216, 222, 233))
        lives_lbl_x = self.width - 20 - (max(0, lives) * 32)
        surface.blit(lives_lbl, (lives_lbl_x - 60, 8))
        for i in range(max(0, lives)):
            x_pos = lives_lbl_x + (i * 32)
            surface.blit(self.sprite_manager.life_sprite, (x_pos, 12))

        # === ROW 2: Stage Title (Centered across full screen width) ===
        lvl_text = self.font.render(stage_name.upper(), True, (235, 203, 139))
        lvl_rect = lvl_text.get_rect(center=(self.width // 2, 38))
        surface.blit(lvl_text, lvl_rect)

        # === ROW 3: Score & High Score (Raw and Normalized rate) ===
        score_text = self.font.render(
            f"SCORE: {score:05d} ({norm_score:.1f}/s)  AVG: {avg_norm:.1f}", True, (236, 239, 244)
        )
        surface.blit(score_text, (20, 58))

        high_text = self.font.render(
            f"HIGH: {high_score:05d} (AVG {high_score_norm:.1f})", True, (129, 161, 193)
        )
        high_rect = high_text.get_rect(topright=(self.width - 20, 58))
        surface.blit(high_text, high_rect)

    def _build_background(self) -> pygame.Surface:
        """Pre-render the static Polar Night background with its subtle grid (once per window size)."""
        bg = pygame.Surface((self.width, self.height))
        bg.fill((46, 52, 64))
        for gy in range(90, self.height, 40):
            pygame.draw.line(bg, (59, 66, 82), (0, gy), (self.width, gy), 1)
        for gx in range(0, self.width, 40):
            pygame.draw.line(bg, (59, 66, 82), (gx, 82), (gx, self.height), 1)
        return bg

    def render(self) -> None:
        """Render single frame of Arkanoid gameplay."""
        if self.screen is None or self.sprite_manager is None:
            return

        if self._background is None or self._background.get_size() != self.screen.get_size():
            self._background = self._build_background()
        self.screen.blit(self._background, (0, 0))

        with self.state_lock:
            state = self.latest_state

        if state is None or not self.connected:
            if self.font is not None:
                status_msg = (
                    "CONNECTING TO SERVER..."
                    if not self.connected
                    else "WAITING FOR AGENT / GAME START..."
                )
                text = self.font.render(status_msg, True, (216, 222, 233))
                rect = text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(text, rect)
            pygame.display.flip()
            return

        # 1. Render HUD
        self.draw_hud(self.screen, state)

        # 2. Render Bricks
        bricks = state.get("bricks", [])
        for b in bricks:
            bx = float(b.get("x", 0.0))
            by = float(b.get("y", 0.0))
            bw = float(b.get("width", 50.0))
            bh = float(b.get("height", 20.0))
            b_type = str(b.get("type", "frost"))
            sprite = self.sprite_manager.get_brick_sprite(b_type, bw, bh)
            self.screen.blit(sprite, (int(round(bx)), int(round(by))))

        # 3. Render Paddle
        px = float(state.get("paddle_x", (self.width - 84.0) / 2.0))
        py = float(state.get("paddle_y", self.height - 60.0))
        pw = float(state.get("paddle_width", 84.0))
        ph = float(state.get("paddle_height", 14.0))
        p_sprite = self.sprite_manager.get_paddle_sprite(pw, ph)
        self.screen.blit(p_sprite, (int(round(px)), int(round(py))))

        # 4. Render Ball
        bx = float(state.get("ball_x", 0.0))
        by = float(state.get("ball_y", 0.0))
        br = float(state.get("ball_radius", 6.0))
        b_sprite = self.sprite_manager.get_ball_sprite(br)
        self.screen.blit(
            b_sprite, (int(round(bx - br)), int(round(by - br)))
        )

        # 5. Overlays (Game Won / Game Over)
        if state.get("game_won"):
            self._render_banner("VICTORY! ALL STAGES CLEARED!", (163, 190, 140))
        elif state.get("game_over"):
            self._render_banner("GAME OVER", (191, 97, 106))

        pygame.display.flip()

    def _render_banner(self, message: str, color: tuple[int, int, int]) -> None:
        """Render banner overlay across center of screen."""
        if self.screen is None or self.font_large is None:
            return

        banner_rect = pygame.Rect(0, self.height // 2 - 40, self.width, 80)
        banner_surf = pygame.Surface(
            (banner_rect.width, banner_rect.height), pygame.SRCALPHA
        )
        banner_surf.fill((46, 52, 64, 225))
        self.screen.blit(banner_surf, banner_rect.topleft)
        pygame.draw.line(
            self.screen,
            color,
            (0, banner_rect.top),
            (self.width, banner_rect.top),
            2,
        )
        pygame.draw.line(
            self.screen,
            color,
            (0, banner_rect.bottom),
            (self.width, banner_rect.bottom),
            2,
        )

        text = self.font_large.render(message, True, color)
        rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, rect)

    def run(self) -> None:
        """Main Pygame event and render loop."""
        self.init_pygame()
        self.running = True
        self._start_network_thread()

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                    if (
                        event.type == pygame.KEYDOWN
                        and event.key == pygame.K_ESCAPE
                    ):
                        self.running = False
                        break

                self.handle_input()
                self.render()
                if self.clock is not None:
                    self.clock.tick(60)
        finally:
            self.running = False
            pygame.quit()


def main() -> None:
    """CLI entry point for launching Arkanoid Pygame Viewer."""
    parser = argparse.ArgumentParser(description="Arkanoid Pygame Viewer")
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8765/ws",
        help="WebSocket server URI",
    )
    parser.add_argument(
        "--width", type=int, default=600, help="Initial window width"
    )
    parser.add_argument(
        "--height", type=int, default=800, help="Initial window height"
    )
    args = parser.parse_args()

    viewer = ArkanoidViewer(
        server_uri=args.server, width=args.width, height=args.height
    )
    viewer.run()


if __name__ == "__main__":
    main()
