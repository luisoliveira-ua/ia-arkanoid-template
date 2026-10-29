"""Predictive Arkanoid agent with a one-ply search over paddle bounce zones."""

import argparse
import asyncio
import math
from typing import Any

from agents.base_agent import BaseAgent

AGENT_ID: str | None = None  # Set to a group member's student number for ranking.


class StudentAgent(BaseAgent):
    """Catch the ball, then choose the bounce region aimed at useful bricks."""

    STEP = 28.0
    REGIONS = ((1 / 6, -45.0), (1 / 2, 0.0), (5 / 6, 45.0))

    @staticmethod
    def _landing_x(state: dict[str, Any]) -> tuple[float, float]:
        """Return predicted impact x and seconds until the paddle plane."""
        x, y = float(state["ball_x"]), float(state["ball_y"])
        vx, vy = float(state["ball_vx"]), float(state["ball_vy"])
        radius = float(state.get("ball_radius", 6))
        paddle_y = float(state["paddle_y"])
        plane = paddle_y - radius
        if vy > 0:
            t = max(0.0, (plane - y) / vy)
        else:
            # While the ball rises, estimate its return after bouncing off the ceiling.
            ceiling = float(state.get("ceiling_y", 82.0)) + radius
            speed_y = max(1e-6, abs(vy))
            t = max(0.0, (y - ceiling) / speed_y) + max(0.0, (plane - ceiling) / speed_y)
        width = float(state.get("width", 600))
        span = width - 2 * radius
        unfolded = x - radius + vx * t
        folded = unfolded % (2 * span)
        return radius + (folded if folded <= span else 2 * span - folded), t

    @staticmethod
    def _ray_score(
        x: float, y: float, angle: float, state: dict[str, Any], bricks: list[dict[str, Any]]
    ) -> float:
        """Trace a nominal outgoing ray and score its first brick collision."""
        speed = float(state.get("ball_speed", 400.0))
        vx, vy = speed * math.sin(math.radians(angle)), -speed * math.cos(math.radians(angle))
        radius = float(state.get("ball_radius", 6.0))
        width = float(state.get("width", 600.0))
        ceiling = float(state.get("ceiling_y", 82.0)) + radius
        # A short swept simulation is sufficient to estimate a useful first target.
        dt = 0.025
        elapsed = 0.0
        for _ in range(240):
            nx, ny = x + vx * dt, y + vy * dt
            if nx < radius:
                nx, vx = 2 * radius - nx, abs(vx)
            elif nx > width - radius:
                nx, vx = 2 * (width - radius) - nx, -abs(vx)
            if ny < ceiling:
                ny, vy = 2 * ceiling - ny, abs(vy)
            elapsed += dt
            for brick in bricks:
                bx, by = float(brick.get("x", 0)), float(brick.get("y", 0))
                bw, bh = float(brick.get("width", 50)), float(brick.get("height", 20))
                if bx - radius <= nx <= bx + bw + radius and by - radius <= ny <= by + bh + radius:
                    points = float(brick.get("points", 0))
                    remaining = max(1, int(brick.get("hits_remaining", 1)))
                    # Reward points while mildly preferring bricks that break sooner.
                    return points / (elapsed + 0.5 * (remaining - 1))
            x, y = nx, ny
        return 0.0

    def _choose_region(self, state: dict[str, Any], landing: float, time_to_paddle: float) -> float:
        """Search paddle positions, action reachability, and the three rebound zones."""
        paddle_x = float(state["paddle_x"])
        paddle_w = float(state["paddle_width"])
        max_x = float(state.get("width", 600)) - paddle_w
        # One action is received per game tick; do not plan beyond what can be reached.
        reachable_steps = max(1, min(30, math.ceil(time_to_paddle * 30)))
        max_travel = reachable_steps * self.STEP
        low, high = max(0.0, paddle_x - max_travel), min(max_x, paddle_x + max_travel)
        bricks = [b for b in state.get("bricks", []) if not b.get("indestructible")]
        paddle_y = float(state["paddle_y"]) - float(state.get("ball_radius", 6))
        best_cost, best_x = math.inf, paddle_x

        for zone_fraction, angle in self.REGIONS:
            target_paddle_x = min(high, max(low, landing - zone_fraction * paddle_w))
            # The impact point must land inside the selected third of the paddle.
            relative = landing - target_paddle_x
            zone_width = paddle_w / 3
            if not (0 <= relative <= paddle_w):
                continue
            hit_score = self._ray_score(landing, paddle_y, angle, state, bricks)
            catch_error = max(0.0, abs(relative - zone_fraction * paddle_w) - zone_width * 0.42)
            distance_cost = abs(target_paddle_x - paddle_x) / self.STEP * 0.015
            # High-value reachable targets outweigh small positioning differences.
            cost = catch_error * 4 + distance_cost - math.log1p(hit_score) * 18
            if cost < best_cost:
                best_cost, best_x = cost, target_paddle_x

        # If no destructible brick is reachable on the nominal rays, prioritize catching.
        if best_cost == math.inf:
            best_x = min(max_x, max(0.0, landing - paddle_w / 2))
        return min(high, max(low, best_x))

    async def deliberate(self) -> dict[str, Any] | None:
        state = self.current_state
        if not state or state.get("game_over") or state.get("game_won"):
            return None
        landing, time_to_paddle = self._landing_x(state)
        if float(state["ball_vy"]) > 0:
            target = self._choose_region(state, landing, time_to_paddle)
        else:
            # During ascent, track the ball so a new descent can be caught quickly.
            target = landing - float(state["paddle_width"]) / 2
        center = float(state["paddle_x"]) + float(state["paddle_width"]) / 2
        delta = target + float(state["paddle_width"]) / 2 - center
        if abs(delta) < self.STEP / 2:
            return None
        direction = "EAST" if delta > 0 else "WEST"
        actions = state.get("valid_actions", state.get("actions", []))
        return next((action for action in actions if action.get("direction") == direction), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Arkanoid predictive search agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Student Agent")
    parser.add_argument("--id", default=None)
    args = parser.parse_args()
    asyncio.run(StudentAgent(args.server, args.name, args.id or AGENT_ID).run())


if __name__ == "__main__":
    main()
