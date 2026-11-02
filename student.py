"""Predictive Arkanoid agent with a one-ply search over paddle bounce zones."""

import argparse
import asyncio
import math
from typing import Any

from agents.base_agent import BaseAgent
from arkanoid_domain import ArkanoidSearchDomain, search_decision

AGENT_ID: str | None = None  # Set to a group member's student number for ranking.


class StudentAgent(BaseAgent):
    """Catch the ball, then choose the bounce region aimed at useful bricks."""

    STEP = 28.0
    DEAD_BAND = STEP * 0.55
    TRACKING_SMOOTHING = 0.25

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._target_x: float | None = None
        self._locked_target_x: float | None = None
        self._was_descending = False
        self._last_descent_velocity: tuple[float, float] | None = None

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

    def _choose_region(self, state: dict[str, Any]) -> float:
        """Use lookahead to break ties without sacrificing immediate target quality."""
        domain, initial, plan, actions = search_decision(state, horizon=2, node_limit=40)
        _, greedy_target, greedy_hit = domain.choose_paddle_target(initial)
        if plan and len(plan) > 1 and actions:
            planned_zone = actions[0]
            target = domain.safe_paddle_target(initial, planned_zone)
            if target is not None:
                planned_hit = domain.predict_first_brick(initial, planned_zone)
                same_immediate_priority = (
                    domain.target_priority(planned_hit) == domain.target_priority(greedy_hit)
                )
                plan_rate = domain.points_per_second(plan[-1]) if plan else 0.0
                greedy_rate = float(greedy_hit.get("full_flight_points_per_second", 0.0) or 0.0)
                if (
                    domain.target_priority(planned_hit) > domain.target_priority(greedy_hit)
                    or (same_immediate_priority and plan_rate > greedy_rate)
                ):
                    return target
        return greedy_target

    async def deliberate(self) -> dict[str, Any] | None:
        state = self.current_state
        if not state or state.get("game_over") or state.get("game_won"):
            return None
        landing, time_to_paddle = self._landing_x(state)
        descending = float(state["ball_vy"]) > 0
        paddle_width = float(state["paddle_width"])
        max_x = float(state.get("width", 600)) - paddle_width
        if descending:
            # Commit to one intercept target during a descent. Rechoosing the
            # bounce zone on every observation can oscillate. Replan after a
            # collision changes the ball's velocity.
            velocity = (float(state["ball_vx"]), float(state["ball_vy"]))
            trajectory_changed = (
                self._last_descent_velocity is not None
                and (abs(velocity[0] - self._last_descent_velocity[0]) > 1.0
                     or abs(velocity[1] - self._last_descent_velocity[1]) > 1.0)
            )
            if not self._was_descending or self._locked_target_x is None or trajectory_changed:
                self._locked_target_x = self._choose_region(state)
            target = self._locked_target_x
            self._last_descent_velocity = velocity
        else:
            # During ascent, track the predicted landing point with smoothing.
            proposed = min(max_x, max(0.0, landing - paddle_width / 2))
            if self._target_x is None:
                self._target_x = proposed
            else:
                alpha = self.TRACKING_SMOOTHING
                self._target_x = (1.0 - alpha) * self._target_x + alpha * proposed
            target = self._target_x
            self._locked_target_x = None
            self._last_descent_velocity = None
        self._was_descending = descending

        paddle_x = float(state["paddle_x"])
        target = min(max_x, max(0.0, target))
        delta = target - paddle_x
        # Commands are discrete 28 px moves. Stop within half a step so the
        # paddle does not bounce back and forth across a fractional target.
        if abs(delta) <= self.DEAD_BAND:
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
