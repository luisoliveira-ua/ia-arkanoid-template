"""Arkanoid agent that chooses paddle moves using the search domain."""

import argparse
import asyncio
import logging
from typing import Any

from agents.base_agent import BaseAgent
from arkanoid_domain import ArkanoidSearchDomain, search_decision


AGENT_ID: str | None = None


class SearchAgent(BaseAgent):
    """Replan a short sequence of rebounds and follow its paddle target."""

    # Paddle commands move a fixed 28 px per game tick. A smaller dead band
    # makes the controller alternate EAST/WEST around an unreachable target.
    DEAD_BAND = 16.0
    TRACKING_SMOOTHING = 0.25
    SEARCH_HORIZON = 2
    SEARCH_NODE_LIMIT = 40

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_prediction_log = -1.0
        self._was_descending = False
        self._locked_zone: str | None = None
        self._locked_hit: dict[str, float | int | None] | None = None
        self._target_x: float | None = None
        self._last_descent_velocity: tuple[float, float] | None = None

    async def deliberate(self) -> dict[str, Any] | None:
        state = self.current_state
        if not state or state.get("game_over") or state.get("game_won"):
            return None

        # Keep the per-tick decision path short; planning diagnostics must not
        # delay paddle commands when the simulation is accelerated.
        domain, initial = ArkanoidSearchDomain.from_observation(state)
        level_time = float(state.get("level_time_elapsed", 0.0))

        # Use the impact/trajectory prediction and reachable rebound-zone
        # scoring on every tick for immediate paddle control.
        descending = float(state["ball_vy"]) > 0
        velocity = (float(state["ball_vx"]), float(state["ball_vy"]))
        if descending:
            # Pick once per descent. Replanning the zone every frame made the
            # paddle alternate between nearby targets. Replan if a collision
            # changes the trajectory during that descent.
            trajectory_changed = (
                self._last_descent_velocity is not None
                and (abs(velocity[0] - self._last_descent_velocity[0]) > 1.0
                     or abs(velocity[1] - self._last_descent_velocity[1]) > 1.0)
            )
            if (not self._was_descending or self._locked_zone is None or self._target_x is None
                    or trajectory_changed):
                _, _, plan, planned_actions = search_decision(
                    state,
                    horizon=self.SEARCH_HORIZON,
                    node_limit=self.SEARCH_NODE_LIMIT,
                )
                greedy_zone, greedy_target, greedy_hit = domain.choose_paddle_target(initial)
                planned_zone = planned_actions[0] if planned_actions else None
                planned_target = (
                    domain.safe_paddle_target(initial, planned_zone)
                    if planned_zone is not None else None
                )
                if plan and len(plan) > 1 and planned_zone is not None and planned_target is not None:
                    planned_hit = domain.predict_first_brick(initial, planned_zone)
                    # Compare both choices by expected points per second.
                    # Tree actions are already filtered against predicted life loss.
                    plan_rate = domain.points_per_second(plan[-1])
                    greedy_rate = float(
                        greedy_hit.get("expected_points_per_second",
                                        greedy_hit.get("full_flight_points_per_second", 0.0)) or 0.0
                    )
                    if plan_rate > greedy_rate:
                        self._locked_zone = planned_zone
                        self._target_x = planned_target
                        hit = planned_hit
                    else:
                        self._locked_zone = greedy_zone
                        self._target_x = greedy_target
                        hit = greedy_hit
                else:
                    self._locked_zone = greedy_zone
                    self._target_x = greedy_target
                    hit = greedy_hit
                self._locked_hit = hit
            else:
                hit = self._locked_hit or {
                    "brick_index": None, "points": 0, "seconds": 0.0,
                    "hit_probability": 0.0,
                }
            selected_zone = self._locked_zone or domain.CENTER
            target_x = self._target_x if self._target_x is not None else float(state["paddle_x"])
            self._last_descent_velocity = velocity
        else:
            # While the ball rises, track its predicted return near paddle
            # center, filtering small prediction changes from frame to frame.
            selected_zone, proposed_x, hit = domain.choose_paddle_target(initial)
            if self._target_x is None:
                self._target_x = proposed_x
            else:
                alpha = self.TRACKING_SMOOTHING
                self._target_x = (1.0 - alpha) * self._target_x + alpha * proposed_x
            target_x = self._target_x
            self._locked_zone = None
            self._locked_hit = None
            self._last_descent_velocity = None
        self._was_descending = descending

        if level_time - self._last_prediction_log >= 1.0:
            if hit["brick_index"] is None:
                logging.info(
                    "Ressalto escolhido: %s; nenhum tijolo previsto nos cenários simulados, a priorizar a interceção",
                    selected_zone,
                )
            else:
                logging.info(
                    "Ressalto escolhido: %s; tijolo %s (%s pontos), probabilidade estimada %.0f%%, impacto em %.2f s",
                    selected_zone,
                    hit["brick_index"],
                    hit["points"],
                    100 * float(hit.get("hit_probability", 0.0) or 0.0),
                    hit["seconds"],
                )
            self._last_prediction_log = level_time

        paddle_x = float(state["paddle_x"])
        delta = target_x - paddle_x
        if abs(delta) <= max(self.DEAD_BAND, domain.paddle_step_distance * 0.55):
            return None

        direction = "EAST" if delta > 0 else "WEST"
        valid_actions = state.get("valid_actions", state.get("actions", []))
        return next(
            (action for action in valid_actions if action.get("direction") == direction),
            None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Arkanoid search agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Arkanoid Search Agent")
    parser.add_argument("--id", default=None)
    args = parser.parse_args()
    asyncio.run(SearchAgent(args.server, args.name, args.id or AGENT_ID).run())


if __name__ == "__main__":
    main()
