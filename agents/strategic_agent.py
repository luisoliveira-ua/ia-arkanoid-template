"""Risk-aware Arkanoid agent with sampled trajectory planning.

At each descent the agent compares reachable paddle zones over a set of angles
from the game's triangular bounce distributions. It values score rate while
keeping a wide catch margin, then tracks the chosen intercept without oscillating.
"""

from __future__ import annotations

import math
from typing import Any

from agents.base_agent import BaseAgent
from arkanoid_domain import ArkanoidSearchDomain, ArkanoidState


class StrategicAgent(BaseAgent):
    """Choose safe, high-throughput rebounds using sampled shot outcomes."""

    DEAD_BAND = 13.0
    TRACKING_SMOOTHING = 0.35
    ANGLE_SAMPLES = 11
    SAFETY_RESERVE_TICKS = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._target_x: float | None = None
        self._locked_target: float | None = None
        self._last_velocity: tuple[float, float] | None = None
        self._was_descending = False
        self._decision: tuple[str, float] | None = None

    @staticmethod
    def _brick_rows(state: ArkanoidState, radius: float) -> tuple[dict[int, list[int]], float]:
        row_height = 24.0
        rows: dict[int, list[int]] = {}
        for index, brick in enumerate(state.bricks):
            if brick.health <= 0 and not brick.indestructible:
                continue
            first = math.floor((brick.y - radius) / row_height)
            last = math.floor((brick.y + brick.height + radius) / row_height)
            for row in range(first, last + 1):
                rows.setdefault(row, []).append(index)
        return rows, row_height

    def _reachable_range(self, domain: ArkanoidSearchDomain, state: ArkanoidState,
                         seconds: float) -> tuple[float, float]:
        ticks = max(0, math.ceil(seconds * domain.fps) - self.SAFETY_RESERVE_TICKS)
        distance = ticks * domain.paddle_step_distance
        return (max(0.0, state.paddle_x - distance),
                min(domain.width - domain.paddle_width, state.paddle_x + distance))

    def _evaluate_zone(self, domain: ArkanoidSearchDomain, state: ArkanoidState,
                       action: str, landing_x: float, inbound_seconds: float,
                       rows: dict[int, list[int]], row_height: float) -> tuple[float, float] | None:
        fraction, mode = domain.REGIONS[action]
        paddle_x = min(domain.width - domain.paddle_width,
                       max(0.0, landing_x - fraction * domain.paddle_width))
        reachable_low, reachable_high = self._reachable_range(domain, state, inbound_seconds)
        if not reachable_low <= paddle_x <= reachable_high:
            return None

        contact = landing_x - paddle_x
        margin = min(contact, domain.paddle_width - contact)
        # Keep the impact well inside the selected third, allowing for prediction
        # error and the server's one-tick action/observation delay.
        required_margin = max(domain.ball_radius + domain.paddle_step_distance * 0.55 + 2.0,
                              domain.paddle_width * 0.25)
        if margin < required_margin:
            return None

        half_width = 8.0 if action == domain.CENTER else 15.0
        rates: list[float] = []
        for sample in range(self.ANGLE_SAMPLES):
            q = (sample + 0.5) / self.ANGLE_SAMPLES
            angle = domain._triangular_quantile(q, mode - half_width,
                                                 mode + half_width, mode)
            points, flight_seconds = domain._simulate_shot(
                state, landing_x, angle, rows, row_height
            )
            rates.append(points / max(0.05, inbound_seconds + flight_seconds))

        rates.sort()
        mean_rate = sum(rates) / len(rates)
        # A modest lower-tail term prefers zones that score consistently across
        # the random bounce cone, while retaining expected throughput as the goal.
        lower_quartile = rates[max(0, len(rates) // 4 - 1)]
        robustness = margin / max(1.0, domain.paddle_width / 2)
        utility = 0.82 * mean_rate + 0.18 * lower_quartile
        utility *= 0.94 + 0.06 * min(1.0, robustness)
        return utility, paddle_x

    def _choose_target(self, domain: ArkanoidSearchDomain,
                       state: ArkanoidState) -> tuple[str, float]:
        landing_x, inbound_seconds = domain.predict_landing(state)
        if state.ball_vy <= 0:
            low, high = self._reachable_range(domain, state, inbound_seconds)
            target = min(domain.width - domain.paddle_width,
                         max(0.0, landing_x - domain.paddle_width / 2))
            return domain.CENTER, min(high, max(low, target))

        rows, row_height = self._brick_rows(state, domain.ball_radius)
        best: tuple[float, str, float] | None = None
        for action in domain.ACTIONS:
            evaluated = self._evaluate_zone(domain, state, action, landing_x,
                                            inbound_seconds, rows, row_height)
            if evaluated is None:
                continue
            utility, target = evaluated
            candidate = (utility, action, target)
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is not None:
            return best[1], best[2]

        # If no offensive zone can be reached with a robust margin, center the
        # paddle under the predicted landing point to preserve the life.
        low, high = self._reachable_range(domain, state, inbound_seconds)
        catch = min(domain.width - domain.paddle_width,
                    max(0.0, landing_x - domain.paddle_width / 2))
        return domain.CENTER, min(high, max(low, catch))

    async def deliberate(self) -> dict[str, Any] | None:
        observation = self.current_state
        if not observation or observation.get("game_over") or observation.get("game_won"):
            return None

        domain, initial = ArkanoidSearchDomain.from_observation(observation)
        descending = initial.ball_vy > 0
        velocity = (initial.ball_vx, initial.ball_vy)
        changed = (self._last_velocity is not None
                   and (abs(velocity[0] - self._last_velocity[0]) > 1.0
                        or abs(velocity[1] - self._last_velocity[1]) > 1.0))

        if descending and (not self._was_descending or self._decision is None or changed):
            self._decision = self._choose_target(domain, initial)
            self._locked_target = self._decision[1]
        elif not descending:
            _, proposed = self._choose_target(domain, initial)
            if self._target_x is None:
                self._target_x = proposed
            else:
                alpha = self.TRACKING_SMOOTHING
                self._target_x = (1.0 - alpha) * self._target_x + alpha * proposed
            self._decision = None
            self._locked_target = None

        if descending:
            self._last_velocity = velocity
        else:
            self._last_velocity = None
        self._was_descending = descending

        target = self._locked_target if descending else self._target_x
        if target is None:
            target = initial.paddle_x
        delta = target - initial.paddle_x
        if abs(delta) <= max(self.DEAD_BAND, domain.paddle_step_distance * 0.45):
            return None

        direction = "EAST" if delta > 0 else "WEST"
        actions = observation.get("valid_actions", observation.get("actions", []))
        return next((item for item in actions if item.get("direction") == direction), None)
