"""Risk-aware Arkanoid agent with sampled trajectory planning.

At each descent the agent compares reachable paddle zones over a set of angles
from the game's triangular bounce distributions. It values score rate while
keeping a wide catch margin, then tracks the chosen intercept without oscillating.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from arkanoid_domain import ArkanoidSearchDomain, ArkanoidState


class StrategicAgent(BaseAgent):
    """Choose safe, high-throughput rebounds using sampled shot outcomes."""

    DEAD_BAND = 15.5
    TRACKING_SMOOTHING = 0.35
    # Randomized strata cover the complete triangular angle distribution while
    # keeping zone comparisons paired on the same set of angle quantiles.
    ANGLE_SAMPLES = 512
    SAFETY_RESERVE_TICKS = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._target_x: float | None = None
        self._locked_target: float | None = None
        self._last_velocity: tuple[float, float] | None = None
        self._was_descending = False
        self._decision: tuple[str, float] | None = None
        self._rng = random.Random()
        self._learned_profile = self._load_learned_profile()
        self._learned_cells = self._learned_profile.get("cells", {})
        self._learned_policy = self._learned_profile.get("policy", {})

    @staticmethod
    def _load_learned_profile() -> dict[str, Any]:
        """Load the trained contextual policy and action values."""
        profile_path = (
            Path(__file__).resolve().parents[1]
            / "training_data"
            / "learned_policy.json"
        )
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return profile if isinstance(profile, dict) else {}

    @staticmethod
    def _brick_bucket(count: int) -> str:
        if count <= 10:
            return "0-10"
        if count <= 20:
            return "11-20"
        if count <= 30:
            return "21-30"
        return "31+"

    def _learned_prior(self, domain: ArkanoidSearchDomain, state: ArkanoidState,
                       landing_x: float, action: str) -> tuple[float, int] | None:
        landing_bucket = min(2, max(0, int(3.0 * landing_x / domain.width)))
        level = str(domain.level_number)
        density = self._brick_bucket(state.remaining_bricks)
        keys = (
            f"{level}|{density}|{landing_bucket}|{action}",
            f"{level}|{density}|*|{action}",
            f"{level}|*|{landing_bucket}|{action}",
            f"{level}|*|*|{action}",
            f"*|{density}|*|{action}",
            f"*|*|*|{action}",
        )
        for key in keys:
            cell = self._learned_cells.get(key)
            if not cell:
                continue
            samples = int(cell.get("samples", 0))
            if samples >= 32:
                return float(cell.get("mean_points_per_second", 0.0)), samples
        return None

    def _learned_action(self, domain: ArkanoidSearchDomain, state: ArkanoidState,
                        landing_x: float) -> str | None:
        """Get the best action learned for this board and landing context."""
        landing_bucket = min(2, max(0, int(3.0 * landing_x / domain.width)))
        level = str(domain.level_number)
        density = self._brick_bucket(state.remaining_bricks)
        keys = (
            f"{level}|{density}|{landing_bucket}",
            f"{level}|{density}|*",
            f"{level}|*|{landing_bucket}",
            f"{level}|*|*",
            f"*|{density}|*",
            "*|*|*",
        )
        for key in keys:
            learned = self._learned_policy.get(key)
            if learned and int(learned.get("samples", 0)) >= 64:
                return str(learned.get("action"))
        return None

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

    def _zone_target(self, domain: ArkanoidSearchDomain, state: ArkanoidState,
                     landing_x: float, seconds: float,
                     action: str) -> tuple[float, float] | None:
        """Choose the safest reachable discrete paddle position for a zone."""
        max_x = domain.width - domain.paddle_width
        steps = max(0, math.ceil(seconds * domain.fps) - self.SAFETY_RESERVE_TICKS)
        positions = {
            min(max_x, max(0.0, state.paddle_x + step * domain.paddle_step_distance))
            for step in range(-steps, steps + 1)
        }
        left_limit = domain.paddle_width * domain.paddle_left_ratio
        right_limit = domain.paddle_width * (1.0 - domain.paddle_right_ratio)
        best: tuple[float, float, float] | None = None

        for target in positions:
            relative = landing_x - target
            contact = min(domain.paddle_width, max(0.0, relative))
            catch_margin = min(
                relative + domain.ball_radius,
                domain.paddle_width + domain.ball_radius - relative,
            )
            if action == domain.LEFT and contact < left_limit:
                zone_margin = left_limit - contact
            elif action == domain.CENTER and left_limit <= contact <= right_limit:
                zone_margin = min(contact - left_limit, right_limit - contact)
            elif action == domain.RIGHT and contact > right_limit:
                zone_margin = contact - right_limit
            else:
                continue

            # Keep a small reserve both inside the paddle and inside the chosen
            # rebound region. The target itself is reachable in 28 px commands.
            robust_margin = min(catch_margin, zone_margin) - 2.0
            if robust_margin < 2.0:
                continue
            candidate = (robust_margin, -abs(target - state.paddle_x), target)
            if best is None or candidate > best:
                best = candidate

        return (best[2], best[0]) if best is not None else None

    def _evaluate_zone(self, domain: ArkanoidSearchDomain, state: ArkanoidState,
                       action: str, landing_x: float, inbound_seconds: float,
                       rows: dict[int, list[int]], row_height: float,
                       quantiles: list[float]) -> tuple[float, float, float] | None:
        target_choice = self._zone_target(
            domain, state, landing_x, inbound_seconds, action
        )
        if target_choice is None:
            return None
        paddle_x, safety_margin = target_choice

        mode, half_width = domain.angle_parameters(action)

        total_points = 0
        total_seconds = 0.0
        for q in quantiles:
            # Integrate over the actual triangular bounce distribution. The
            # game samples a fresh angle on every paddle hit; no single ray is
            # reliable enough to choose a zone by itself.
            angle = domain._triangular_quantile(q, mode - half_width,
                                                 mode + half_width, mode)
            points, flight_seconds = domain._simulate_shot_details(
                state, landing_x, angle, rows, row_height
            )
            duration = max(0.05, inbound_seconds + flight_seconds)
            total_points += points
            total_seconds += duration

        # Optimize total score over the total time required to earn it.
        utility = total_points / max(0.05, total_seconds)
        prior = self._learned_prior(domain, state, landing_x, action)
        if prior is not None:
            learned_rate, prior_samples = prior
            # Online physics remains dominant. The trained table contributes
            # a small, confidence-weighted prior for similar game situations.
            prior_weight = min(0.35, 0.50 * prior_samples / (prior_samples + 128.0))
            utility = (1.0 - prior_weight) * utility + prior_weight * learned_rate
        return utility, paddle_x, safety_margin

    def _choose_target(self, domain: ArkanoidSearchDomain,
                       state: ArkanoidState) -> tuple[str, float]:
        landing_x, inbound_seconds = domain.predict_landing(state)
        if state.ball_vy <= 0:
            low, high = self._reachable_range(domain, state, inbound_seconds)
            target = min(domain.width - domain.paddle_width,
                         max(0.0, landing_x - domain.paddle_width / 2))
            return domain.CENTER, min(high, max(low, target))

        rows, row_height = self._brick_rows(state, domain.ball_radius)
        # Randomized stratification gives every part of the game's triangular
        # angle distribution equal coverage. Sharing quantiles across zones
        # reduces noise when comparing their expected score rates.
        quantiles = [
            (sample + self._rng.random()) / self.ANGLE_SAMPLES
            for sample in range(self.ANGLE_SAMPLES)
        ]
        candidates: list[tuple[float, float, str, float]] = []
        for action in domain.ACTIONS:
            evaluated = self._evaluate_zone(domain, state, action, landing_x,
                                            inbound_seconds, rows, row_height, quantiles)
            if evaluated is None:
                continue
            utility, target, safety_margin = evaluated
            candidates.append((utility, safety_margin, action, target))

        if candidates:
            # Maximize the simulated score rate directly. Safety margin only
            # breaks exact score-rate ties; the trained contextual policy is
            # followed when its action stays within 3% of the best online rate.
            best = max(candidates, key=lambda row: (row[0], row[1]))
            learned_action = self._learned_action(domain, state, landing_x)
            selected = next(
                (
                    candidate for candidate in candidates
                    if candidate[2] == learned_action
                    and candidate[0] >= best[0] * 0.97
                ),
                best,
            )
            return selected[2], selected[3]

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
        if abs(delta) <= max(self.DEAD_BAND, domain.paddle_step_distance * 0.55):
            return None

        direction = "EAST" if delta > 0 else "WEST"
        actions = observation.get("valid_actions", observation.get("actions", []))
        return next((item for item in actions if item.get("direction") == direction), None)
