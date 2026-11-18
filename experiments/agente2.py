"""Arkanoid agent that chooses rebounds by full-flight points per second.

Run with: python agente2.py --id <student-number>

The agent predicts the next paddle intercept, then compares the three paddle
regions using Monte Carlo samples from the game's triangular bounce-angle
distributions. It maximizes estimated points per second among safe catches.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
import time
from typing import Any

from agents.base_agent import BaseAgent
from experiments.arkanoid_domain import ArkanoidSearchDomain, ArkanoidState

AGENT_ID: str | None = None
MAX_SIM_EVENTS = 2048
CollisionTables = tuple[
    list[float], list[float], list[float], list[float],
    list[int], list[int], list[int], list[bool], int,
]
RouteStats = dict[tuple[int, int], tuple[int, float, float]]


class Agent2(BaseAgent):
    """Predictive Arkanoid controller with robust shot selection."""

    # Randomized strata reduce Monte Carlo noise in each triangular angle cone.
    ANGLE_SAMPLES = 512
    # A small dead band avoids alternating EAST/WEST when a target lies between
    # two discrete 28 px paddle positions.
    DEAD_BAND = 14.0
    # Replan after a collision changes the velocity; otherwise lock one region
    # throughout the descent so paddle movement remains consistent.
    VELOCITY_CHANGE_EPSILON = 2.0
    SAFETY_TICKS = 2
    LOOP_SEEK_PROBABILITY = 0.90
    LEVEL_ONE_LOOP_PROBABILITY = 0.70
    MIN_BRICKS_FOR_LOOP = 5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The server publishes its own wall-clock acceleration in each state.
        # Keep normal speed as a safe fallback for older server versions.
        self.acceleration = 1.0
        self._last_planning_seconds = 0.0
        self._descending = False
        self._last_velocity: tuple[float, float] | None = None
        self._locked_target: float | None = None
        self._ascending_target: float | None = None
        self._last_stage: Any = None
        self._last_lives: Any = None
        self._rng = random.Random()

    def _reachable_interval(
        self, domain: ArkanoidSearchDomain, state: ArkanoidState, seconds: float,
        safety_ticks: int,
    ) -> tuple[float, float]:
        steps = max(0, math.ceil(seconds * domain.fps) - safety_ticks)
        travel = steps * domain.paddle_step_distance
        return (
            max(0.0, state.paddle_x - travel),
            min(domain.width - domain.paddle_width, state.paddle_x + travel),
        )

    @staticmethod
    def _robust_zone_target(
        domain: ArkanoidSearchDomain,
        state: ArkanoidState,
        landing_x: float,
        seconds: float,
        action: str,
        safety_ticks: int,
    ) -> tuple[float, float] | None:
        """Choose an actually reachable paddle step inside the requested zone.

        The target is selected from discrete 28 px paddle positions, not from
        an ideal continuous coordinate. It maximizes the smaller of the catch
        margin and the distance to the neighboring bounce-zone boundary.
        """
        max_x = domain.width - domain.paddle_width
        steps = max(0, math.ceil(seconds * domain.fps) - safety_ticks)
        positions = {
            min(max_x, max(0.0, state.paddle_x + step * domain.paddle_step_distance))
            for step in range(-steps, steps + 1)
        }
        left_limit = domain.paddle_width / 3.0
        right_limit = 2.0 * domain.paddle_width / 3.0
        best: tuple[float, float] | None = None

        for target in positions:
            relative = landing_x - target
            catch_margin = min(
                relative + domain.ball_radius,
                domain.paddle_width + domain.ball_radius - relative,
            )
            if action == domain.LEFT and relative < left_limit:
                zone_margin = left_limit - relative
            elif action == domain.CENTER and left_limit <= relative <= right_limit:
                zone_margin = min(relative - left_limit, right_limit - relative)
            elif action == domain.RIGHT and relative > right_limit:
                zone_margin = relative - right_limit
            else:
                continue

            # Keep at least a small cushion for trajectory rounding and
            # continuous-collision timing differences.
            robust_margin = min(catch_margin, zone_margin) - 2.0
            if robust_margin < 2.0:
                continue
            candidate = (robust_margin, target)
            if best is None or candidate[0] > best[0]:
                best = candidate

        return (best[1], best[0]) if best is not None else None

    def _safety_tick_budget(self, domain: ArkanoidSearchDomain) -> int:
        """Convert recent planning latency into accelerated simulation ticks."""
        latency_ticks = math.ceil(
            self._last_planning_seconds * self.acceleration * domain.fps
        )
        return self.SAFETY_TICKS + latency_ticks

    def _sampled_rate(
        self,
        domain: ArkanoidSearchDomain,
        action: str,
        landing_x: float,
        inbound_seconds: float,
        brick_rows: dict[int, list[int]],
        collision_tables: CollisionTables,
        sampled_angles: list[float],
    ) -> tuple[float, float, float, float, float, float, float, RouteStats]:
        """Estimate score rate and probabilities of loops and metal-to-target routes."""
        rates: list[float] = []
        total_points = 0.0
        total_seconds = 0.0
        loops = 0
        metal_hits = 0
        metal_loops = 0
        roof_block_loops = 0
        route_totals: dict[tuple[int, int], list[float]] = {}
        for angle in sampled_angles:
            (points, _, flight_seconds, _, looped, hit_metal, metal_loop,
             roof_block_loop, route) = self._simulate_shot_fast(
                domain, landing_x, angle, brick_rows, collision_tables
            )
            loops += int(looped)
            metal_hits += int(hit_metal)
            metal_loops += int(metal_loop)
            roof_block_loops += int(roof_block_loop)
            flight_time = max(0.05, inbound_seconds + flight_seconds)
            rates.append(points / flight_time)
            total_points += points
            total_seconds += flight_time
            if route is not None:
                route_total = route_totals.setdefault(route, [0.0, 0.0, 0.0])
                route_total[0] += 1
                route_total[1] += points
                route_total[2] += flight_time

        rates.sort()
        lower_tail = rates[max(0, len(rates) // 10 - 1)]
        return (
            total_points / max(0.05, total_seconds),
            lower_tail,
            sum(rate > 0.0 for rate in rates) / len(rates),
            loops / len(sampled_angles),
            metal_hits / len(sampled_angles),
            metal_loops / len(sampled_angles),
            roof_block_loops / len(sampled_angles),
            {
                route: (int(values[0]), values[1], values[2])
                for route, values in route_totals.items()
            },
        )

    @staticmethod
    def _simulate_shot_fast(
        domain: ArkanoidSearchDomain,
        start_x: float,
        angle_deg: float,
        brick_rows: dict[int, list[int]],
        collision_tables: CollisionTables,
    ) -> tuple[float, float, float, float, bool, bool, bool, bool, tuple[int, int] | None]:
        """Simulate one full shot with event-based swept collisions.

        The game uses axis-aligned expanded-box collision checks, so jumping
        directly from collision to collision is both faster and more faithful
        than taking hundreds of tiny time steps. Fractional points credit
        damage on multi-hit bricks toward their eventual score.
        """
        radius = domain.ball_radius
        x, y = start_x, domain.paddle_y - radius
        angle = math.radians(angle_deg)
        vx = domain.ball_speed * math.sin(angle)
        vy = -domain.ball_speed * math.cos(angle)
        ceiling = domain.ceiling_y + radius
        lefts, rights, tops, bottoms, initial_health, hit_requirements, points_by_index, metals, remaining = collision_tables
        health = initial_health.copy()
        seen_rows = [0] * len(health)
        event_id = 0
        last_brick: int | None = None
        elapsed = 0.0
        points = 0.0
        progress = 0.0
        first_target_rate = 0.0
        hit_metal = False
        metal_loop = False
        first_metal_index: int | None = None
        metal_to_target: tuple[int, int] | None = None
        # Repeated position, velocity, and brick-health state means the
        # deterministic trajectory has entered a closed cycle.
        seen_states: dict[tuple[float, float, float, float, int | None], tuple[int, int]] = {}
        ceiling_bounces = 0
        brick_bounces = 0
        max_seconds = domain.max_flight_seconds
        paddle_plane = domain.paddle_y - radius

        for _ in range(MAX_SIM_EVENTS):
            loop_key = (
                round(x, 3), round(y, 3), round(vx, 3), round(vy, 3),
                last_brick,
            )
            previous_counts = seen_states.get(loop_key)
            if previous_counts is not None:
                roof_block_loop = (
                    ceiling_bounces > previous_counts[0]
                    and brick_bounces > previous_counts[1]
                )
                return (points, progress, elapsed, first_target_rate, True,
                        hit_metal, metal_loop, roof_block_loop, metal_to_target)
            seen_states[loop_key] = (ceiling_bounces, brick_bounces)

            if last_brick is not None:
                if not (
                    lefts[last_brick] <= x <= rights[last_brick]
                    and tops[last_brick] <= y <= bottoms[last_brick]
                ):
                    last_brick = None

            # Find the next non-brick event first; bricks can only collide
            # before that boundary or paddle crossing.
            boundary_time = max_seconds - elapsed
            boundary_kind = "limit"
            if vx > 0.0:
                candidate = (domain.width - radius - x) / vx
                if 0.0 <= candidate < boundary_time:
                    boundary_time, boundary_kind = candidate, "wall"
            elif vx < 0.0:
                candidate = (radius - x) / vx
                if 0.0 <= candidate < boundary_time:
                    boundary_time, boundary_kind = candidate, "wall"
            if vy < 0.0:
                candidate = (ceiling - y) / vy
                if 0.0 <= candidate < boundary_time:
                    boundary_time, boundary_kind = candidate, "ceiling"
            elif vy > 0.0:
                candidate = (paddle_plane - y) / vy
                if 0.0 <= candidate < boundary_time:
                    boundary_time, boundary_kind = candidate, "paddle"

            brick_time = boundary_time
            hit_index: int | None = None
            hit_nx = hit_ny = 0.0
            event_id += 1
            end_y = y + vy * boundary_time
            if vy >= 0.0:
                row_start = math.floor(y / 24.0) - 1
                row_stop = math.floor(end_y / 24.0) + 2
            else:
                row_start = math.floor(end_y / 24.0) - 1
                row_stop = math.floor(y / 24.0) + 2
            for row in range(row_start, row_stop):
                for index in brick_rows.get(row, ()):
                    if seen_rows[index] == event_id:
                        continue
                    seen_rows[index] = event_id
                    if index == last_brick or (
                        health[index] <= 0 and not metals[index]
                    ):
                        continue
                    left, right = lefts[index], rights[index]
                    top, bottom = tops[index], bottoms[index]

                    if vx > 0.0:
                        t1x, t2x, nx = (left - x) / vx, (right - x) / vx, -1.0
                    elif vx < 0.0:
                        t1x, t2x, nx = (right - x) / vx, (left - x) / vx, 1.0
                    elif left <= x <= right:
                        t1x, t2x, nx = -math.inf, math.inf, 0.0
                    else:
                        continue

                    if vy > 0.0:
                        t1y, t2y, ny = (top - y) / vy, (bottom - y) / vy, -1.0
                    elif vy < 0.0:
                        t1y, t2y, ny = (bottom - y) / vy, (top - y) / vy, 1.0
                    elif top <= y <= bottom:
                        t1y, t2y, ny = -math.inf, math.inf, 0.0
                    else:
                        continue

                    enter, leave = max(t1x, t1y), min(t2x, t2y)
                    if enter <= leave and leave >= 0.0:
                        normal_x = nx if t1x > t1y else 0.0
                        normal_y = ny if t1y >= t1x else 0.0
                        if vx * normal_x + vy * normal_y < 0.0:
                            collision_time = max(0.0, enter)
                            if collision_time < brick_time:
                                brick_time, hit_index = collision_time, index
                                hit_nx, hit_ny = normal_x, normal_y

            if hit_index is None:
                x += vx * boundary_time
                y += vy * boundary_time
                elapsed += boundary_time
                if boundary_kind == "wall":
                    vx = -vx
                    x = min(domain.width - radius, max(radius, x))
                    continue
                if boundary_kind == "ceiling":
                    ceiling_bounces += 1
                    vy = -vy
                    y = ceiling
                    continue
                return (points, progress, elapsed, first_target_rate, False,
                        hit_metal, False, False, metal_to_target)

            x += vx * brick_time
            y += vy * brick_time
            elapsed += brick_time
            brick_bounces += 1
            if hit_nx:
                vx = -vx
                x += hit_nx * 0.01
            elif hit_ny:
                vy = -vy
                y += hit_ny * 0.01
            if metals[hit_index]:
                hit_metal = True
                metal_loop = True
                if first_metal_index is None:
                    first_metal_index = hit_index
            else:
                if first_metal_index is not None and metal_to_target is None:
                    metal_to_target = (first_metal_index, hit_index)
                if first_target_rate == 0.0:
                    first_target_rate = points_by_index[hit_index] / max(
                        0.05, elapsed * hit_requirements[hit_index]
                    )
                progress += points_by_index[hit_index] / hit_requirements[hit_index]
                health[hit_index] -= 1
                # The board changed, so only detect cycles after the new
                # health layout has stabilized.
                seen_states.clear()
                metal_loop = False
                if health[hit_index] <= 0:
                    points += points_by_index[hit_index]
                    remaining -= 1
                    if remaining <= 0:
                        return (points, progress, elapsed, first_target_rate, False,
                                hit_metal, False, False, metal_to_target)
            last_brick = hit_index

            # If another event begins while still in the same box, advance a
            # tiny amount along the reflected direction to avoid a zero-time hit.
            if brick_time < 1e-9:
                x += vx * 1e-5
                y += vy * 1e-5

        speed_y = max(abs(vy), 1e-6)
        if vy > 0.0:
            return_seconds = max(0.0, (paddle_plane - y) / speed_y)
        else:
            return_seconds = max(0.0, (y - ceiling) / speed_y)
            return_seconds += max(0.0, (paddle_plane - ceiling) / speed_y)
        return (points, progress, elapsed + return_seconds, first_target_rate, False,
                hit_metal, False, False, metal_to_target)

    def _select_target(
        self, domain: ArkanoidSearchDomain, state: ArkanoidState
    ) -> tuple[str, float, float]:
        planning_started = time.perf_counter()
        landing_x, inbound_seconds = domain.predict_landing(state)
        safety_ticks = self._safety_tick_budget(domain)
        low, high = self._reachable_interval(
            domain, state, inbound_seconds, safety_ticks
        )
        paddle_max = domain.width - domain.paddle_width

        # Plan the rebound even while the ball rises. That gives the paddle
        # time to reach the chosen contact zone; any later collision changes
        # the observed velocity and triggers a fresh plan.
        # One row per reachable zone: action, target, rates, hit/loop/metal
        # probabilities, safety, and metal-to-target routes.
        candidate_table: list[
            tuple[str, float, float, float, float, float, float, float, float, float, RouteStats]
        ] = []
        # Shared randomized strata give every zone comparable coverage of its
        # angle distribution while retaining Monte Carlo sampling.
        angle_quantiles = [
            (sample + self._rng.random()) / self.ANGLE_SAMPLES
            for sample in range(self.ANGLE_SAMPLES)
        ]
        row_height = 24.0
        brick_rows: dict[int, list[int]] = {}
        lefts: list[float] = []
        rights: list[float] = []
        tops: list[float] = []
        bottoms: list[float] = []
        initial_health: list[int] = []
        hit_requirements: list[int] = []
        points_by_index: list[int] = []
        metals: list[bool] = []
        remaining = 0
        for index, brick in enumerate(state.bricks):
            lefts.append(brick.x - domain.ball_radius)
            rights.append(brick.x + brick.width + domain.ball_radius)
            tops.append(brick.y - domain.ball_radius)
            bottoms.append(brick.y + brick.height + domain.ball_radius)
            initial_health.append(brick.health)
            hit_requirements.append(max(1, brick.health))
            points_by_index.append(brick.points)
            metals.append(brick.indestructible)
            if brick.health <= 0 and not brick.indestructible:
                continue
            if not brick.indestructible:
                remaining += 1
            first = math.floor((brick.y - domain.ball_radius) / row_height)
            last = math.floor((brick.y + brick.height + domain.ball_radius) / row_height)
            for row in range(first, last + 1):
                brick_rows.setdefault(row, []).append(index)

        collision_tables: CollisionTables = (
            lefts, rights, tops, bottoms, initial_health,
            hit_requirements, points_by_index, metals, remaining,
        )

        for action in domain.ACTIONS:
            target_choice = self._robust_zone_target(
                domain, state, landing_x, inbound_seconds, action, safety_ticks
            )
            if target_choice is None:
                continue
            target, edge_margin = target_choice

            _, mode = domain.REGIONS[action]
            half_width = 8.0 if action == domain.CENTER else 15.0
            sampled_angles = [
                domain._triangular_quantile(
                    quantile, mode - half_width, mode + half_width, mode
                )
                for quantile in angle_quantiles
            ]
            (mean_rate, lower_rate, hit_probability, loop_probability,
             metal_hit_probability, metal_loop_probability, roof_block_probability,
             route_stats) = self._sampled_rate(
                domain, action, landing_x, inbound_seconds,
                brick_rows, collision_tables, sampled_angles,
            )
            # Maximize the estimated real score rate directly. Safety only
            # breaks exact rate ties; per-brick and loop stats stay diagnostic.
            safety = min(1.0, edge_margin / (domain.paddle_width / 2))
            candidate_table.append((
                action, target, mean_rate, lower_rate, hit_probability,
                loop_probability, metal_hit_probability, metal_loop_probability,
                roof_block_probability, safety, route_stats,
            ))

        if candidate_table:
            # The objective is the complete simulated flight's total points
            # divided by elapsed time. A roof-and-brick loop only breaks an
            # exact score-rate tie; it cannot displace a better direct shot.
            can_seek_loop = remaining > self.MIN_BRICKS_FOR_LOOP
            loop_probability_threshold = (
                self.LEVEL_ONE_LOOP_PROBABILITY
                if domain.level_number == 1 else self.LOOP_SEEK_PROBABILITY
            )
            best = max(
                candidate_table,
                key=lambda row: (
                    row[2], can_seek_loop and row[8] >= loop_probability_threshold,
                    row[8] if can_seek_loop and row[8] >= loop_probability_threshold else 0.0,
                    row[9],
                ),
            )
            self._last_planning_seconds = time.perf_counter() - planning_started
            readable_table = []
            for row in candidate_table:
                routes = []
                for (metal_index, target_index), (count, points, seconds) in sorted(
                    row[10].items(), key=lambda item: (item[1][0], item[1][1]), reverse=True
                )[:3]:
                    metal = state.bricks[metal_index]
                    target_brick = state.bricks[target_index]
                    routes.append(
                        f"M{metal_index}@({metal.x:.0f},{metal.y:.0f})"
                        f"->B{target_index}@({target_brick.x:.0f},{target_brick.y:.0f})"
                        f"[{target_brick.points}pts,{target_brick.health}hp]"
                        f" p={count / self.ANGLE_SAMPLES:.0%}"
                        f" pts/s={points / max(0.05, seconds):.2f}"
                    )
                readable_table.append((
                    row[0], round(row[1], 1), round(row[2], 2), round(row[3], 2),
                    round(row[4], 2), round(row[5], 2), round(row[6], 2),
                    round(row[7], 2), round(row[8], 2), round(row[9], 2), routes,
                ))
            logging.info(
                "Agent2 por zona (ação, alvo, pts/s, cauda, acerto, loop, toque metal, loop metal, loop teto+tijolo, segurança, rotas metal→tijolo): %s",
                readable_table,
            )
            if can_seek_loop and best[8] >= loop_probability_threshold:
                logging.info(
                    "Escolhida trajetória em loop teto+tijolo no melhor pts/s: probabilidade %.0f%%, %.2f pts/s",
                    100 * best[8], best[2],
                )
            return best[0], best[1], best[2]

        # No offensive zone is safely reachable. Make the best catch possible.
        catch = min(paddle_max, max(0.0, landing_x - domain.paddle_width / 2))
        self._last_planning_seconds = time.perf_counter() - planning_started
        return domain.CENTER, min(high, max(low, catch)), 0.0

    def _move_toward(self, target: float, state: dict[str, Any]) -> dict[str, Any] | None:
        paddle_x = float(state["paddle_x"])
        delta = target - paddle_x
        if abs(delta) <= self.DEAD_BAND:
            return None
        direction = "EAST" if delta > 0 else "WEST"
        actions = state.get("valid_actions", state.get("actions", []))
        return next((item for item in actions if item.get("direction") == direction), None)

    async def deliberate(self) -> dict[str, Any] | None:
        observation = self.current_state
        if not observation or observation.get("game_over") or observation.get("game_won"):
            return None

        try:
            reported_acceleration = float(observation.get("acceleration", self.acceleration))
        except (TypeError, ValueError):
            reported_acceleration = self.acceleration
        if math.isfinite(reported_acceleration) and reported_acceleration > 0.0:
            self.acceleration = reported_acceleration

        velocity = (float(observation["ball_vx"]), float(observation["ball_vy"]))
        descending = velocity[1] > 0
        stage = observation.get("stage", observation.get("level"))
        lives = observation.get("lives")
        reset = (
            (self._last_stage is not None and stage != self._last_stage)
            or (self._last_lives is not None and lives != self._last_lives)
        )
        changed = (
            reset
            or (
                self._last_velocity is not None
                and (
                    abs(velocity[0] - self._last_velocity[0]) > self.VELOCITY_CHANGE_EPSILON
                    or abs(velocity[1] - self._last_velocity[1]) > self.VELOCITY_CHANGE_EPSILON
                )
            )
        )

        # Most frames have identical velocity and are just ball travel. Reuse
        # the previous target without reparsing all bricks or re-running the
        # trajectory planner; only collisions require a new prediction.
        if descending and self._descending and self._locked_target is not None and not changed:
            self._last_velocity = velocity
            self._last_stage, self._last_lives = stage, lives
            return self._move_toward(self._locked_target, observation)
        if (
            not descending
            and not self._descending
            and self._ascending_target is not None
            and not changed
        ):
            self._last_velocity = velocity
            self._last_stage, self._last_lives = stage, lives
            return self._move_toward(self._ascending_target, observation)

        domain, state = ArkanoidSearchDomain.from_observation(observation)
        if descending:
            self._locked_target = self._select_target(domain, state)[1]
            self._ascending_target = None
        else:
            self._locked_target = None
            _, self._ascending_target, _ = self._select_target(domain, state)

        self._descending = descending
        self._last_velocity = velocity
        self._last_stage, self._last_lives = stage, lives
        target = self._locked_target if descending else self._ascending_target
        if target is None:
            target = float(observation["paddle_x"])
        return self._move_toward(target, observation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive Arkanoid agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Arkanoid Agent 2")
    args = parser.parse_args()
    asyncio.run(Agent2(args.server, args.name).run())


if __name__ == "__main__":
    main()
