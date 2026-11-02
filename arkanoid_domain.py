"""Search problem for choosing Arkanoid paddle rebounds.

The search state is a snapshot at the moment the ball reaches the paddle.
Actions select the paddle third used for the rebound. ``result`` simulates
that flight until the ball returns to the paddle (or is lost), updating brick
health, score, and elapsed game time. The goal is to clear all destructible
bricks; search strategies can compare completed plans by points / seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from tree import SearchDomain, SearchProblem, SearchTree


@dataclass(frozen=True)
class BrickState:
    x: float
    y: float
    width: float
    height: float
    health: int
    points: int
    indestructible: bool = False


@dataclass(frozen=True)
class ArkanoidState:
    """A planning node containing a complete observed game state."""

    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    paddle_x: float
    bricks: tuple[BrickState, ...]
    score: int = 0
    elapsed: float = 0.0
    lives: int = 3
    last_hit_indices: tuple[int, ...] = ()

    @property
    def remaining_bricks(self) -> int:
        return sum(not b.indestructible and b.health > 0 for b in self.bricks)


class ArkanoidSearchDomain(SearchDomain):
    """SearchDomain-like interface: actions, result, cost, heuristic, satisfies.

    The three actions choose the left, center, or right third of the paddle.
    Physics is deterministic here: the center uses a vertical rebound and the
    side thirds use 45-degree rebounds. This approximates the game's angular
    cones so a tree-search algorithm can plan over repeatable states.
    """

    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    ACTIONS = (LEFT, CENTER, RIGHT)
    CENTER_HALF_ANGLE = 8.0
    SIDE_ANGLE = 45.0
    SIDE_HALF_ANGLE = 15.0
    # Side rebounds stay inside the side zones while retaining a catch margin.
    REGIONS = {LEFT: (0.32, -45.0), CENTER: (1 / 2, 0.0), RIGHT: (0.68, 45.0)}

    def __init__(
        self,
        width: float = 600.0,
        height: float = 800.0,
        ceiling_y: float = 82.0,
        paddle_y: float = 740.0,
        paddle_width: float = 84.0,
        paddle_speed: float = 350.0,
        ball_radius: float = 6.0,
        ball_speed: float = 400.0,
        fps: int = 30,
        paddle_step_distance: float = 28.0,
        max_flight_seconds: float = 30.0,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.ceiling_y = ceiling_y
        self.paddle_y = paddle_y
        self.paddle_width = paddle_width
        self.paddle_speed = paddle_speed
        self.ball_radius = ball_radius
        self.ball_speed = ball_speed
        self.fps = fps
        self.paddle_step_distance = paddle_step_distance
        self.max_flight_seconds = max_flight_seconds
        self._initial_score = 0
        self._initial_level_time = 0.0
        self.level_number = 1
        self._time_limit = math.inf
        self._transition_cache: dict[tuple[ArkanoidState, str], ArkanoidState] = {}
        self._rate_cache: dict[ArkanoidState, float] = {}

    @classmethod
    def from_observation(cls, observation: dict[str, Any]) -> tuple[ArkanoidSearchDomain, ArkanoidState]:
        """Build a domain and starting node from a server state dictionary."""
        domain = cls(
            width=float(observation.get("width", 600)),
            height=float(observation.get("height", 800)),
            ceiling_y=float(observation.get("ceiling_y", 82)),
            paddle_y=float(observation["paddle_y"]),
            paddle_width=float(observation["paddle_width"]),
            ball_radius=float(observation.get("ball_radius", 6)),
            ball_speed=float(observation.get("ball_speed", 400)),
        )
        domain._initial_score = int(observation.get("level_score", observation.get("score", 0)))
        domain.level_number = int(observation.get("stage", observation.get("level", 1)))
        domain._initial_level_time = float(observation.get("level_time_elapsed", 0.0))
        limit = float(observation.get("level_time_limit", 0.0))
        domain._time_limit = max(0.0, limit - domain._initial_level_time) if limit > 0 else math.inf
        bricks = tuple(
            BrickState(
                x=float(b["x"]), y=float(b["y"]),
                width=float(b.get("width", 50)), height=float(b.get("height", 20)),
                health=int(b.get("hits_remaining", b.get("health", 1))),
                # Metal remains a collision obstacle but never a reward target.
                points=0 if bool(b.get("indestructible", False)) else int(b.get("points", 0)),
                indestructible=bool(b.get("indestructible", False)),
            )
            for b in observation.get("bricks", [])
        )
        state = ArkanoidState(
            ball_x=float(observation["ball_x"]), ball_y=float(observation["ball_y"]),
            ball_vx=float(observation["ball_vx"]), ball_vy=float(observation["ball_vy"]),
            paddle_x=float(observation["paddle_x"]), bricks=bricks,
            score=0,
            elapsed=0.0,
            lives=int(observation.get("lives", 3)),
        )
        return domain, state

    def actions(self, state: ArkanoidState) -> list[str]:
        """Return reachable rebound actions that preserve the current life."""
        if self.satisfies(state, "clear") or state.lives <= 0 or state.elapsed >= self._time_limit:
            return []
        allowed: list[str] = []
        for action in self.ACTIONS:
            if self.safe_paddle_target(state, action) is None:
                continue
            next_state = self.result(state, action)
            if next_state.lives < state.lives:
                continue
            allowed.append(action)
        return allowed
    def predict_landing(self, state: ArkanoidState) -> tuple[float, float]:
        """Predict the ball's next paddle impact using wall-reflected motion."""

        radius = self.ball_radius
        plane = self.paddle_y - radius
        if state.ball_vy > 0:
            seconds = max(0.0, (plane - state.ball_y) / state.ball_vy)
        else:
            ceiling = self.ceiling_y + radius
            speed_y = max(abs(state.ball_vy), 1e-6)
            seconds = max(0.0, (state.ball_y - ceiling) / speed_y)
            seconds += max(0.0, (plane - ceiling) / speed_y)
        span = self.width - 2 * radius
        unfolded = state.ball_x - radius + state.ball_vx * seconds
        folded = unfolded % (2 * span)
        landing_x = radius + (folded if folded <= span else 2 * span - folded)
        return landing_x, seconds

    def safe_paddle_target(self, state: ArkanoidState, action: str) -> float | None:
        """Return a reachable paddle position with a robust catch margin."""
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown rebound action: {action}")
        landing_x, seconds = self.predict_landing(state)
        fraction, _ = self.REGIONS[action]
        target = min(self.width - self.paddle_width,
                     max(0.0, landing_x - fraction * self.paddle_width))
        steps = max(0, math.ceil(seconds * self.fps) - 2)
        max_travel = steps * self.paddle_step_distance
        if not max(0.0, state.paddle_x - max_travel) <= target <= min(
            self.width - self.paddle_width, state.paddle_x + max_travel
        ):
            return None
        relative = landing_x - target
        margin = max(
            self.ball_radius + self.paddle_step_distance * 0.55 + 2.0,
            self.paddle_width * 0.25,
        )
        if relative < margin or self.paddle_width - relative < margin:
            return None
        return target

    def predict_first_brick(self, state: ArkanoidState, action: str) -> dict[str, float | int | None]:
        """Estimate a likely first target over the paddle's random angle cone."""
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown rebound action: {action}")
        landing_x, _ = self.predict_landing(state)
        fraction, _ = self.REGIONS[action]
        target_x = min(self.width - self.paddle_width,
                       max(0.0, landing_x - fraction * self.paddle_width))
        if action == self.CENTER:
            mode, half_width = 0.0, self.CENTER_HALF_ANGLE
        else:
            mode = -self.SIDE_ANGLE if action == self.LEFT else self.SIDE_ANGLE
            half_width = self.SIDE_HALF_ANGLE

        samples = 13 if state.remaining_bricks <= 10 else 7
        outcomes: list[tuple[int, float]] = []
        for sample in range(samples):
            q = (sample + 0.5) / samples
            angle = self._triangular_quantile(q, mode - half_width, mode + half_width, mode)
            index, seconds = self._trace_first_brick(state, landing_x, angle)
            if index is not None:
                outcomes.append((index, seconds))

        if not outcomes:
            return {"target_paddle_x": target_x, "brick_index": None, "points": 0,
                    "hits_remaining": 0, "seconds": 0.0, "hit_probability": 0.0,
                    "expected_reward": 0.0}
        frequencies: dict[int, int] = {}
        for index, _ in outcomes:
            frequencies[index] = frequencies.get(index, 0) + 1
        likely_index = max(frequencies, key=frequencies.get)
        brick = state.bricks[likely_index]
        matching_times = [seconds for index, seconds in outcomes if index == likely_index]
        probability = len(matching_times) / samples
        reward = sum(
            # Estimate the time cost of clearing a multi-hit brick as one
            # flight per remaining hit. This makes the policy compare actual
            # points-per-clear-time, instead of overvaluing durable bricks.
            state.bricks[index].points
            / max(0.05, seconds * max(1, state.bricks[index].health))
            for index, seconds in outcomes
        ) / samples
        return {
            "target_paddle_x": target_x,
            "brick_index": likely_index,
            "points": brick.points,
            "hits_remaining": brick.health,
            "seconds": sum(matching_times) / len(matching_times),
            "hit_probability": probability,
            "expected_reward": reward,
        }

    @staticmethod
    def target_priority(hit: dict[str, float | int | None]) -> tuple[float, float, float]:
        """Compare targets by ease, hit probability, then conditional points/sec."""
        hits = max(1, int(hit.get("hits_remaining", 1) or 1))
        probability = float(hit.get("hit_probability", 0.0) or 0.0)
        points = float(hit.get("points", 0.0) or 0.0)
        seconds = max(0.05, float(hit.get("seconds", 0.0) or 0.0) * hits)
        return 1.0 / hits, probability, points / seconds

    @staticmethod
    def _triangular_quantile(probability: float, low: float, high: float, mode: float) -> float:
        split = (mode - low) / (high - low)
        if probability < split:
            return low + math.sqrt(probability * (high - low) * (mode - low))
        return high - math.sqrt((1 - probability) * (high - low) * (high - mode))

    def _brick_entry(
        self, x: float, y: float, dx: float, dy: float, brick: BrickState
    ) -> tuple[float, str] | None:
        """Return the swept collision fraction and the face hit, as the server does."""
        r = self.ball_radius
        min_x, max_x = brick.x - r, brick.x + brick.width + r
        min_y, max_y = brick.y - r, brick.y + brick.height + r

        if dx != 0.0:
            tx1, tx2 = (min_x - x) / dx, (max_x - x) / dx
            if tx1 > tx2:
                tx1, tx2 = tx2, tx1
        else:
            if x < min_x or x > max_x:
                return None
            tx1, tx2 = -math.inf, math.inf

        if dy != 0.0:
            ty1, ty2 = (min_y - y) / dy, (max_y - y) / dy
            if ty1 > ty2:
                ty1, ty2 = ty2, ty1
        else:
            if y < min_y or y > max_y:
                return None
            ty1, ty2 = -math.inf, math.inf

        entry, exit_time = max(tx1, ty1), min(tx2, ty2)
        if entry > exit_time or exit_time < 0.0 or entry > 1.0:
            return None
        return max(0.0, entry), ("y" if ty1 >= tx1 else "x")

    def _first_brick_collision(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        vx: float,
        vy: float,
        bricks: list[BrickState] | tuple[BrickState, ...],
        last_index: int | None,
        candidate_indices: set[int],
    ) -> tuple[int, float, str] | None:
        """Find the earliest swept hit, matching the official brick resolver."""
        r = self.ball_radius
        min_x, max_x = min(x, x + dx) - r, max(x, x + dx) + r
        min_y, max_y = min(y, y + dy) - r, max(y, y + dy) + r
        earliest = 1.0 + 1e-12
        best: tuple[int, float, str] | None = None

        for index in sorted(candidate_indices):
            brick = bricks[index]
            if brick.health <= 0 and not brick.indestructible:
                continue
            if (brick.x > max_x or brick.x + brick.width < min_x
                    or brick.y > max_y or brick.y + brick.height < min_y):
                continue
            if index == last_index and (
                brick.x - r <= x <= brick.x + brick.width + r
                and brick.y - r <= y <= brick.y + brick.height + r
            ):
                continue
            impact = self._brick_entry(x, y, dx, dy, brick)
            if impact is None:
                continue
            fraction, face = impact
            normal_x = (-1.0 if dx > 0 else 1.0) if face == "x" else 0.0
            normal_y = (-1.0 if dy > 0 else 1.0) if face == "y" else 0.0
            if vx * normal_x + vy * normal_y >= 0.0:
                continue
            if fraction < earliest:
                earliest = fraction
                best = (index, fraction, face)
        return best

    def _trace_first_brick(
        self, state: ArkanoidState, start_x: float, angle_deg: float
    ) -> tuple[int | None, float]:
        """Trace a ray, reflecting from walls, ceiling, and metal bricks."""
        radius = self.ball_radius
        x, y = start_x, self.paddle_y - radius
        angle = math.radians(angle_deg)
        vx, vy = self.ball_speed * math.sin(angle), -self.ball_speed * math.cos(angle)
        ceiling = self.ceiling_y + radius
        dt = 1.0 / (self.fps * 2)
        last_metal: int | None = None
        row_height = 24.0
        brick_rows: dict[int, list[int]] = {}
        for index, brick in enumerate(state.bricks):
            if brick.health <= 0 and not brick.indestructible:
                continue
            first_row = math.floor((brick.y - radius) / row_height)
            last_row = math.floor((brick.y + brick.height + radius) / row_height)
            for row in range(first_row, last_row + 1):
                brick_rows.setdefault(row, []).append(index)

        for tick in range(1200):
            dx, dy = vx * dt, vy * dt
            nx, ny = x + dx, y + dy
            if nx < radius:
                nx, vx = 2 * radius - nx, abs(vx)
            elif nx > self.width - radius:
                nx, vx = 2 * (self.width - radius) - nx, -abs(vx)
            if ny < ceiling:
                ny, vy = 2 * ceiling - ny, abs(vy)

            if state.remaining_bricks <= 10:
                first_row = math.floor((min(y, y + dy) - radius) / row_height)
                last_row = math.floor((max(y, y + dy) + radius) / row_height)
                candidate_indices = {
                    index for row in range(first_row, last_row + 1)
                    for index in brick_rows.get(row, ())
                }
                impact = self._first_brick_collision(
                    x, y, dx, dy, vx, vy, state.bricks, last_metal, candidate_indices
                )
            else:
                collision = next((
                    index for index in brick_rows.get(math.floor(ny / row_height), ())
                    if index != last_metal
                    and (state.bricks[index].health > 0 or state.bricks[index].indestructible)
                    and state.bricks[index].x - radius <= nx <= state.bricks[index].x + state.bricks[index].width + radius
                    and state.bricks[index].y - radius <= ny <= state.bricks[index].y + state.bricks[index].height + radius
                ), None)
                impact = None
                if collision is not None:
                    brick = state.bricks[collision]
                    face = "x" if x < brick.x - radius or x > brick.x + brick.width + radius else "y"
                    impact = (collision, 1.0, face)
            collision = impact[0] if impact is not None else None
            if impact is not None:
                collision, fraction, face = impact
                brick = state.bricks[collision]
                if not brick.indestructible:
                    return collision, (tick + fraction) * dt
                x += dx * fraction
                y += dy * fraction
                if face == "x":
                    vx = -vx
                else:
                    vy = -vy
                remaining = dt * (1.0 - fraction)
                nx, ny = x + vx * remaining, y + vy * remaining
                last_metal = collision
            elif last_metal is not None:
                metal = state.bricks[last_metal]
                if not (metal.x - radius <= nx <= metal.x + metal.width + radius
                        and metal.y - radius <= ny <= metal.y + metal.height + radius):
                    last_metal = None
            x, y = nx, ny
        return None, 10.0

    def choose_paddle_target(self, state: ArkanoidState) -> tuple[str, float, dict[str, float | int | None]]:
        """Choose a reachable interception zone using StudentAgent's ray scoring."""
        landing_x, seconds = self.predict_landing(state)
        paddle_max_x = self.width - self.paddle_width
        # Keep two game ticks in reserve for observation/action latency.
        available_steps = max(0, math.ceil(seconds * self.fps) - 2)
        max_travel = available_steps * self.paddle_step_distance
        reachable_low = max(0.0, state.paddle_x - max_travel)
        reachable_high = min(paddle_max_x, state.paddle_x + max_travel)

        if state.ball_vy <= 0:
            target = min(paddle_max_x, max(0.0, landing_x - self.paddle_width / 2))
            return self.CENTER, min(reachable_high, max(reachable_low, target)), {
                "target_paddle_x": target, "brick_index": None, "points": 0,
                "hits_remaining": 0, "seconds": 0.0,
            }

        candidates: list[tuple[str, float, dict[str, float | int | None], float, float, float, float, float, int]] = []
        for action in self.ACTIONS:
            fraction, _ = self.REGIONS[action]
            target = min(paddle_max_x, max(0.0, landing_x - fraction * self.paddle_width))
            if not reachable_low <= target <= reachable_high:
                continue
            relative = landing_x - target
            if not 0 <= relative <= self.paddle_width:
                continue
            # Reserve space for the discrete move dead band, ball radius, and
            # a small prediction error while still allowing safe side rebounds.
            safety_margin = max(
                self.ball_radius + self.paddle_step_distance * 0.55 + 2.0,
                self.paddle_width * 0.25,
            )
            if relative < safety_margin or self.paddle_width - relative < safety_margin:
                continue

            hit = self.predict_first_brick(state, action)
            ray_value = float(hit.get("expected_reward", 0.0) or 0.0)
            hit_probability = float(hit.get("hit_probability", 0.0) or 0.0)
            shot_rate = self.expected_shot_rate(state, action, landing_x, seconds)
            points = float(hit.get("points", 0) or 0)
            hits_remaining = max(1, int(hit.get("hits_remaining", 1) or 1))
            target_seconds = max(0.05, float(hit.get("seconds", 0.0) or 0.0) * hits_remaining)
            target_rate = points / target_seconds
            if shot_rate <= 0 and ray_value <= 0 and target_rate <= 0:
                continue
            catch_margin = min(relative, self.paddle_width - relative)
            travel = abs(target - state.paddle_x)
            candidates.append((action, target, hit, target_rate, hit_probability, shot_rate,
                               catch_margin, travel, hits_remaining))

        if candidates:
            best: tuple[tuple[float, float, float, float, float], str, float,
                        dict[str, float | int | None]] | None = None
            for action, target, hit, target_rate, probability, shot_rate, margin, travel, hits in candidates:
                # Rank by expected throughput; safety margins break close ties.
                shot_weight = {1: 0.50, 2: 0.35, 3: 0.80}.get(self.level_number, 0.50)
                expected_rate = shot_weight * shot_rate + (1.0 - shot_weight) * ray_value
                hit["target_points_per_second"] = target_rate
                hit["full_flight_points_per_second"] = shot_rate
                hit["expected_points_per_second"] = expected_rate
                hit["ease_score"] = 1.0 / max(1, hits)
                rank = (expected_rate, probability, margin, -travel, 1.0 / max(1, hits))
                if best is None or rank > best[0]:
                    best = (rank, action, target, hit)
            assert best is not None
            return best[1], best[2], best[3]

        # If no scored zone is reachable, move to the nearest safe catch point.
        catch_target = min(paddle_max_x, max(0.0, landing_x - self.paddle_width / 2))
        target = min(reachable_high, max(reachable_low, catch_target))
        return self.CENTER, target, {
            "target_paddle_x": target, "brick_index": None, "points": 0,
            "hits_remaining": 0, "seconds": 0.0,
        }

    def expected_shot_rate(
        self, state: ArkanoidState, action: str, landing_x: float, inbound_seconds: float
    ) -> float:
        """Estimate points/second over a sampled paddle-to-paddle flight."""
        fraction, mode = self.REGIONS[action]
        del fraction  # The landing coordinate is already supplied by the caller.
        half_width = self.CENTER_HALF_ANGLE if action == self.CENTER else self.SIDE_HALF_ANGLE
        samples = 7 if state.remaining_bricks <= 10 else 5
        row_height = 24.0
        brick_rows: dict[int, list[int]] = {}
        for index, brick in enumerate(state.bricks):
            if brick.health <= 0 and not brick.indestructible:
                continue
            first_row = math.floor((brick.y - self.ball_radius) / row_height)
            last_row = math.floor((brick.y + brick.height + self.ball_radius) / row_height)
            for row in range(first_row, last_row + 1):
                brick_rows.setdefault(row, []).append(index)

        total_points = 0
        total_seconds = inbound_seconds * samples
        sample_rates: list[float] = []
        for sample in range(samples):
            q = (sample + 0.5) / samples
            angle = self._triangular_quantile(q, mode - half_width, mode + half_width, mode)
            points, seconds = self._simulate_shot(state, landing_x, angle, brick_rows, row_height)
            total_points += points
            total_seconds += seconds
            sample_rates.append(points / max(inbound_seconds + seconds, 1e-6))
        mean_rate = total_points / max(total_seconds, 1e-6)
        sample_mean = sum(sample_rates) / max(1, len(sample_rates))
        variance = sum((rate - sample_mean) ** 2 for rate in sample_rates) / max(1, len(sample_rates))
        return mean_rate + 0.50 * math.sqrt(variance)

    def _simulate_shot(
        self,
        state: ArkanoidState,
        start_x: float,
        angle_deg: float,
        brick_rows: dict[int, list[int]],
        row_height: float,
    ) -> tuple[int, float]:
        """Simulate one sampled rebound until it returns to the paddle."""
        x, y = start_x, self.paddle_y - self.ball_radius
        angle = math.radians(angle_deg)
        vx = self.ball_speed * math.sin(angle)
        vy = -self.ball_speed * math.cos(angle)
        bricks = list(state.bricks)
        points = 0
        remaining_brick_count = state.remaining_bricks
        elapsed = 0.0
        last_hit: int | None = None
        dt = 1.0 / (self.fps * 2)
        ceiling = self.ceiling_y + self.ball_radius
        max_ticks = max(1, int(self.max_flight_seconds * self.fps * 2))

        for _ in range(max_ticks):
            elapsed += dt
            dx, dy = vx * dt, vy * dt
            nx, ny = x + dx, y + dy
            if nx < self.ball_radius:
                nx, vx = 2 * self.ball_radius - nx, abs(vx)
            elif nx > self.width - self.ball_radius:
                nx, vx = 2 * (self.width - self.ball_radius) - nx, -abs(vx)
            if ny < ceiling:
                ny, vy = 2 * ceiling - ny, abs(vy)

            if remaining_brick_count <= 10:
                first_row = math.floor((min(y, y + dy) - self.ball_radius) / row_height)
                last_row = math.floor((max(y, y + dy) + self.ball_radius) / row_height)
                candidate_indices = {
                    index for row in range(first_row, last_row + 1)
                    for index in brick_rows.get(row, ())
                }
                impact = self._first_brick_collision(
                    x, y, dx, dy, vx, vy, bricks, last_hit, candidate_indices
                )
            else:
                hit_index = next((
                    index for index in brick_rows.get(math.floor(ny / row_height), ())
                    if index != last_hit
                    and (bricks[index].health > 0 or bricks[index].indestructible)
                    and bricks[index].x - self.ball_radius <= nx <= bricks[index].x + bricks[index].width + self.ball_radius
                    and bricks[index].y - self.ball_radius <= ny <= bricks[index].y + bricks[index].height + self.ball_radius
                ), None)
                impact = None
                if hit_index is not None:
                    brick = bricks[hit_index]
                    face = "x" if x < brick.x - self.ball_radius or x > brick.x + brick.width + self.ball_radius else "y"
                    impact = (hit_index, 1.0, face)
            hit_index = impact[0] if impact is not None else None
            if impact is not None:
                hit_index, fraction, face = impact
                brick = bricks[hit_index]
                if not brick.indestructible:
                    health = brick.health - 1
                    bricks[hit_index] = BrickState(
                        brick.x, brick.y, brick.width, brick.height,
                        health, brick.points, brick.indestructible,
                    )
                    if health <= 0:
                        points += brick.points
                        remaining_brick_count -= 1
                x += dx * fraction
                y += dy * fraction
                if face == "x":
                    vx = -vx
                else:
                    vy = -vy
                remaining = dt * (1.0 - fraction)
                nx, ny = x + vx * remaining, y + vy * remaining
                last_hit = hit_index
            elif last_hit is not None:
                previous = bricks[last_hit]
                if not (
                    previous.x - self.ball_radius <= nx <= previous.x + previous.width + self.ball_radius
                    and previous.y - self.ball_radius <= ny <= previous.y + previous.height + self.ball_radius
                ):
                    last_hit = None

            if points >= sum(b.points for b in state.bricks if not b.indestructible and b.health > 0):
                return points, elapsed
            if vy > 0 and ny + self.ball_radius >= self.paddle_y:
                return points, elapsed
            x, y = nx, ny

        return points, elapsed

    def result(self, state: ArkanoidState, action: str) -> ArkanoidState:
        """Apply a rebound choice and simulate the next ball flight."""
        cache_key = (state, action)
        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown rebound action: {action}")

        # Aim for the selected third at the predicted paddle impact point.
        landing_x, _ = self.predict_landing(state)
        fraction, angle_deg = self.REGIONS[action]
        paddle_x = min(self.width - self.paddle_width,
                       max(0.0, landing_x - fraction * self.paddle_width))
        # Reaching the target takes time. If it cannot be reached before impact,
        # the flight simulator will record a miss rather than assuming a catch.
        # Match one server update per planning step. Paddle moves are discrete
        # commands of step_distance pixels, not a continuous 350 px/s motion.
        dt = 1.0 / (self.fps * 2)
        x, y = state.ball_x, state.ball_y
        vx, vy = state.ball_vx, state.ball_vy
        bricks = list(state.bricks)
        score = state.score
        elapsed = state.elapsed
        paddle_start = state.paddle_x
        paddle_direction = 0.0 if paddle_x == paddle_start else math.copysign(1.0, paddle_x - paddle_start)
        last_hit_index: int | None = None
        hit_indices: list[int] = []
        ticks = min(int(self.max_flight_seconds * self.fps * 2), 200000)

        for tick in range(ticks):
            elapsed += dt
            movement = self.paddle_step_distance if tick % 2 == 0 else 0.0
            remaining = abs(paddle_x - paddle_start)
            paddle_start += paddle_direction * min(movement, remaining)
            nx, ny = x + vx * dt, y + vy * dt
            if nx < self.ball_radius:
                nx, vx = 2 * self.ball_radius - nx, abs(vx)
            elif nx > self.width - self.ball_radius:
                nx, vx = 2 * (self.width - self.ball_radius) - nx, -abs(vx)
            if ny < self.ceiling_y + self.ball_radius:
                ny, vy = 2 * (self.ceiling_y + self.ball_radius) - ny, abs(vy)

            hit_index = next((i for i, b in enumerate(bricks) if (b.health > 0 or b.indestructible)
                              and i != last_hit_index
                              and b.x - self.ball_radius <= nx <= b.x + b.width + self.ball_radius
                              and b.y - self.ball_radius <= ny <= b.y + b.height + self.ball_radius), None)
            if hit_index is not None:
                hit_indices.append(hit_index)
                brick = bricks[hit_index]
                if not brick.indestructible:
                    health = brick.health - 1
                    bricks[hit_index] = BrickState(brick.x, brick.y, brick.width, brick.height,
                                                   health, brick.points, brick.indestructible)
                    if health <= 0:
                        score += brick.points
                if x < brick.x - self.ball_radius or x > brick.x + brick.width + self.ball_radius:
                    vx = -vx
                else:
                    vy = -vy
                last_hit_index = hit_index
            elif last_hit_index is not None:
                previous = bricks[last_hit_index]
                still_overlapping = (
                    previous.x - self.ball_radius <= nx <= previous.x + previous.width + self.ball_radius
                    and previous.y - self.ball_radius <= ny <= previous.y + previous.height + self.ball_radius
                )
                if not still_overlapping:
                    last_hit_index = None

            if vy > 0 and ny + self.ball_radius >= self.paddle_y:
                impact_x = nx
                reached = paddle_start <= impact_x + self.ball_radius and impact_x <= paddle_start + self.paddle_width + self.ball_radius
                if reached:
                    # Apply the selected deterministic rebound angle.
                    angle = math.radians(abs(angle_deg))
                    direction = -1 if angle_deg < 0 else (1 if angle_deg > 0 else 0)
                    vx, vy = direction * self.ball_speed * math.sin(angle), -self.ball_speed * math.cos(angle)
                    next_state = ArkanoidState(nx, self.paddle_y - self.ball_radius,
                                               vx, vy, paddle_start, tuple(bricks), score, elapsed,
                                               state.lives, tuple(hit_indices))
                else:
                    next_state = ArkanoidState(nx, ny, vx, vy, paddle_start, tuple(bricks), score, elapsed,
                                               max(0, state.lives - 1), tuple(hit_indices))
                self._transition_cache[cache_key] = next_state
                return next_state
            x, y = nx, ny

        # Flight cutoff prevents an unbounded transition in a search tree.
        next_state = ArkanoidState(x, y, vx, vy, paddle_start, tuple(bricks), score, elapsed,
                                   max(0, state.lives - 1), tuple(hit_indices))
        self._transition_cache[cache_key] = next_state
        return next_state

    def cost(self, state: ArkanoidState, action: str) -> float:
        """Elapsed time for the transition, penalized when it loses a life."""
        next_state = self.result(state, action)
        return max(1e-6, next_state.elapsed - state.elapsed) + 1000.0 * (state.lives - next_state.lives)

    def heuristic(self, state: ArkanoidState, goal: Any = None) -> float:
        """Optimistic remaining time: each remaining brick needs at least one hit."""
        return state.remaining_bricks * 0.1

    def satisfies(self, state: ArkanoidState, goal: Any = None) -> bool:
        """The level objective is complete when no destructible bricks remain."""
        return state.remaining_bricks == 0

    def points_per_second(self, state: ArkanoidState) -> float:
        """Score a plan node by observed plus projected score over elapsed time."""
        cached = self._rate_cache.get(state)
        if cached is not None:
            return cached

        projected_points = 0
        projected_seconds = 0.0
        near_paddle = state.ball_y >= self.paddle_y - 2 * self.ball_radius
        if state.ball_vy < 0 and near_paddle:
            angle = math.degrees(math.atan2(state.ball_vx, -state.ball_vy))
            row_height = 24.0
            brick_rows: dict[int, list[int]] = {}
            for index, brick in enumerate(state.bricks):
                if brick.health <= 0 and not brick.indestructible:
                    continue
                first_row = math.floor((brick.y - self.ball_radius) / row_height)
                last_row = math.floor((brick.y + brick.height + self.ball_radius) / row_height)
                for row in range(first_row, last_row + 1):
                    brick_rows.setdefault(row, []).append(index)
            projected_points, projected_seconds = self._simulate_shot(
                state, state.ball_x, angle, brick_rows, row_height
            )

        rate = (state.score + projected_points) / max(state.elapsed + projected_seconds, 1e-6)
        self._rate_cache[state] = rate
        return rate






def search_plan(
    observation: dict[str, Any], strategy: str = "rate", horizon: int = 6
) -> tuple[ArkanoidSearchDomain, ArkanoidState, list[ArkanoidState] | None]:
    """Return the domain, initial node, and bounded search path for an observation."""
    domain, initial = ArkanoidSearchDomain.from_observation(observation)
    problem = SearchProblem(domain, initial, "clear")
    tree = SearchTree(problem, strategy=strategy, limite=horizon)
    return domain, initial, tree.search()


def search_decision(
    observation: dict[str, Any], horizon: int = 2, node_limit: int = 40
) -> tuple[ArkanoidSearchDomain, ArkanoidState, list[ArkanoidState] | None, list[str]]:
    """Return a bounded plan and its actions for a live receding-horizon agent."""
    domain, initial = ArkanoidSearchDomain.from_observation(observation)
    problem = SearchProblem(domain, initial, "clear")
    tree = SearchTree(problem, strategy="rate", limite=horizon, node_limit=node_limit)
    path = tree.search()
    action_path = tree.get_action_path(tree.solution) if tree.solution is not None else []
    return domain, initial, path, action_path


def search_observation(
    observation: dict[str, Any], strategy: str = "rate", horizon: int = 6
) -> list[ArkanoidState] | None:
    """Plan from an observation and return its sequence of predicted states.

    If clearing the level is beyond the horizon, rate search returns the best
    partial plan. A live agent can execute its first move and replan next tick.
    """
    return search_plan(observation, strategy, horizon)[2]
