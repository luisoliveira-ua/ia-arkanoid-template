"""Core game logic and physics engine for IA Arkanoid."""

import math
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np

from server.config import AppConfig, load_config, load_map_file
from server.highscores import HighScoreManager


class Brick:
    """Represents a single brick in the Arkanoid arena."""

    def __init__(
        self,
        index: int,
        x: float,
        y: float,
        width: float,
        height: float,
        brick_type: str = "red",
        health: int = 1,
        indestructible: bool = False,
        points: int = 50,
    ) -> None:
        self.index = index
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.brick_type = brick_type
        self.health = health
        self.max_health = health
        self.indestructible = indestructible
        self.points = points
        self.active = True
        self._dict_cache: dict[str, Any] | None = None

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def hits_remaining(self) -> int:
        """Hits required to destroy this brick (-1 if indestructible)."""
        return -1 if self.indestructible else max(0, self.health)

    @property
    def palette(self) -> str:
        """Nord color palette group for current state."""
        if self.indestructible:
            return "snow_storm"
        if self.health <= 1:
            return "frost"
        return "aurora"

    @property
    def dynamic_type(self) -> str:
        """Current dynamic brick type reflecting hits remaining."""
        if self.indestructible:
            return "metal"
        if self.health <= 1:
            return "frost"
        if self.health == 2:
            return "aurora_green"
        if self.health == 3:
            return "aurora_yellow"
        if self.health == 4:
            return "aurora_orange"
        return "aurora_red"

    @property
    def color(self) -> str:
        """Current Nord hex color reflecting hits remaining."""
        if self.indestructible:
            return "#D8DEE9"  # nord4 (Snow Storm)
        if self.health <= 1:
            return "#88C0D0"  # nord8 (Frost Ice Blue - 1 hit left)
        if self.health == 2:
            return "#A3BE8C"  # nord14 (Aurora Green - 2 hits)
        if self.health == 3:
            return "#EBCB8B"  # nord13 (Aurora Yellow - 3 hits)
        if self.health == 4:
            return "#D08770"  # nord12 (Aurora Orange - 4 hits)
        return "#BF616A"  # nord11 (Aurora Red - 5+ hits)

    def hit(self) -> bool:
        """Register a hit on this brick. Returns True if the brick was destroyed."""
        if self.indestructible:
            return False
        self._dict_cache = None
        self.health -= 1
        if self.health <= 0:
            self.active = False
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serializable brick snapshot, cached until the brick takes damage (treat as read-only)."""
        if self._dict_cache is None:
            self._dict_cache = self._build_dict()
        return self._dict_cache

    def _build_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "type": self.dynamic_type,
            "health": self.health,
            "hits_remaining": self.hits_remaining,
            "palette": self.palette,
            "color": self.color,
            "max_health": self.max_health,
            "indestructible": self.indestructible,
            "active": self.active,
            "points": self.points,
        }


class Arkanoid:
    """Arkanoid simulation environment with Earth gravity and 3-region paddle physics."""

    def __init__(
        self,
        config: AppConfig | None = None,
        high_score_manager: HighScoreManager | None = None,
    ) -> None:
        self.config = config or load_config()
        self.high_score_manager = (
            high_score_manager
            if high_score_manager is not None
            else HighScoreManager(filepath=self.config.game.highscores_file)
        )
        self.player_name = "PLAYER"
        self.agent_id: str | None = None
        self.score_listener: Callable[[dict[str, Any]], None] | None = None
        self._score_recorded = False

        self.width = self.config.screen.width
        self.height = self.config.screen.height
        self.ceiling_y = getattr(self.config.screen, "ceiling_y", 82.0)

        self.paddle_width = self.config.paddle.width
        self.paddle_height = self.config.paddle.height
        self.paddle_y = self.config.paddle.y
        self.paddle_x = (self.width - self.paddle_width) / 2.0
        self.step_distance = self.config.paddle.step_distance

        self.ball_radius = self.config.ball.radius
        self.ball_speed = self.config.ball.speed

        self.ball_x: float = 0.0
        self.ball_y: float = 0.0
        self.ball_vx: float = 0.0
        self.ball_vy: float = 0.0

        self.lives = self.config.game.lives
        self.score = 0
        self.elapsed_time: float = 0.0
        self.level_elapsed: float = 0.0
        self._level_start_score = 0  # score when the current level was loaded
        self.level_norms: list[float] = []  # points/sec of every level cleared so far (an uncleared level earns none)
        self.timed_out = False
        self.high_score = self.high_score_manager.get_highest_score()
        self.high_score_normalized = (
            self.high_score_manager.get_highest_normalized_score()
        )
        self.game_over = False
        self.game_won = False

        self.level_sequence = self.config.game.level_sequence
        self.level_idx = 0
        self.current_level_name = ""

        self.bricks: list[Brick] = []
        # Vectorized array columns: [left, top, right, bottom, active, index, indestructible, health]
        self.brick_array = np.empty((0, 8), dtype=np.float32)

        # Zone-bounded spatial grid for O(1) swept collisions
        self.grid_cell_w: float = 60.0
        self.grid_cell_h: float = 25.0
        self.grid_cols: int = 0
        self.grid_rows: int = 0
        self.grid_min_x: float = 0.0
        self.grid_max_x: float = 0.0
        self.grid_min_y: float = 0.0
        self.grid_max_y: float = 0.0
        self.grid: list[list[Brick]] = []
        self._query_id: int = 0
        self._last_query_id: list[int] = []

        self.reset_game()

    @property
    def level_score(self) -> int:
        """Points scored in the current level."""
        return self.score - self._level_start_score

    @property
    def normalized_score(self) -> float:
        """Per-level metric: points per second of the current level (its clock restarts at every level).

        It is 0 when the game ended on this level without clearing it: only cleared levels earn a normalized score.
        """
        if self.game_over and not self.game_won:
            return 0.0
        if self.level_elapsed > 0.0:
            return round(self.level_score / self.level_elapsed, 2)
        return 0.0

    @property
    def avg_norm_score(self) -> float:
        """Grading metric: mean over all the levels of the sequence of their normalized score.

        A level that is not cleared (the game ended on it) or never reached counts 0, so the value only grows
        when a level is cleared and does not depend on how long the levels before it took.
        """
        if not self.level_sequence:
            return 0.0
        return round(sum(self.level_norms) / len(self.level_sequence), 2)

    def reset_game(self) -> None:
        """Reset the full game session back to level 1."""
        self.lives = self.config.game.lives
        self.score = 0
        self.elapsed_time = 0.0
        self.level_elapsed = 0.0
        self._level_start_score = 0
        self.level_norms = []
        self.timed_out = False
        self.game_over = False
        self.game_won = False
        self._score_recorded = False
        self.high_score = self.high_score_manager.get_highest_score()
        self.high_score_normalized = (
            self.high_score_manager.get_highest_normalized_score()
        )
        self.level_idx = 0
        self.load_level(self.level_idx)
        self.reset_paddle_and_ball()

    def record_high_score(self, player_name: str | None = None) -> bool:
        """Record final score and normalized score to persistent high score manager."""
        if self._score_recorded:
            return False
        if player_name:
            self.player_name = player_name
        added = self.high_score_manager.add_score(
            self.score,
            player=self.player_name,
            agent_id=self.agent_id or "",
            time_elapsed=self.elapsed_time,
            avg_norm_score=self.avg_norm_score,
        )
        self.high_score = self.high_score_manager.get_highest_score()
        self.high_score_normalized = (
            self.high_score_manager.get_highest_normalized_score()
        )
        self._score_recorded = True
        if self.score_listener is not None:
            self.score_listener(
                {
                    "id": self.agent_id,
                    "name": self.player_name,
                    "score": self.score,
                    "avg_norm_score": self.avg_norm_score,
                    "levels_cleared": len(self.level_norms),
                    "levels_played": len(self.level_sequence),
                    "setting": self.levels_label(),
                    "time_elapsed": self.elapsed_time,
                    "level": self._level_number(),
                    "won": self.game_won,
                    "timed_out": self.timed_out,
                }
            )
        return added

    def levels_label(self) -> str:
        """Canonical label of the configured levels for the dashboard: the sorted set of map numbers, runs compressed.

        "1,2,3", "3,1,2" and the default sequence all give "1-3"; "1,3" stays "1,3"; map files without a number
        are listed by name. Order and repeats do not matter: only the levels played decide who is comparable.
        """
        nums: set[int] = set()
        for name in self.level_sequence:
            digits = "".join(ch for ch in name if ch.isdigit())
            if not digits:
                return ",".join(self.level_sequence)[:48]
            nums.add(int(digits))
        ordered = sorted(nums)
        parts: list[str] = []
        i = 0
        while i < len(ordered):
            j = i
            while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
                j += 1
            parts.append(str(ordered[i]) if j == i else f"{ordered[i]}-{ordered[j]}")
            i = j + 1
        return ",".join(parts)

    def _level_number(self) -> int | None:
        """Map number (`maps/levelN.json`) of the level the game ended on, None if the file name has no number."""
        idx = min(self.level_idx, len(self.level_sequence) - 1)
        digits = "".join(ch for ch in self.level_sequence[idx] if ch.isdigit())
        return int(digits) if digits else None

    def load_level(self, level_idx: int) -> None:
        """Load a level map from the configured sequence."""
        if level_idx < 0 or level_idx >= len(self.level_sequence):
            self.game_won = True
            return

        self.level_elapsed = 0.0
        self._level_start_score = self.score
        map_filename = self.level_sequence[level_idx]
        map_path = os.path.join(self.config.game.maps_dir, map_filename)
        if not os.path.exists(map_path):
            # Fallback to root or default level
            map_path = map_filename

        try:
            map_data = load_map_file(map_path)
        except Exception:
            # Fallback to default generated grid if file is unreadable
            map_data = {"name": f"Stage {level_idx + 1}", "bricks": []}

        self.current_level_name = str(map_data.get("name", f"Stage {level_idx + 1}"))
        self._init_bricks_from_data(map_data.get("bricks", []))

    def _init_bricks_from_data(self, brick_list: list[dict[str, Any]]) -> None:
        """Initialize Brick objects and NumPy collision array from parsed map data."""
        self.bricks = []
        data = []
        b_width = self.config.bricks.width
        b_height = self.config.bricks.height

        for idx, b_info in enumerate(brick_list):
            b_type = str(b_info.get("type", "red"))
            type_cfg = self.config.bricks.types.get(b_type)

            health = int(b_info.get("health", type_cfg.health if type_cfg else 1))
            indestructible = bool(
                b_info.get(
                    "indestructible", type_cfg.indestructible if type_cfg else False
                )
            )
            points = int(b_info.get("points", type_cfg.points if type_cfg else 50))

            x = float(b_info.get("x", 0.0))
            y = float(b_info.get("y", 0.0))
            width = float(b_info.get("width", b_width))
            height = float(b_info.get("height", b_height))

            brick = Brick(
                index=idx,
                x=x,
                y=y,
                width=width,
                height=height,
                brick_type=b_type,
                health=health,
                indestructible=indestructible,
                points=points,
            )
            self.bricks.append(brick)
            data.append(
                [
                    brick.left,
                    brick.top,
                    brick.right,
                    brick.bottom,
                    1.0,  # active
                    float(idx),
                    1.0 if indestructible else 0.0,
                    float(health),
                ]
            )

        if data:
            self.brick_array = np.array(data, dtype=np.float32)
        else:
            self.brick_array = np.empty((0, 8), dtype=np.float32)

        self.build_spatial_grid()

    def build_spatial_grid(self) -> None:
        """Construct a 2D spatial uniform grid spanning the active brick zone for O(1) swept collisions."""
        if not self.bricks:
            self.grid = []
            self.grid_cols = 0
            self.grid_rows = 0
            self._last_query_id = []
            return

        self.grid_min_x = float(min(b.left for b in self.bricks))
        self.grid_max_x = float(max(b.right for b in self.bricks))
        self.grid_min_y = float(min(b.top for b in self.bricks))
        self.grid_max_y = float(max(b.bottom for b in self.bricks))

        zone_w = max(self.grid_cell_w, self.grid_max_x - self.grid_min_x + 1.0)
        zone_h = max(self.grid_cell_h, self.grid_max_y - self.grid_min_y + 1.0)

        self.grid_cols = int(math.ceil(zone_w / self.grid_cell_w))
        self.grid_rows = int(math.ceil(zone_h / self.grid_cell_h))

        self.grid = [[] for _ in range(self.grid_cols * self.grid_rows)]
        for b in self.bricks:
            c0 = max(0, min(self.grid_cols - 1, int((b.left - self.grid_min_x) // self.grid_cell_w)))
            c1 = max(0, min(self.grid_cols - 1, int((b.right - self.grid_min_x) // self.grid_cell_w)))
            r0 = max(0, min(self.grid_rows - 1, int((b.top - self.grid_min_y) // self.grid_cell_h)))
            r1 = max(0, min(self.grid_rows - 1, int((b.bottom - self.grid_min_y) // self.grid_cell_h)))
            for r in range(r0, r1 + 1):
                off = r * self.grid_cols
                for c in range(c0, c1 + 1):
                    self.grid[off + c].append(b)

        self._last_query_id = [0] * len(self.bricks)

    def reset_paddle_and_ball(self) -> None:
        """Reset paddle to center and place ball ready to launch."""
        self.paddle_x = (self.width - self.paddle_width) / 2.0
        self.reset_ball()

    def reset_ball(self) -> None:
        """Position the ball directly above the paddle center and launch with slight random slant."""
        self.ball_x = self.paddle_x + self.paddle_width / 2.0
        self.ball_y = self.paddle_y - self.ball_radius - 2.0

        # Initial launch angle between -30 and +30 degrees
        angle = random.uniform(-math.pi / 6.0, math.pi / 6.0)
        self.ball_vx = self.ball_speed * math.sin(angle)
        self.ball_vy = -self.ball_speed * math.cos(angle)

    def move_paddle(self, direction: str) -> None:
        """Move paddle horizontally given direction string."""
        if self.game_over:
            return

        direction_upper = direction.upper()
        if direction_upper in ("WEST", "LEFT", "A"):
            self.paddle_x = max(0.0, self.paddle_x - self.step_distance)
        elif direction_upper in ("EAST", "RIGHT", "D"):
            self.paddle_x = min(
                self.width - self.paddle_width, self.paddle_x + self.step_distance
            )

    def _handle_bottom_miss(self) -> None:
        """Handle ball falling below arena floor."""
        self.lives -= 1
        if self.lives <= 0:
            self.game_over = True
            self.record_high_score()
        else:
            self.reset_paddle_and_ball()

    def _calculate_paddle_bounce_angle(self, relative_x: float) -> float:
        """
        Bounce angle (radians from vertical, negative = left) for the 3-region paddle:
        - Center third: straight up (90 deg from horizontal) with a cone of +-alpha.
        - Left third: 45 deg to the left with a cone of +-beta.
        - Right third: 45 deg to the right with a cone of +-beta.
        alpha < beta. Within each cone the angle is triangular-distributed around the nominal direction.
        """
        w = self.paddle_width
        left_threshold = w * self.config.paddle.left_ratio
        right_threshold = w * (1.0 - self.config.paddle.right_ratio)
        cfg = self.config.paddle.angles

        if relative_x < left_threshold:
            centre, half = -cfg.side_deg, cfg.beta_deg
        elif relative_x > right_threshold:
            centre, half = cfg.side_deg, cfg.beta_deg
        else:
            centre, half = 0.0, cfg.alpha_deg
        return math.radians(random.triangular(centre - half, centre + half, centre))

    def _handle_paddle_bounce(self, hit_x: float) -> None:
        """Resolve ball bounce off the paddle using the 3-region angle distribution."""
        self.ball_x = float(
            max(self.ball_radius, min(self.width - self.ball_radius, hit_x))
        )
        self.ball_y = float(self.paddle_y - self.ball_radius)
        relative_x = float(self.ball_x - self.paddle_x)
        relative_x = float(max(0.0, min(float(self.paddle_width), relative_x)))
        bounce_angle = self._calculate_paddle_bounce_angle(relative_x)
        self.ball_vx = float(self.ball_speed * math.sin(bounce_angle))
        self.ball_vy = float(-self.ball_speed * math.cos(bounce_angle))

    def _find_earliest_brick_collision(
        self, bx: float, by: float, dx: float, dy: float, max_t: float
    ) -> tuple[float, Brick | None, float, float]:
        """Query the zone-bounded spatial grid and find earliest swept impact with a brick."""
        if not self.bricks or not self.grid:
            return max_t, None, 0.0, 0.0

        r = self.ball_radius
        x0, x1 = (bx, bx + dx) if dx >= 0.0 else (bx + dx, bx)
        y0, y1 = (by, by + dy) if dy >= 0.0 else (by + dy, by)

        # Early rejection if trajectory bounding box does not intersect brick zone
        if (
            y1 + r < self.grid_min_y
            or y0 - r > self.grid_max_y
            or x1 + r < self.grid_min_x
            or x0 - r > self.grid_max_x
        ):
            return max_t, None, 0.0, 0.0

        c_start = max(0, int((x0 - r - self.grid_min_x) // self.grid_cell_w))
        c_end = min(
            self.grid_cols - 1, int((x1 + r - self.grid_min_x) // self.grid_cell_w)
        )
        r_start = max(0, int((y0 - r - self.grid_min_y) // self.grid_cell_h))
        r_end = min(
            self.grid_rows - 1, int((y1 + r - self.grid_min_y) // self.grid_cell_h)
        )

        self._query_id += 1
        q_id = self._query_id

        earliest_t = max_t
        hit_brick: Brick | None = None
        hit_nx, hit_ny = 0.0, 0.0

        for r_idx in range(r_start, r_end + 1):
            row_offset = r_idx * self.grid_cols
            for c_idx in range(c_start, c_end + 1):
                for b in self.grid[row_offset + c_idx]:
                    if self._last_query_id[b.index] == q_id:
                        continue
                    self._last_query_id[b.index] = q_id
                    if not b.active:
                        continue

                    min_bx = b.left - r
                    max_bx = b.right + r
                    min_by = b.top - r
                    max_by = b.bottom + r

                    # Slab test along X
                    if dx != 0.0:
                        inv_dx = 1.0 / dx
                        t1x = (min_bx - bx) * inv_dx
                        t2x = (max_bx - bx) * inv_dx
                        if t1x > t2x:
                            t1x, t2x = t2x, t1x
                            nx = 1.0
                        else:
                            nx = -1.0
                    else:
                        if bx < min_bx or bx > max_bx:
                            continue
                        t1x, t2x = -1e9, 1e9
                        nx = 0.0

                    # Slab test along Y
                    if dy != 0.0:
                        inv_dy = 1.0 / dy
                        t1y = (min_by - by) * inv_dy
                        t2y = (max_by - by) * inv_dy
                        if t1y > t2y:
                            t1y, t2y = t2y, t1y
                            ny = 1.0
                        else:
                            ny = -1.0
                    else:
                        if by < min_by or by > max_by:
                            continue
                        t1y, t2y = -1e9, 1e9
                        ny = 0.0

                    t_enter = max(t1x, t1y)
                    t_exit = min(t2x, t2y)

                    if t_enter <= t_exit and t_exit >= 0.0:
                        cand_nx = nx if t1x > t1y else 0.0
                        cand_ny = ny if t1y >= t1x else 0.0

                        # Ensure ball trajectory is heading into surface
                        dot = self.ball_vx * cand_nx + self.ball_vy * cand_ny
                        if dot < 0.0:
                            cand_t = max(0.0, t_enter)
                            if cand_t < earliest_t:
                                earliest_t = cand_t
                                hit_brick = b
                                hit_nx = cand_nx
                                hit_ny = cand_ny

        return earliest_t, hit_brick, hit_nx, hit_ny

    def _handle_brick_hit(self, brick: Brick, hit_nx: float, hit_ny: float) -> None:
        """Resolve specular velocity reflection, positional nudge, damage, and scoring for a brick hit."""
        dot = self.ball_vx * hit_nx + self.ball_vy * hit_ny
        if dot < 0.0:
            self.ball_vx -= 2.0 * dot * hit_nx
            self.ball_vy -= 2.0 * dot * hit_ny
            spd = math.hypot(self.ball_vx, self.ball_vy)
            if spd > 0.0:
                self.ball_vx = float((self.ball_vx / spd) * self.ball_speed)
                self.ball_vy = float((self.ball_vy / spd) * self.ball_speed)

        r = self.ball_radius
        if abs(hit_nx) > 0.5:
            self.ball_x = float(
                brick.left - r - 1e-2 if hit_nx < 0.0 else brick.right + r + 1e-2
            )
        elif abs(hit_ny) > 0.5:
            self.ball_y = float(
                brick.top - r - 1e-2 if hit_ny < 0.0 else brick.bottom + r + 1e-2
            )

        self.ball_x = float(max(r, min(self.width - r, self.ball_x)))
        self.ball_y = float(max(self.ceiling_y + r, self.ball_y))

        # Points are awarded only when a brick breaks; damaging or bouncing off bricks scores nothing.
        if brick.hit():
            self.brick_array[brick.index, 4] = 0.0
            self.score += brick.points
        elif not brick.indestructible:
            self.brick_array[brick.index, 7] = float(brick.health)

        if self.score > self.high_score:
            self.high_score = self.score

        self._check_level_progression()

    def _check_level_progression(self) -> None:
        """Transition to next level when all destructible bricks are eliminated."""
        if len(self.brick_array) == 0:
            return

        destructible_remains = np.any(
            (self.brick_array[:, 4] == 1.0) & (self.brick_array[:, 6] == 0.0)
        )

        if not destructible_remains:
            if self.score > self.high_score:
                self.high_score = self.score

            if self.level_elapsed > 0.0:  # close the level: its own points / its own time
                self.level_norms.append(self.level_score / self.level_elapsed)
            self.level_idx += 1
            if self.level_idx < len(self.level_sequence):
                self.load_level(self.level_idx)
                self.reset_paddle_and_ball()
            else:
                self.game_won = True
                self.game_over = True
                self.record_high_score()

    def update(self, dt: float) -> None:
        """
        Advance simulation by time delta dt (seconds) using continuous collision
        detection (Swept CCD) with zone-bounded spatial grid partitioning.
        """
        if self.game_over or self.game_won:
            return

        limit = self.config.game.level_time_limit
        if limit > 0.0 and self.level_elapsed >= limit:
            # The level was not cleared in time: the game ends, it never advances to the next level.
            self.timed_out = True
            self.game_over = True
            self.record_high_score()
            return

        self.elapsed_time += dt
        self.level_elapsed += dt
        if self.score > self.high_score:
            self.high_score = self.score
        if self.avg_norm_score > self.high_score_normalized:
            self.high_score_normalized = self.avg_norm_score

        if len(self._last_query_id) != len(self.bricks):
            self.build_spatial_grid()

        remaining_time = min(dt, 0.1)
        max_bounces = 5

        while remaining_time > 1e-6 and max_bounces > 0:
            max_bounces -= 1
            consumed_time = self._step_simulation(remaining_time)
            remaining_time -= consumed_time
            if self.game_over or self.game_won:
                break

    def _step_simulation(self, dt: float) -> float:
        """Execute a single continuous collision step (Swept CCD) across walls, paddle, floor, and bricks."""
        r = self.ball_radius
        bx = self.ball_x
        by = self.ball_y
        vx = self.ball_vx
        vy = self.ball_vy

        dx = vx * dt
        dy = vy * dt

        earliest_t = 1.0
        hit_kind: str | None = None

        # 1. Floor miss
        if dy > 0.0 and by + dy + r >= self.height:
            t_floor = (self.height - r - by) / dy
            if 0.0 <= t_floor < earliest_t:
                earliest_t = max(0.0, t_floor)
                hit_kind = "floor"

        # 2. Walls and Ceiling
        if dx < 0.0 and bx + dx < r:
            t = (r - bx) / dx
            if 0.0 <= t < earliest_t:
                earliest_t = max(0.0, t)
                hit_kind = "wall_left"
        elif dx > 0.0 and bx + dx > self.width - r:
            t = (self.width - r - bx) / dx
            if 0.0 <= t < earliest_t:
                earliest_t = max(0.0, t)
                hit_kind = "wall_right"

        if dy < 0.0 and by + dy < self.ceiling_y + r:
            t = (self.ceiling_y + r - by) / dy
            if 0.0 <= t < earliest_t:
                earliest_t = max(0.0, t)
                hit_kind = "ceiling"

        # 3. Paddle collision (moving downwards)
        if vy > 0.0:
            p_top = self.paddle_y
            t_pad = (p_top - r - by) / dy
            if 0.0 <= t_pad < earliest_t:
                x_at_pad = bx + t_pad * dx
                p_left = self.paddle_x
                p_right = self.paddle_x + self.paddle_width
                if p_left - r <= x_at_pad <= p_right + r:
                    earliest_t = max(0.0, t_pad)
                    hit_kind = "paddle"

        # 4. Brick collisions via Zone-Bounded Spatial Grid
        brick_t, hit_brick, hit_nx, hit_ny = self._find_earliest_brick_collision(
            bx, by, dx, dy, earliest_t
        )
        if hit_brick is not None and brick_t < earliest_t:
            earliest_t = brick_t
            hit_kind = "brick"

        # Resolve earliest collision
        if hit_kind is None:
            self.ball_x = float(bx + dx)
            self.ball_y = float(by + dy)
            return dt

        if hit_kind == "floor":
            self.ball_x = float(bx + dx * earliest_t)
            self.ball_y = float(self.height)
            self._handle_bottom_miss()
            return dt

        if hit_kind == "paddle":
            self._handle_paddle_bounce(bx + dx * earliest_t)
            return max(dt * earliest_t, 1e-6)

        if hit_kind in ("wall_left", "wall_right", "ceiling"):
            self.ball_x = float(bx + dx * earliest_t)
            self.ball_y = float(by + dy * earliest_t)
            if hit_kind == "wall_left":
                self.ball_x = r
                self.ball_vx = abs(self.ball_vx)
            elif hit_kind == "wall_right":
                self.ball_x = self.width - r
                self.ball_vx = -abs(self.ball_vx)
            elif hit_kind == "ceiling":
                self.ball_y = self.ceiling_y + r
                self.ball_vy = abs(self.ball_vy)
            return max(dt * earliest_t, 1e-6)

        if hit_kind == "brick" and hit_brick is not None:
            self.ball_x = float(bx + dx * earliest_t)
            self.ball_y = float(by + dy * earliest_t)
            self._handle_brick_hit(hit_brick, hit_nx, hit_ny)
            return max(dt * earliest_t, 1e-6)

        return dt

    def get_state(self) -> dict[str, Any]:
        """Compile and serialize the complete game state dictionary."""
        valid_actions: list[dict[str, Any]] = []
        if not self.game_over and not self.game_won:
            if self.paddle_x > 0.0:
                valid_actions.append({"action": "move", "direction": "WEST"})
            if self.paddle_x < self.width - self.paddle_width:
                valid_actions.append({"action": "move", "direction": "EAST"})

        return {
            "width": self.width,
            "height": self.height,
            "paddle_x": self.paddle_x,
            "paddle_y": self.paddle_y,
            "paddle_width": self.paddle_width,
            "paddle_height": self.paddle_height,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_vx": self.ball_vx,
            "ball_vy": self.ball_vy,
            "ball_radius": self.ball_radius,
            "ball_speed": self.ball_speed,
            "gravity": 0.0,
            "lives": self.lives,
            "score": self.score,
            "normalized_score": self.normalized_score,
            "avg_norm_score": self.avg_norm_score,
            "timer": round(self.elapsed_time, 2),
            "time_elapsed": round(self.elapsed_time, 2),
            "high_score": self.high_score,
            "high_score_normalized": self.high_score_normalized,
            "high_scores": self.high_score_manager.get_top_scores(),
            "high_scores_normalized": self.high_score_manager.get_top_normalized_scores(),
            "game_over": self.game_over,
            "game_won": self.game_won,
            "timed_out": self.timed_out,
            "level_time_limit": self.config.game.level_time_limit,
            "level_time_elapsed": round(self.level_elapsed, 2),
            "stage": self.level_idx + 1,
            "stage_name": self.current_level_name,
            "max_stages": len(self.level_sequence),
            "level": self.level_idx + 1,
            "level_name": self.current_level_name,
            "max_levels": len(self.level_sequence),
            "ceiling_y": self.ceiling_y,
            "bricks": [b.to_dict() for b in self.bricks if b.active],
            "actions": valid_actions,
            "valid_actions": valid_actions,
        }
