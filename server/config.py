"""Configuration data classes and loader for Arkanoid."""

import json
import os
from dataclasses import dataclass, field
from typing import Any

from consts import (
    BALL_INITIAL_SPEED,
    BALL_RADIUS,
    BALL_SPEED,
    BRICK_HEIGHT,
    BRICK_TYPES,
    BRICK_WIDTH,
    CEILING_Y,
    GAME_FPS,
    GAME_HIGHSCORES_FILE,
    GAME_LEVEL_SEQUENCE,
    GAME_LEVEL_TIME_LIMIT,
    GAME_LIVES,
    GAME_MAPS_DIR,
    GRID_GAP_X,
    GRID_GAP_Y,
    GRID_MARGIN_X,
    GRID_MARGIN_Y,
    PADDLE_ALPHA_DEG,
    PADDLE_BETA_DEG,
    PADDLE_CENTER_RATIO,
    PADDLE_HEIGHT,
    PADDLE_LEFT_RATIO,
    PADDLE_RIGHT_RATIO,
    PADDLE_SIDE_DEG,
    PADDLE_SPEED,
    PADDLE_STEP_DISTANCE,
    PADDLE_WIDTH,
    PADDLE_Y,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)


@dataclass
class ScreenConfig:
    width: int = SCREEN_WIDTH
    height: int = SCREEN_HEIGHT
    ceiling_y: float = CEILING_Y


@dataclass
class PaddleAnglesConfig:
    """Bounce angles in degrees, measured from the vertical (0 = straight up)."""

    # half-width of the center cone (around 0 = vertical, i.e. 90 deg from horizontal)
    alpha_deg: float = PADDLE_ALPHA_DEG
    side_deg: float = PADDLE_SIDE_DEG  # nominal left/right deflection
    beta_deg: float = PADDLE_BETA_DEG  # half-width of the side cones; must be > alpha_deg


@dataclass
class PaddleConfig:
    width: float = PADDLE_WIDTH
    height: float = PADDLE_HEIGHT
    y: float = PADDLE_Y
    speed: float = PADDLE_SPEED
    step_distance: float = PADDLE_STEP_DISTANCE
    left_ratio: float = PADDLE_LEFT_RATIO
    center_ratio: float = PADDLE_CENTER_RATIO
    right_ratio: float = PADDLE_RIGHT_RATIO
    angles: PaddleAnglesConfig = field(default_factory=PaddleAnglesConfig)


@dataclass
class BallConfig:
    radius: float = BALL_RADIUS
    speed: float = BALL_SPEED
    initial_speed: float = BALL_INITIAL_SPEED


@dataclass
class BrickTypeConfig:
    health: int = 1
    indestructible: bool = False
    points: int = 50
    color: str = "#ffffff"


def _default_brick_types() -> dict[str, BrickTypeConfig]:
    return {
        name: BrickTypeConfig(
            health=int(info["health"]),
            indestructible=bool(info["indestructible"]),
            points=int(info["points"]),
            color=str(info["color"]),
        )
        for name, info in BRICK_TYPES.items()
    }


@dataclass
class BricksConfig:
    width: float = BRICK_WIDTH
    height: float = BRICK_HEIGHT
    types: dict[str, BrickTypeConfig] = field(default_factory=_default_brick_types)


@dataclass
class GameRulesConfig:
    lives: int = GAME_LIVES
    fps: int = GAME_FPS
    maps_dir: str = GAME_MAPS_DIR
    highscores_file: str = GAME_HIGHSCORES_FILE
    level_sequence: list[str] = field(
        default_factory=lambda: list(GAME_LEVEL_SEQUENCE)
    )
    # seconds of game time per level, 0 disables; game over when exceeded
    level_time_limit: float = GAME_LEVEL_TIME_LIMIT


@dataclass
class AppConfig:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    paddle: PaddleConfig = field(default_factory=PaddleConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    bricks: BricksConfig = field(default_factory=BricksConfig)
    game: GameRulesConfig = field(default_factory=GameRulesConfig)


def load_config(config_path: str = "config.json") -> AppConfig:
    """Load configuration from JSON file or return default configuration."""
    if not os.path.exists(config_path):
        return AppConfig()

    with open(config_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    screen_data = data.get("screen", {})
    screen = ScreenConfig(
        width=int(screen_data.get("width", 600)),
        height=int(screen_data.get("height", 800)),
        ceiling_y=float(screen_data.get("ceiling_y", 82.0)),
    )

    p_data = data.get("paddle", {})
    a_data = p_data.get("angles", {})
    angles = PaddleAnglesConfig(
        alpha_deg=float(a_data.get("alpha_deg", 8.0)),
        side_deg=float(a_data.get("side_deg", 45.0)),
        beta_deg=float(a_data.get("beta_deg", 15.0)),
    )
    if not 0.0 <= angles.alpha_deg < angles.beta_deg or angles.side_deg + angles.beta_deg >= 90.0:
        raise ValueError("paddle angles must satisfy 0 <= alpha < beta and side + beta < 90")
    paddle = PaddleConfig(
        width=float(p_data.get("width", 84.0)),
        height=float(p_data.get("height", 14.0)),
        y=float(p_data.get("y", 740.0)),
        speed=float(p_data.get("speed", 350.0)),
        step_distance=float(p_data.get("step_distance", 28.0)),
        left_ratio=float(p_data.get("left_ratio", 0.3333)),
        center_ratio=float(p_data.get("center_ratio", 0.3334)),
        right_ratio=float(p_data.get("right_ratio", 0.3333)),
        angles=angles,
    )

    b_data = data.get("ball", {})
    ball_speed = float(b_data.get("speed", b_data.get("initial_speed", 400.0)))
    ball = BallConfig(
        radius=float(b_data.get("radius", 6.0)),
        speed=ball_speed,
        initial_speed=ball_speed,
    )

    br_data = data.get("bricks", {})
    types_data = br_data.get("types", {})
    brick_types: dict[str, BrickTypeConfig] = {}
    for name, t_info in types_data.items():
        brick_types[name] = BrickTypeConfig(
            health=int(t_info.get("health", 1)),
            indestructible=bool(t_info.get("indestructible", False)),
            points=int(t_info.get("points", 50)),
            color=str(t_info.get("color", "#ffffff")),
        )
    bricks = BricksConfig(
        width=float(br_data.get("width", 50.0)),
        height=float(br_data.get("height", 20.0)),
        types=brick_types,
    )

    g_data = data.get("game", {})
    game = GameRulesConfig(
        lives=int(g_data.get("lives", 3)),
        fps=int(g_data.get("fps", 30)),
        maps_dir=str(g_data.get("maps_dir", "maps")),
        highscores_file=str(g_data.get("highscores_file", "highscores.csv")),
        level_sequence=list(
            g_data.get("level_sequence", ["level1.json", "level2.json", "level3.json"])
        ),
        level_time_limit=float(g_data.get("level_time_limit", GAME_LEVEL_TIME_LIMIT)),
    )

    return AppConfig(screen=screen, paddle=paddle, ball=ball, bricks=bricks, game=game)


def load_map_file(map_path: str) -> dict[str, Any]:
    """Read and validate a map JSON file, normalizing grid and brick formats."""
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map file not found: {map_path}")
    with open(map_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    if "bricks" not in data or not isinstance(data["bricks"], list):
        raise ValueError(f"Invalid map format in {map_path}: missing 'bricks' list")

    grid = data.get("grid", {})
    margin_x = float(grid.get("margin_x", GRID_MARGIN_X))
    margin_y = float(grid.get("margin_y", GRID_MARGIN_Y))
    gap_x = float(grid.get("gap_x", GRID_GAP_X))
    gap_y = float(grid.get("gap_y", GRID_GAP_Y))
    b_w = float(data.get("brick_width", BRICK_WIDTH))
    b_h = float(data.get("brick_height", BRICK_HEIGHT))

    for b in data["bricks"]:
        if "row" in b and "col" in b and "x" not in b and "y" not in b:
            b["x"] = margin_x + float(b["col"]) * (b_w + gap_x)
            b["y"] = margin_y + float(b["row"]) * (b_h + gap_y)

        b_type = b.get("type")
        health = b.get("health")
        if b_type is not None and b_type in BRICK_TYPES:
            cfg_type = BRICK_TYPES[b_type]
            if health is None:
                b["health"] = cfg_type["health"]
            if "indestructible" not in b:
                b["indestructible"] = cfg_type["indestructible"]
            if "points" not in b:
                b["points"] = cfg_type["points"]
        elif health is not None:
            if health == -1:
                b.setdefault("type", "metal")
                b.setdefault("indestructible", True)
                b.setdefault("points", 0)
            elif health == 1:
                b.setdefault("type", "red")
                b.setdefault("indestructible", False)
                b.setdefault("points", 50)
            elif health == 2:
                b.setdefault("type", "blue")
                b.setdefault("indestructible", False)
                b.setdefault("points", 100)
            elif health == 3:
                b.setdefault("type", "green")
                b.setdefault("indestructible", False)
                b.setdefault("points", 150)
            elif health >= 4:
                b.setdefault("type", "yellow")
                b.setdefault("indestructible", False)
                b.setdefault("points", 200)

        if b.get("indestructible", False):
            b["points"] = 0

    return data


def parse_levels(spec: str | int | list[Any], maps_dir: str = "maps") -> list[str]:
    """
    Resolve a level selection into map filenames.

    Accepts an int (3), a list ([1, 2]) or a string: "1", "1,2,3", "1-3", "1,3-5".
    Numeric N maps to `levelN.json`; non-numeric tokens are treated as map file names.
    """
    if isinstance(spec, int):
        tokens = [str(spec)]
    elif isinstance(spec, list):
        tokens = [str(t) for t in spec]
    else:
        tokens = [t.strip() for t in str(spec).split(",") if t.strip()]

    names: list[str] = []
    for token in tokens:
        if token.isdigit():
            names.append(f"level{int(token)}.json")
        elif "-" in token and all(part.strip().isdigit() for part in token.split("-", 1)):
            lo, hi = (int(part) for part in token.split("-", 1))
            if lo > hi:
                raise ValueError(f"Invalid level range: {token!r}")
            names.extend(f"level{n}.json" for n in range(lo, hi + 1))
        else:
            names.append(token if token.endswith(".json") else f"{token}.json")

    if not names:
        raise ValueError("No levels selected")
    for name in names:
        if not os.path.exists(os.path.join(maps_dir, name)):
            raise ValueError(f"Level map not found: {os.path.join(maps_dir, name)}")
    return names
