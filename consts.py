"""Game constants for Arkanoid."""

from typing import Any

# Arena screen dimensions
SCREEN_WIDTH: int = 600
SCREEN_HEIGHT: int = 800
CEILING_Y: float = 82.0

# Paddle dimensions, position, and physics
PADDLE_WIDTH: float = 84.0
PADDLE_HEIGHT: float = 14.0
PADDLE_Y: float = 740.0
PADDLE_SPEED: float = 350.0
PADDLE_STEP_DISTANCE: float = 28.0
PADDLE_LEFT_RATIO: float = 0.3333
PADDLE_CENTER_RATIO: float = 0.3334
PADDLE_RIGHT_RATIO: float = 0.3333

PADDLE_ALPHA_DEG: float = 8.0
PADDLE_SIDE_DEG: float = 45.0
PADDLE_BETA_DEG: float = 15.0

# Ball properties
BALL_RADIUS: float = 6.0
BALL_SPEED: float = 400.0
BALL_INITIAL_SPEED: float = 400.0

# Standard brick dimensions
BRICK_WIDTH: float = 50.0
BRICK_HEIGHT: float = 20.0

# Brick types configuration (metal has 0 points)
BRICK_TYPES: dict[str, dict[str, Any]] = {
    "red": {"health": 1, "indestructible": False, "points": 50, "color": "#bf616a"},
    "blue": {"health": 2, "indestructible": False, "points": 100, "color": "#81a1c1"},
    "green": {"health": 3, "indestructible": False, "points": 150, "color": "#a3be8c"},
    "yellow": {"health": 4, "indestructible": False, "points": 200, "color": "#ebcb8b"},
    "metal": {"health": -1, "indestructible": True, "points": 0, "color": "#d8dee9"},
}

# Virtual discrete grid defaults for level maps
GRID_COLS: int = 9
GRID_ROWS: int = 10
GRID_MARGIN_X: float = 35.0
GRID_MARGIN_Y: float = 120.0
GRID_GAP_X: float = 10.0
GRID_GAP_Y: float = 5.0

# Game rules and session parameters
GAME_LIVES: int = 3
GAME_FPS: int = 30
GAME_MAPS_DIR: str = "maps"
GAME_HIGHSCORES_FILE: str = "highscores.csv"
GAME_LEVEL_TIME_LIMIT: float = 400.0
GAME_LEVEL_SEQUENCE: list[str] = ["level1.json", "level2.json", "level3.json"]
