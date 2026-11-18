"""Learned estimate of the time still needed to clear a level, from a paddle contact.

One small multilayer perceptron per map (numpy only, so it runs with the project's
dependencies). Its input is the remaining health of every destructible brick and the contact
position, encoded with radial bumps so the network can tell nearby contacts apart. Its output is
the expected seconds until the last brick breaks under the agent's own policy. The weights are
trained by ``train_value.py`` (fitted policy iteration on the exact flight model) and stored in
``training_data/value_model.npz``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agents.fast_physics import Geometry

DATA_DIR = Path(__file__).resolve().parent / "training_data"  # this folder's data

DEFAULT_MODEL_PATH = DATA_DIR / "value_model.npz"
RBF_CENTERS = np.linspace(6.0, 594.0, 25)
RBF_WIDTH = 24.0
TARGET_SCALE = 100.0  # seconds per network output unit


def layout_of(geo: Geometry, initial_health: list[int]) -> np.ndarray:
    """Brick layout rows ``(index, x, y, max_health or -1 for metal)`` identifying a map."""
    rows = [(brick_id, geo.lefts[i] + geo.radius, geo.tops[i] + geo.radius,
             -1.0 if geo.metal[i] else float(initial_health[i]))
            for i, brick_id in enumerate(geo.ids)]
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


class LevelNet:
    """Value network of one map plus the gather that maps a board onto its inputs."""

    def __init__(self, layout: np.ndarray, weights: list[np.ndarray]) -> None:
        self.layout = layout
        self.weights = weights
        destructible = layout[layout[:, 3] >= 0.0]
        self.indices = destructible[:, 0].astype(int)
        self.max_health = np.maximum(destructible[:, 3], 1.0)

    def matches(self, geo: Geometry) -> bool:
        known = {int(row[0]): (row[1], row[2]) for row in self.layout}
        for i, brick_id in enumerate(geo.ids):
            pos = known.get(brick_id)
            if pos is None or abs(pos[0] - (geo.lefts[i] + geo.radius)) > 0.5 \
                    or abs(pos[1] - (geo.tops[i] + geo.radius)) > 0.5:
                return False
        return True

    def columns(self, geo: Geometry) -> np.ndarray:
        """Geometry slot of each destructible brick of the layout (-1 when absent, i.e. broken)."""
        return np.array([geo.slot.get(int(i), -1) for i in self.indices], dtype=int)

    def features(self, health: np.ndarray, xs: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Inputs for boards ``health`` (rows in geometry slot order) and contacts ``xs``."""
        padded = np.concatenate([np.maximum(health, 0), np.zeros((len(health), 1))], axis=1)
        bricks = padded[:, columns] / self.max_health
        bumps = np.exp(-0.5 * ((xs[:, None] - RBF_CENTERS[None, :]) / RBF_WIDTH) ** 2)
        return np.concatenate([bricks, bumps, xs[:, None] / 600.0], axis=1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden = inputs
        last = len(self.weights) // 2 - 1
        for layer in range(last + 1):
            hidden = hidden @ self.weights[2 * layer] + self.weights[2 * layer + 1]
            if layer < last:
                hidden = np.maximum(hidden, 0.0)
        return hidden[:, 0] * TARGET_SCALE


class ValueModel:
    """Per-map value networks; ``for_geometry`` picks the one trained on the observed layout."""

    def __init__(self, nets: list[LevelNet]) -> None:
        self.nets = nets

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> ValueModel | None:
        try:
            data = np.load(path)
        except (OSError, ValueError):
            return None
        nets = []
        for level in range(int(data["levels"])):
            n_weights = int(data[f"n{level}"])
            weights = [data[f"w{level}_{k}"] for k in range(n_weights)]
            nets.append(LevelNet(data[f"layout{level}"], weights))
        return cls(nets)

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        arrays: dict[str, Any] = {"levels": np.array(len(self.nets))}
        for level, net in enumerate(self.nets):
            arrays[f"layout{level}"] = net.layout
            arrays[f"n{level}"] = np.array(len(net.weights))
            for k, w in enumerate(net.weights):
                arrays[f"w{level}_{k}"] = w.astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    def for_geometry(self, geo: Geometry) -> LevelNet | None:
        return next((net for net in self.nets if net.matches(geo)), None)
