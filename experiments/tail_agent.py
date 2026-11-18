"""Arkanoid agent that breaks the bricks usually left for the end while it still has the chance.

Most of a level's lost time is its tail: the last few hits are lone bricks that take many flights
to reach. ``tail_weights.py`` measured, on recorded expectimax games, how often each brick is
still standing when only a few hits are left. This agent keeps the expectimax search and only
changes what a hit is worth in it: a hit on brick ``i`` counts ``1 + TAIL_WEIGHT * late_i``,
normalized so the average hit of the map still counts 1 (the time price stays calibrated). The
search then prefers rebounds that chip at those bricks while other bricks are still around.

Unknown maps (no measured rates) play exactly like ``ExpectimaxAgent``.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from agents.expectimax_agent import ExpectimaxAgent
from agents.fast_physics import Geometry

DATA_DIR = Path(__file__).resolve().parent / "training_data"  # this folder's data

DEFAULT_WEIGHTS_PATH = DATA_DIR / "tail_weights.json"


class TailAgent(ExpectimaxAgent):
    """Expectimax planner with hits weighted by how often each brick ends up in the tail."""

    TAIL_WEIGHT = 1.0

    def __init__(self, *args: Any, weights_path: Path = DEFAULT_WEIGHTS_PATH, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:
            self._rates = json.loads(weights_path.read_text(encoding="utf-8"))["maps"]
        except (OSError, ValueError, KeyError):
            self._rates = {}
        self._weights: np.ndarray | None = None

    def _sync_level(self, state: dict[str, Any]) -> Geometry:
        old = self._geo
        geo = super()._sync_level(state)
        if geo is not old:
            self._weights = self._map_weights(geo, geo.health_from(state))
        return geo

    def _map_weights(self, geo: Geometry, health: list[int]) -> np.ndarray | None:
        """Hit weight per brick slot from the measured rates of the map with these brick positions."""
        slots = {(round(geo.lefts[i] + geo.radius), round(geo.tops[i] + geo.radius)): i
                 for i, hp in enumerate(health) if hp > 0}
        for rows in self._rates.values():
            found = {(round(r["x"]), round(r["y"])): r["late"] for r in rows}
            if set(found) != set(slots):
                continue
            w = np.zeros(len(health))
            for pos, i in slots.items():
                w[i] = 1.0 + self.TAIL_WEIGHT * found[pos]
            hp = np.maximum(np.array(health, dtype=np.float64), 0.0)
            return w * hp.sum() / float((w * hp).sum())
        return None

    def _weighted(self, health: list[int], new_health: Any) -> Any:
        """Weighted hits of flights from ``health`` that left the boards ``new_health``."""
        assert self._weights is not None
        damage = np.maximum(np.asarray(health) - np.asarray(new_health), 0)
        return damage @ self._weights

    def _flights(self, geo: Geometry, health: list[int], contact_x: float, zone: int,
                 n: int) -> list[tuple[float, int, bool, float, list[int]]]:
        flights = super()._flights(geo, health, contact_x, zone, n)
        if self._weights is None:
            return flights
        hits = self._weighted(health, [f[4] for f in flights])
        return [(t, float(w) if h else 0, c, nx, nh) for (t, h, c, nx, nh), w in zip(flights, hits)]

    def _grid_flights(self, geo: Geometry, health: list[int], n_bins: int,
                      n: int) -> Generator[None, None, dict[str, Any]]:
        data = yield from super()._grid_flights(geo, health, n_bins, n)
        if self._weights is not None and data["boards"]:
            # Recomputed from the boards every time, so a cached grid is never weighted twice.
            per_board = self._weighted(health, data["boards"])
            data["hits"] = np.where(data["valid"][:, :, None], per_board[data["board"]], 0.0)
        return data
