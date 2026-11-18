"""Arkanoid agent that picks each rebound by the learned time-to-clear of the board it leaves.

``ExpectimaxAgent`` values a rebound with contact values computed on the current board held
fixed, so it cannot see that opening a column pays off several rebounds later. The value network
of ``agents.value_model`` (trained offline by ``train_value.py``) estimates the expected seconds
still needed to clear the level from any board and contact position. For the actual contact this
agent simulates every reachable zone over stratified angle quantiles on the exact physics and
minimizes ``flight seconds + V(next board, next contact)``, the real objective of the level.

The network only knows the maps it was trained on; on any other layout the agent falls back to
the expectimax search it inherits.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from agents.expectimax_agent import ExpectimaxAgent
from agents.fast_physics import Geometry
from experiments.value_model import DEFAULT_MODEL_PATH, LevelNet, ValueModel


class ValueAgent(ExpectimaxAgent):
    """Expectimax agent whose root decision uses the learned time-to-clear when the map is known."""

    SWITCH_MARGIN: float | None = None  # None: the network decides alone; else seconds it must save to overrule

    def __init__(self, *args: Any, model_path: Path = DEFAULT_MODEL_PATH, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model = ValueModel.load(model_path)
        self._net: LevelNet | None = None
        self._columns: np.ndarray | None = None

    def _sync_level(self, state: dict[str, Any]) -> Geometry:
        old = self._geo
        geo = super()._sync_level(state)
        if geo is not old:
            self._net = self._model.for_geometry(geo) if self._model else None
            self._columns = self._net.columns(geo) if self._net else None
        return geo

    def _expected_seconds(self, flights: list[Any]) -> float:
        """Mean of flight seconds plus the network's seconds to clear from where each flight ends."""
        assert self._net is not None and self._columns is not None
        total = sum(t for t, *_ in flights)
        open_ = [(nx, nh) for _, _, cleared, nx, nh in flights if not cleared]
        if open_:
            boards = np.array([nh for _, nh in open_], dtype=np.float64)
            xs = np.array([nx for nx, _ in open_], dtype=np.float64)
            total += float(self._net.forward(self._net.features(boards, xs, self._columns)).sum())
        return total / len(flights)

    def _root(self, geo: Geometry, health: list[int], contact_x: float, zones: tuple[int, ...],
              rho: float) -> Generator[None, None, None]:
        if self._net is not None and self.SWITCH_MARGIN is None:
            seconds = yield from self._net_seconds(geo, health, contact_x, zones)
            self._decisions[1] = min(zones, key=lambda z: (seconds[z], z != 0, z))
            return
        yield from super()._root(geo, health, contact_x, zones, rho)
        if self._net is None or len(zones) == 1:
            return
        # The search decided; the network overrules it only when it expects a clearly faster clear.
        incumbent = self._decisions[max(self._decisions)]
        seconds = yield from self._net_seconds(geo, health, contact_x, zones)
        best = min(zones, key=lambda z: (seconds[z], z != 0, z))
        if seconds[best] < seconds[incumbent] - self.SWITCH_MARGIN:
            self._decisions[6] = best

    def _net_seconds(self, geo: Geometry, health: list[int], contact_x: float,
                     zones: tuple[int, ...]) -> Generator[None, None, dict[int, float]]:
        seconds = {}
        for z in zones:
            seconds[z] = self._expected_seconds(self._flights(geo, health, contact_x, z, self.ROOT_SAMPLES))
            yield
        return seconds
