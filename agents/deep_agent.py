"""Arkanoid agent that extends the expectimax plan with a second rebound simulated exactly.

``ExpectimaxAgent`` values the contact a rebound returns to by interpolating a table computed on a
grid of contacts, with the board of that contact held fixed for the rest of the level. This agent
keeps that search and adds one more exact layer: for every sampled outcome of the root rebound it
simulates every zone of the next rebound from the exact return contact and on the exact board the
first flight left, and only then reads the tables. All second-layer flights of a contact run as a
few vectorized batches (``fly_batch`` with one board per ball), spread across ticks.

The maximum over zones of a sampled mean is biased upwards, so the second layer uses the same
angle quantiles for every zone (common random numbers) and many samples per zone.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np

from agents.expectimax_agent import ExpectimaxAgent
from agents.fast_physics import Geometry, fly_batch, launch, triangular_quantile


class DeepAgent(ExpectimaxAgent):
    """Expectimax planner whose root outcomes are valued by an exact second rebound."""

    DEEP_ROOT_SAMPLES = 16  # root angle quantiles per zone
    DEEP_CHILD_SAMPLES = 24  # angle quantiles per zone of the second rebound
    DEEP_BATCH = 768  # second-rebound flights per vectorized batch (one search step)

    def _table_value(self, board: tuple[int, ...], x: Any) -> Any:
        xs, values = self._tables.get(board, self._tables_fallback)
        return np.interp(x, xs, values)

    def _root(self, geo: Geometry, health: list[int], contact_x: float, zones: tuple[int, ...],
              rho: float) -> Generator[None, None, None]:
        yield from super()._root(geo, health, contact_x, zones, rho)
        if len(zones) == 1 or not self._tables or 4 in self._decisions:
            return  # one choice, no tables yet, or the endgame solver decided
        self._tables_fallback = self._tables[tuple(health)]
        k1, k2 = self.DEEP_ROOT_SAMPLES, self.DEEP_CHILD_SAMPLES
        roots = {z: self._flights(geo, health, contact_x, z, k1) for z in zones}
        yield
        # Second-rebound jobs: (root zone, root sample, child zone, child sample) from each open outcome.
        quantiles = {z: [triangular_quantile((k + 0.5) / k2, m - h, m + h, m)
                         for k in range(k2)] for z, (m, h) in ((z, geo.zone_angle(z)) for z in (-1, 0, 1))}
        meta, starts, boards = [], [], []
        for z, outcomes in roots.items():
            for i, (_, _, cleared, nx, nh) in enumerate(outcomes):
                if cleared:
                    continue
                for cz in self._feasible(geo, nx) or (0,):
                    for angle in quantiles[cz]:
                        meta.append((z, i, cz))
                        starts.append(launch(geo, nx, angle))
                        boards.append(nh)
        if not meta:
            return
        starts_a = np.array(starts)
        boards_a = np.array(boards, dtype=np.int64)
        reward = np.zeros(len(meta))
        for lo in range(0, len(meta), self.DEEP_BATCH):
            hi = lo + self.DEEP_BATCH
            t, hits, cleared, nx, nh = fly_batch(geo, boards_a[lo:hi], starts_a[lo:hi, 0], starts_a[lo:hi, 1],
                                                 starts_a[lo:hi, 2], starts_a[lo:hi, 3])
            future = np.array([0.0 if c else float(self._table_value(tuple(b), x))
                               for c, b, x in zip(cleared.tolist(), nh.tolist(), nx.tolist())])
            reward[lo:hi] = hits - rho * t + self.GAMMA * future
            yield
        # Child value: best zone by mean over its samples; root zone: mean over its outcomes.
        child: dict[tuple[int, int], dict[int, list[float]]] = {}
        for (z, i, cz), r in zip(meta, reward.tolist()):
            child.setdefault((z, i), {}).setdefault(cz, []).append(r)
        scores = {}
        for z, outcomes in roots.items():
            total = 0.0
            for i, (t, hits, cleared, _, _) in enumerate(outcomes):
                total += hits - rho * t
                if not cleared:
                    total += self.GAMMA * max(sum(v) / len(v) for v in child[(z, i)].values())
            scores[z] = total / k1
        self._decisions[5] = max(zones, key=lambda z: (scores[z], z == 0, -z))
