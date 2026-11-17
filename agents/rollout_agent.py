"""Arkanoid agent that checks the expectimax choice with rollouts of whole rebound sequences.

``ExpectimaxAgent`` values a rebound by its own flight plus a table of contact values computed on a
board held fixed, so it only sees a few rebounds ahead and treats later boards approximately. This
agent keeps that plan and then plays each candidate zone forward: the root flight with that zone,
then up to ``HORIZON`` more rebounds where a cheap base policy (the current board's contact-value
table) picks the zone and the angle is drawn from the server's distribution. Every outcome is
replayed on the exact physics, so the boards the rollout breaks its way into are the real ones.

All zones share the same random angle sequences (common random numbers), so most of the
randomness cancels in the comparison. The rollout choice replaces the expectimax one only when its
paired advantage is larger than its standard error; otherwise the expectimax decision stands.
This is one step of policy improvement over the base policy, done online at every contact.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from typing import Any

import numpy as np

from agents.expectimax_agent import ExpectimaxAgent
from agents.fast_physics import Geometry


class RolloutAgent(ExpectimaxAgent):
    """Expectimax planner followed by paired rollouts that may overrule its zone choice."""

    ROLLOUTS = 48  # angle sequences per zone
    MIN_ROLLOUTS = 16  # sequences before the rollouts may decide
    HORIZON = 10  # rebounds after the root one
    SWITCH_SE = 1.0  # standard errors the rollout advantage must exceed to overrule expectimax

    def _q_table(self, geo: Geometry, health: list[int], rho: float) -> Generator[None, None, Any]:
        """Contact values and per-zone action values of this board (flights come from the cache)."""
        table: list[Any] = []
        yield from self._contact_values(geo, health, rho, self.GRID_BINS, self.GRID_SAMPLES, table)
        xs, values = table[0]
        d = yield from self._grid_flights(geo, health, self.GRID_BINS, self.GRID_SAMPLES)
        reward = d["hits"] - rho * d["secs"]
        cont = np.where(d["cleared"], 0.0,
                        self.GAMMA * np.where(d["hits"] > 0, self.HIT_CONTINUATION, 1.0))
        q = (reward + cont * np.interp(d["next_x"], xs, values)).mean(axis=2)
        return xs, values, np.where(d["valid"], q, -1e9)

    def _root(self, geo: Geometry, health: list[int], contact_x: float, zones: tuple[int, ...],
              rho: float) -> Generator[None, None, None]:
        yield from super()._root(geo, health, contact_x, zones, rho)
        if len(zones) == 1 or not self.USE_VALUE_GRID:
            return
        incumbent = self._decisions[max(self._decisions)]
        xs, values, q = yield from self._q_table(geo, health, rho)
        x0, dx = float(xs[0]), float(xs[1] - xs[0])
        last = len(xs) - 1

        def base(x: float) -> int:
            i = min(last, max(0, int(round((x - x0) / dx))))
            return max(self._feasible(geo, x) or (0,), key=lambda z: q[i, z + 1])

        # Same angle sequences for every zone, fixed by the situation so a replan repeats them.
        seed = int(contact_x * 1000) * 7919 + sum(h for h in health if h > 0)
        draws = np.random.default_rng(seed).random((self.ROLLOUTS, self.HORIZON + 1))
        returns: dict[int, list[float]] = {z: [] for z in zones}
        for r in range(self.ROLLOUTS):
            for z in zones:
                h, x, zone, total = health, contact_x, z, 0.0
                for s in range(self.HORIZON + 1):
                    t, hits, cleared, nx, nh = self._flight(geo, h, x, zone, float(draws[r, s]))
                    total += hits - rho * t
                    yield
                    if cleared:
                        break
                    h, x = nh, nx
                    zone = base(x)
                else:
                    total += self.GAMMA * float(np.interp(x, xs, values))
                returns[z].append(total)
            if r + 1 >= self.MIN_ROLLOUTS:
                self._decisions[5] = self._rollout_choice(returns, incumbent)

    def _rollout_choice(self, returns: dict[int, list[float]], incumbent: int) -> int:
        """Best zone by mean return, kept only if it beats the incumbent beyond the paired noise."""
        best = max(returns, key=lambda z: sum(returns[z]))
        if best == incumbent:
            return incumbent
        diffs = [a - b for a, b in zip(returns[best], returns[incumbent])]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        return best if mean > self.SWITCH_SE * math.sqrt(var / n) else incumbent
