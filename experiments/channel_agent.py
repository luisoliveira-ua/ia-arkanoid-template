"""Arkanoid agent that opens the side channels to the ceiling loop early.

When the ball gets between the ceiling and the top row of bricks it bounces between them and
breaks bricks several times faster than from below (level 1: about 1.5 bricks/s against 0.13).
The way up is the gap between a side wall and the outermost brick column, barely wider than
the ball. Breaking the outermost columns widens that gap, but ``ExpectimaxAgent`` values a
rebound on the board held fixed and never sees that payoff. This agent counts a hit on a brick
of an outermost column as ``1 + EDGE_WEIGHT`` hits (normalized so the average hit of the map
still counts 1) and keeps the rest of the expectimax search.
"""

from __future__ import annotations

import numpy as np

from agents.fast_physics import Geometry
from experiments.tail_agent import TailAgent


class ChannelAgent(TailAgent):
    """Expectimax planner that prefers hits on the brick columns next to the side walls."""

    EDGE_WEIGHT = 1.0

    def _map_weights(self, geo: Geometry, health: list[int]) -> np.ndarray | None:
        alive = [i for i, hp in enumerate(health) if hp != 0]
        if not alive:
            return None
        left = min(geo.lefts[i] for i in alive)
        right = max(geo.rights[i] for i in alive)
        w = np.zeros(len(health))
        for i, hp in enumerate(health):
            if hp > 0:
                edge = abs(geo.lefts[i] - left) < 1.0 or abs(geo.rights[i] - right) < 1.0
                w[i] = 1.0 + (self.EDGE_WEIGHT if edge else 0.0)
        hp = np.maximum(np.array(health, dtype=np.float64), 0.0)
        return w * hp.sum() / float((w * hp).sum())
