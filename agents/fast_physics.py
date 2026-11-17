"""Event-driven replica of the ball physics in ``server/logic.py``.

The server advances the ball tick by tick with swept collision tests. Between
two paddle contacts nothing random happens, so the same collisions can be
resolved directly from one event to the next. This module reproduces the
server's tests (walls, ceiling, bricks with their surface normals, nudges and
clamps, tie order) so a whole flight is predicted exactly in about a
millisecond, which is what the planner needs to compare many rebounds.
"""

from __future__ import annotations

import math
from typing import Any

INF = float("inf")


class Geometry:
    """Static arena and brick layout of one level (brick health is kept apart)."""

    __slots__ = (
        "width", "ceiling", "radius", "speed", "paddle_y", "paddle_width", "step",
        "left_limit", "right_limit", "alpha", "side", "beta", "fps",
        "ids", "slot", "min_x", "max_x", "min_y", "max_y",
        "lefts", "rights", "tops", "bottoms", "metal", "points",
    )

    def __init__(self, state: dict[str, Any], angles: tuple[float, float, float] = (8.0, 45.0, 15.0),
                 ratios: tuple[float, float] = (0.3333, 0.3333), fps: float = 30.0,
                 step: float = 28.0) -> None:
        self.width = float(state.get("width", 600.0))
        self.ceiling = float(state.get("ceiling_y", 82.0))
        self.radius = float(state.get("ball_radius", 6.0))
        self.speed = float(state.get("ball_speed", 400.0))
        self.paddle_y = float(state.get("paddle_y", 740.0))
        self.paddle_width = float(state.get("paddle_width", 84.0))
        self.step = step
        self.fps = fps
        self.left_limit = self.paddle_width * ratios[0]
        self.right_limit = self.paddle_width * (1.0 - ratios[1])
        self.alpha, self.side, self.beta = angles
        bricks = sorted(state.get("bricks", []), key=lambda b: int(b["index"]))
        self.ids = [int(b["index"]) for b in bricks]
        self.slot = {brick_id: i for i, brick_id in enumerate(self.ids)}
        r = self.radius
        self.lefts = [float(b["x"]) - r for b in bricks]
        self.rights = [float(b["x"]) + float(b["width"]) + r for b in bricks]
        self.tops = [float(b["y"]) - r for b in bricks]
        self.bottoms = [float(b["y"]) + float(b["height"]) + r for b in bricks]
        self.metal = [bool(b.get("indestructible")) for b in bricks]
        self.points = [int(b.get("points", 0)) for b in bricks]
        self.min_x = min(self.lefts, default=0.0)
        self.max_x = max(self.rights, default=0.0)
        self.min_y = min(self.tops, default=0.0)
        self.max_y = max(self.bottoms, default=0.0)

    def health_from(self, state: dict[str, Any]) -> list[int]:
        """Brick health in layout order; bricks missing from the observation are broken."""
        health = [0] * len(self.ids)
        for brick in state.get("bricks", []):
            i = self.slot.get(int(brick["index"]))
            if i is not None and brick.get("active", True):
                health[i] = -1 if self.metal[i] else int(brick.get("health", 1))
        return health

    def zone_angle(self, zone: int) -> tuple[float, float]:
        """Nominal angle and half cone in degrees for zone -1 (left), 0 (center), 1 (right)."""
        if zone == 0:
            return 0.0, self.alpha
        return zone * self.side, self.beta


def feasible_zones(geo: Geometry, contact_x: float, margin: float = 1.0) -> tuple[int, ...]:
    """Zones (-1, 0, 1) that some paddle position in [0, width - paddle] gives for this contact."""
    max_p = geo.width - geo.paddle_width
    spans = (
        (-1, -geo.radius + margin, geo.left_limit - margin),
        (0, geo.left_limit + margin, geo.right_limit - margin),
        (1, geo.right_limit + margin, geo.paddle_width + geo.radius - margin),
    )
    return tuple(zone for zone, lo, hi in spans if contact_x - hi <= max_p and contact_x - lo >= 0.0)


def triangular_quantile(q: float, low: float, high: float, mode: float) -> float:
    """Inverse CDF of the triangular distribution used by ``random.triangular``."""
    split = (mode - low) / (high - low)
    if q < split:
        return low + math.sqrt(q * (high - low) * (mode - low))
    return high - math.sqrt((1.0 - q) * (high - low) * (high - mode))


def fly(geo: Geometry, health: list[int], x: float, y: float, vx: float, vy: float,
        max_time: float = 60.0) -> tuple[float, int, bool, float, list[int]]:
    """Advance the ball until it reaches the paddle plane moving down, or the level is cleared.

    ``health`` is copied, never modified. Returns ``(seconds, hits_that_count, cleared, x_at_paddle,
    new_health)``; a "hit that counts" is one point of damage to a destructible brick.
    """
    h = health[:]
    r = geo.radius
    lefts, rights, tops, bottoms, metal = geo.lefts, geo.rights, geo.tops, geo.bottoms, geo.metal
    alive = [i for i, hp in enumerate(h) if hp != 0]
    remaining = sum(hp for hp in h if hp > 0)
    min_x, max_x, min_y, max_y = geo.min_x, geo.max_x, geo.min_y, geo.max_y
    wall_lo, wall_hi = r, geo.width - r
    ceil = geo.ceiling + r
    plane = geo.paddle_y - r
    speed = geo.speed
    t_total = 0.0
    hits = 0

    while t_total < max_time:
        # Same event order as Arkanoid._step_simulation: walls, ceiling, paddle, then bricks (strictly earlier).
        t_min = INF
        kind = 0
        if vx < 0.0:
            t_min, kind = (wall_lo - x) / vx, 1
        elif vx > 0.0:
            t_min, kind = (wall_hi - x) / vx, 2
        if vy < 0.0:
            t = (ceil - y) / vy
            if t < t_min:
                t_min, kind = t, 3
        elif vy > 0.0:
            t = (plane - y) / vy
            if t < t_min:
                t_min, kind = t, 4
        if t_min < 0.0:
            t_min = 0.0

        # Bricks: only when the segment's bounding box touches the brick zone.
        hit = -1
        hnx = hny = 0.0
        ex = x + vx * t_min
        ey = y + vy * t_min
        if not ((ey if vy < 0 else y) > max_y or (y if vy < 0 else ey) < min_y
                or (ex if vx < 0 else x) > max_x or (x if vx < 0 else ex) < min_x):
            inv_x = 1.0 / vx if vx != 0.0 else 0.0
            inv_y = 1.0 / vy if vy != 0.0 else 0.0
            for i in alive:
                if vx != 0.0:
                    t1x = (lefts[i] - x) * inv_x
                    t2x = (rights[i] - x) * inv_x
                    if t1x > t2x:
                        t1x, t2x = t2x, t1x
                        nx = 1.0
                    else:
                        nx = -1.0
                else:
                    if x < lefts[i] or x > rights[i]:
                        continue
                    t1x, t2x, nx = -1e9, 1e9, 0.0
                if vy != 0.0:
                    t1y = (tops[i] - y) * inv_y
                    t2y = (bottoms[i] - y) * inv_y
                    if t1y > t2y:
                        t1y, t2y = t2y, t1y
                        ny = 1.0
                    else:
                        ny = -1.0
                else:
                    if y < tops[i] or y > bottoms[i]:
                        continue
                    t1y, t2y, ny = -1e9, 1e9, 0.0
                t_enter = t1x if t1x > t1y else t1y
                t_exit = t2x if t2x < t2y else t2y
                if t_enter <= t_exit and t_exit >= 0.0:
                    cnx = nx if t1x > t1y else 0.0
                    cny = ny if t1y >= t1x else 0.0
                    if vx * cnx + vy * cny < 0.0:
                        cand = t_enter if t_enter > 0.0 else 0.0
                        if cand < t_min:
                            t_min, hit, hnx, hny = cand, i, cnx, cny
            if hit >= 0:
                kind = 5

        t_total += t_min
        if kind == 4:
            return t_total, hits, False, x + vx * t_min, h
        x += vx * t_min
        y += vy * t_min
        if kind == 1:
            x, vx = wall_lo, abs(vx)
        elif kind == 2:
            x, vx = wall_hi, -abs(vx)
        elif kind == 3:
            y, vy = ceil, abs(vy)
        elif kind == 5:
            dot = vx * hnx + vy * hny
            vx -= 2.0 * dot * hnx
            vy -= 2.0 * dot * hny
            spd = math.hypot(vx, vy)
            vx = vx / spd * speed
            vy = vy / spd * speed
            if abs(hnx) > 0.5:
                x = lefts[hit] - 1e-2 if hnx < 0.0 else rights[hit] + 1e-2
            elif abs(hny) > 0.5:
                y = tops[hit] - 1e-2 if hny < 0.0 else bottoms[hit] + 1e-2
            x = max(wall_lo, min(wall_hi, x))
            y = max(ceil, y)
            if not metal[hit]:
                h[hit] -= 1
                hits += 1
                remaining -= 1
                if h[hit] == 0:
                    alive = [j for j in alive if j != hit]
                if remaining == 0:
                    return t_total, hits, True, x, h
        else:
            break
    return t_total, hits, False, x, h


def launch(geo: Geometry, contact_x: float, angle_deg: float) -> tuple[float, float, float, float]:
    """Ball position and velocity right after a paddle rebound at ``contact_x`` with ``angle_deg``."""
    r = geo.radius
    x = max(r, min(geo.width - r, contact_x))
    a = math.radians(angle_deg)
    return x, geo.paddle_y - r, geo.speed * math.sin(a), -geo.speed * math.cos(a)


def fly_batch(geo: Geometry, health: Any, x: Any, y: Any, vx: Any, vy: Any,
              max_time: float = 60.0) -> tuple[Any, Any, Any, Any, Any]:
    """Vectorized ``fly`` for many balls (numpy arrays in, numpy arrays out).

    ``health`` is one board shared by all balls, or one board per ball (shape ``(n, bricks)``).

    Every ball runs the same event loop as ``fly``, all of them advanced together one event per
    iteration; balls that reach the paddle plane (or clear the level) drop out of the loop.
    Returns ``(seconds, hits, cleared, x_at_paddle, new_health)`` with ``new_health`` of shape
    ``(n, bricks)``. It is several times faster than calling ``fly`` in a loop.
    """
    import numpy as np

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    vx = np.array(vx, dtype=np.float64)
    vy = np.array(vy, dtype=np.float64)
    n = len(x)
    full = np.asarray(health, dtype=np.int64)
    if full.ndim == 1:
        full = np.tile(full, (n, 1))
    cols = np.flatnonzero((full != 0).any(axis=0))  # bricks broken on every board never collide again
    h = full[:, cols].copy()
    secs = np.zeros(n)
    hits = np.zeros(n, dtype=np.int64)
    cleared = np.zeros(n, dtype=bool)
    out_x = np.zeros(n)
    remaining = np.maximum(h, 0).sum(axis=1)
    lefts, rights = np.array(geo.lefts)[cols], np.array(geo.rights)[cols]
    tops, bottoms = np.array(geo.tops)[cols], np.array(geo.bottoms)[cols]
    metal = np.array(geo.metal, dtype=bool)[cols]
    r = geo.radius
    wall_lo, wall_hi, ceil, plane, speed = r, geo.width - r, geo.ceiling + r, geo.paddle_y - r, geo.speed
    act = np.flatnonzero(remaining > 0)  # a board with nothing left to break is already cleared
    cleared[remaining == 0] = True
    out_x[remaining == 0] = x[remaining == 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        while act.size:
            ax, ay, avx, avy = x[act], y[act], vx[act], vy[act]
            # Walls, ceiling and paddle plane, in the order of ``fly``.
            t_min = np.full(act.size, INF)
            kind = np.zeros(act.size, dtype=np.int64)
            t_wall = np.where(avx < 0.0, (wall_lo - ax) / avx, (wall_hi - ax) / avx)
            has = avx != 0.0
            t_min[has] = t_wall[has]
            kind[has] = np.where(avx[has] < 0.0, 1, 2)
            t_vert = np.where(avy < 0.0, (ceil - ay) / avy, (plane - ay) / avy)
            better = (avy != 0.0) & (t_vert < t_min)
            t_min[better] = t_vert[better]
            kind[better] = np.where(avy[better] < 0.0, 3, 4)
            t_min = np.maximum(t_min, 0.0)

            # Bricks: slab test against every live brick, earliest approaching contact wins.
            bx, by = ax[:, None], ay[:, None]
            inv_x = np.where(avx != 0.0, 1.0 / avx, 0.0)[:, None]
            inv_y = np.where(avy != 0.0, 1.0 / avy, 0.0)[:, None]
            a1, a2 = (lefts - bx) * inv_x, (rights - bx) * inv_x
            swap_x = a1 > a2
            t1x, t2x = np.where(swap_x, a2, a1), np.where(swap_x, a1, a2)
            nx = np.where(swap_x, 1.0, -1.0)
            b1, b2 = (tops - by) * inv_y, (bottoms - by) * inv_y
            swap_y = b1 > b2
            t1y, t2y = np.where(swap_y, b2, b1), np.where(swap_y, b1, b2)
            ny = np.where(swap_y, 1.0, -1.0)
            ok = h[act] != 0
            still_x = (avx == 0.0)[:, None]
            if still_x.any():
                ok &= ~still_x | ((bx >= lefts) & (bx <= rights))
                t1x, t2x, nx = (np.where(still_x, -1e9, t1x), np.where(still_x, 1e9, t2x),
                                np.where(still_x, 0.0, nx))
            still_y = (avy == 0.0)[:, None]
            if still_y.any():
                ok &= ~still_y | ((by >= tops) & (by <= bottoms))
                t1y, t2y, ny = (np.where(still_y, -1e9, t1y), np.where(still_y, 1e9, t2y),
                                np.where(still_y, 0.0, ny))
            x_first = t1x > t1y
            t_enter = np.where(x_first, t1x, t1y)
            t_exit = np.where(t2x < t2y, t2x, t2y)
            cnx = np.where(x_first, nx, 0.0)
            cny = np.where(t1y >= t1x, ny, 0.0)
            ok &= (t_enter <= t_exit) & (t_exit >= 0.0) & (avx[:, None] * cnx + avy[:, None] * cny < 0.0)
            cand = np.where(ok, np.maximum(t_enter, 0.0), INF)
            hit = np.argmin(cand, axis=1)
            rows = np.arange(act.size)
            brick = cand[rows, hit] < t_min
            t_min = np.where(brick, cand[rows, hit], t_min)
            kind[brick] = 5

            secs[act] += t_min
            done = (kind == 4) | (kind == 0)
            out_x[act[done]] = ax[done] + avx[done] * t_min[done]
            ax = ax + avx * t_min
            ay = ay + avy * t_min
            k = kind == 1
            ax[k], avx[k] = wall_lo, np.abs(avx[k])
            k = kind == 2
            ax[k], avx[k] = wall_hi, -np.abs(avx[k])
            k = kind == 3
            ay[k], avy[k] = ceil, np.abs(avy[k])

            k = np.flatnonzero(brick)
            if k.size:
                j = hit[k]
                hnx, hny = cnx[k, j], cny[k, j]
                dot = avx[k] * hnx + avy[k] * hny
                nvx = avx[k] - 2.0 * dot * hnx
                nvy = avy[k] - 2.0 * dot * hny
                spd = np.hypot(nvx, nvy)
                avx[k], avy[k] = nvx / spd * speed, nvy / spd * speed
                px, py = ax[k], ay[k]
                side = np.abs(hnx) > 0.5
                px = np.where(side, np.where(hnx < 0.0, lefts[j] - 1e-2, rights[j] + 1e-2), px)
                top = ~side & (np.abs(hny) > 0.5)
                py = np.where(top, np.where(hny < 0.0, tops[j] - 1e-2, bottoms[j] + 1e-2), py)
                ax[k] = np.clip(px, wall_lo, wall_hi)
                ay[k] = np.maximum(py, ceil)
                soft = ~metal[j]
                ball = act[k[soft]]
                h[ball, j[soft]] -= 1
                hits[ball] += 1
                remaining[ball] -= 1
                won = ball[remaining[ball] == 0]
                cleared[won] = True
                out_x[won] = ax[k[soft]][remaining[ball] == 0]

            x[act], y[act], vx[act], vy[act] = ax, ay, avx, avy
            timeout = secs[act] >= max_time
            out_x[act[timeout]] = ax[timeout]
            act = act[~(done | cleared[act] | timeout)]
    new_health = full.copy()
    new_health[:, cols] = h
    return secs, hits, cleared, out_x, new_health
