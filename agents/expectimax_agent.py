"""Arkanoid agent that plans paddle rebounds with an expectimax search over exact flights.

The level's points are fixed (50 per hit), so the normalized score is maximized by clearing
the level in the least time. Between two paddle contacts the game is deterministic, and
``agents.fast_physics`` reproduces it exactly. When the ball leaves the paddle the agent
already knows where and when it will come back and which bricks it will have broken.

For that contact it compares the reachable paddle zones (left / center / right). Chance nodes
integrate the server's triangular angle distributions with stratified quantiles, and each
outcome is scored as ``hits - rho * seconds`` (average-reward form of "least time to clear")
plus the discounted value of the contact the ball returns to. Contact values come from value
iteration over a grid of contact positions, on the current board and on the boards the rebound
most often leaves behind. The search runs in stages (greedy, current-board values, next-board
values) spread across ticks under a time budget, so the agent never blocks the 30 Hz loop; the
last completed stage decides. The paddle then moves on the 28 px lattice to the exact position
that puts the contact inside the chosen zone.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Generator
from typing import Any

import numpy as np

from agents.base_agent import BaseAgent
from agents.fast_physics import Geometry, feasible_zones, fly, fly_batch, launch, triangular_quantile


class ExpectimaxAgent(BaseAgent):
    """Choose each rebound zone by expectimax with value-iterated contact values on the exact physics."""

    ROOT_SAMPLES = 32  # angle quantiles per zone for the actual contact
    GRID_BINS = 100  # contact positions in the value-iteration grid
    GRID_SAMPLES = 12  # angle quantiles per zone and grid contact
    GAMMA = 0.7  # discount per rebound for the value of the next contact
    VALUE_ITERATIONS = 40
    USE_VALUE_GRID = True
    HIT_CONTINUATION = 1.0  # weight of the next contact's value after a flight that hit bricks
    RHO_SCALE = 0.35  # time is priced below the observed hit rate: multi-hit flights pay off later
    NEXT_BOARD_TABLES = True
    NEXT_BOARD_LIMIT = 12
    NEXT_BOARD_BINS = 40
    NEXT_BOARD_SAMPLES = 6
    TICK_BUDGET_S = 0.012  # planning time per tick (the server ticks every 33 ms)
    BATCH_FLIGHTS = 128  # grid flights simulated per vectorized batch (one search step)
    ZONE_MARGIN = 1.0  # px kept inside the chosen zone and inside the paddle
    DEADLINE_TICKS = 4  # spare ticks kept when deciding when the paddle must start moving
    RHO_PRIOR = 0.3  # hits per second used before the level provides its own rate
    RHO_PRIOR_WEIGHT = 20.0  # seconds of play the prior is worth
    FLIGHT_CACHE_SIZE = 256  # boards whose grid flights are kept (0: no cache)
    ENDGAME_HITS = 0  # at or below this many hits left, minimize the expected time to clear (0: off)
    ENDGAME_BINS = 40
    ENDGAME_SAMPLES = 6
    ENDGAME_ITERATIONS = 150

    def __init__(self, *args: Any, unlimited_budget: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.unlimited_budget = unlimited_budget
        self._geo: Geometry | None = None
        self._level_key: tuple[Any, ...] | None = None
        self._start_hits = 0
        self._plan_key: tuple[Any, ...] | None = None
        self._plan: Generator[None, None, None] | None = None
        self._decisions: dict[int, int] = {}  # search stage -> best zone
        self._contact_x = -1.0
        self._last_timer: float | None = None
        self._last_wall: float | None = None
        self._interval = 1.0 / 30.0
        self._stride = 1.0  # mean server ticks between the states this agent answers
        self.max_depth_reached: list[int] = []
        self._flight_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._tables: dict[tuple[int, ...], Any] = {}
        self._endgame: dict[tuple[int, ...], Any] = {}

    # ------------------------------------------------------------------ search

    def _rho(self, state: dict[str, Any], health: list[int]) -> float:
        elapsed = float(state.get("level_time_elapsed", 0.0))
        done = self._start_hits - sum(h for h in health if h > 0)
        rate = (done + self.RHO_PRIOR * self.RHO_PRIOR_WEIGHT) / (elapsed + self.RHO_PRIOR_WEIGHT)
        return self.RHO_SCALE * rate

    def _feasible(self, geo: Geometry, contact_x: float) -> tuple[int, ...]:
        """Zones some paddle position in [0, width - paddle] can give for this contact."""
        return feasible_zones(geo, contact_x, self.ZONE_MARGIN)

    def _flight(self, geo: Geometry, health: list[int], contact_x: float, zone: int,
                quantile: float) -> tuple[float, int, bool, float, list[int]]:
        mode, half = geo.zone_angle(zone)
        angle = triangular_quantile(quantile, mode - half, mode + half, mode)
        return fly(geo, health, *launch(geo, contact_x, angle))

    def _flights(self, geo: Geometry, health: list[int], contact_x: float, zone: int,
                 n: int) -> list[tuple[float, int, bool, float, list[int]]]:
        """Flights from ``contact_x`` in ``zone`` for ``n`` stratified angle quantiles, in one batch."""
        mode, half = geo.zone_angle(zone)
        starts = np.array([launch(geo, contact_x, triangular_quantile((k + 0.5) / n, mode - half, mode + half, mode))
                           for k in range(n)])
        t, h, c, nx, nh = fly_batch(geo, health, starts[:, 0], starts[:, 1], starts[:, 2], starts[:, 3])
        return list(zip(t.tolist(), h.tolist(), c.tolist(), nx.tolist(), nh.tolist()))

    def _grid_flights(self, geo: Geometry, health: list[int], n_bins: int,
                      n: int) -> Generator[None, None, dict[str, Any]]:
        """Flights from every grid contact, zone and angle quantile on this board (cached per board).

        The outcomes do not depend on ``rho``, and most rebounds leave the board unchanged, so the next
        contact usually finds its grid here instead of simulating thousands of flights again.
        """
        key = (tuple(health), n_bins, n)
        cached = self._flight_cache.get(key)
        if cached is not None:
            return cached
        xs = np.linspace(geo.radius, geo.width - geo.radius, n_bins)
        secs = np.zeros((n_bins, 3, n))
        hits = np.zeros((n_bins, 3, n))
        next_x = np.zeros((n_bins, 3, n))
        cleared = np.zeros((n_bins, 3, n), dtype=bool)
        board = np.zeros((n_bins, 3, n), dtype=np.int32)  # index into ``boards``
        valid = np.zeros((n_bins, 3), dtype=bool)
        boards: dict[tuple[int, ...], int] = {}
        jobs = []  # (bin, zone column, sample, launch state)
        for i, x in enumerate(xs):
            for zone in self._feasible(geo, float(x)):
                valid[i, zone + 1] = True
                mode, half = geo.zone_angle(zone)
                for k in range(n):
                    angle = triangular_quantile((k + 0.5) / n, mode - half, mode + half, mode)
                    jobs.append((i, zone + 1, k, *launch(geo, float(x), angle)))
        # Batches of flights on the vectorized physics; one yield per batch keeps the tick budget.
        for start in range(0, len(jobs), self.BATCH_FLIGHTS):
            chunk = np.array(jobs[start:start + self.BATCH_FLIGHTS])
            idx = (chunk[:, 0].astype(int), chunk[:, 1].astype(int), chunk[:, 2].astype(int))
            t, h, c, nx, nh = fly_batch(geo, health, chunk[:, 3], chunk[:, 4], chunk[:, 5], chunk[:, 6])
            secs[idx], hits[idx], cleared[idx], next_x[idx] = t, h, c, nx
            board[idx] = [boards.setdefault(tuple(row), len(boards)) for row in nh.tolist()]
            yield
        data = {"xs": xs, "secs": secs, "hits": hits, "next_x": next_x, "cleared": cleared,
                "board": board, "boards": list(boards), "valid": valid}
        if self.FLIGHT_CACHE_SIZE:
            if len(self._flight_cache) >= self.FLIGHT_CACHE_SIZE:
                self._flight_cache.clear()
            self._flight_cache[key] = data
        return data

    def _contact_values(self, geo: Geometry, health: list[int], rho: float, n_bins: int, n: int,
                        out: list[Any]) -> Generator[None, None, None]:
        """Value of every contact position on this board, by value iteration over a contact grid.

        Each grid contact and zone is simulated over stratified angle quantiles; an outcome is worth
        ``hits - rho * seconds`` plus the discounted value of the contact the ball returns to. The board
        is held fixed inside the iteration, which is what lets the agent aim several rebounds ahead.
        """
        d = yield from self._grid_flights(geo, health, n_bins, n)
        xs, next_x, valid = d["xs"], d["next_x"], d["valid"]
        reward = d["hits"] - rho * d["secs"]
        cont = np.where(d["cleared"], 0.0,
                        self.GAMMA * np.where(d["hits"] > 0, self.HIT_CONTINUATION, 1.0))
        values = np.zeros(n_bins)
        for _ in range(self.VALUE_ITERATIONS):
            q = (reward + cont * np.interp(next_x, xs, values)).mean(axis=2)
            values = np.where(valid, q, -1e9).max(axis=1)
        out.append((xs, values))

    def _time_to_clear(self, geo: Geometry, health: list[int]) -> Generator[None, None, Any]:
        """Expected seconds to clear the level from every contact position, for an endgame board.

        This is the real objective (the level's points are fixed), with no time price and no discount.
        A rebound that breaks bricks leads to a board with fewer hits left, solved first by recursion;
        a rebound that misses keeps the board, which value iteration resolves. The values hold for the
        rest of the level, so they are kept until the level changes.
        """
        key = tuple(health)
        if key in self._endgame:
            return self._endgame[key]
        d = yield from self._grid_flights(geo, health, self.ENDGAME_BINS, self.ENDGAME_SAMPLES)
        xs, next_x, valid, board, cleared = d["xs"], d["next_x"], d["valid"], d["board"], d["cleared"]
        same = (board == d["boards"].index(key)) & ~cleared if key in d["boards"] else np.zeros_like(cleared)
        fixed = d["secs"].copy()
        for j, b in enumerate(d["boards"]):
            mask = (board == j) & ~cleared & (b != key)
            if mask.any():
                bx, bv = yield from self._time_to_clear(geo, list(b))
                fixed[mask] += np.interp(next_x[mask], bx, bv)
        values = np.zeros(len(xs))
        for _ in range(self.ENDGAME_ITERATIONS):
            q = (fixed + same * np.interp(next_x, xs, values)).mean(axis=2)
            values = np.where(valid, q, 1e6).min(axis=1)
        self._endgame[key] = (xs, values)
        return xs, values

    def _score(self, flights: dict[int, list[Any]], tables: dict[tuple[int, ...], Any],
               fallback: Any, rho: float) -> dict[int, float]:
        """Mean outcome per zone, valuing each return contact with the table of the board it leaves."""
        scores = {}
        for z, fs in flights.items():
            total = 0.0
            for t, hits, cleared, nx, nh in fs:
                total += hits - rho * t
                if not cleared:
                    xs, values = tables.get(tuple(nh), fallback)
                    total += self.GAMMA * float(np.interp(nx, xs, values))
            scores[z] = total / len(fs)
        return scores

    def _root(self, geo: Geometry, health: list[int], contact_x: float, zones: tuple[int, ...],
              rho: float) -> Generator[None, None, None]:
        n = self.ROOT_SAMPLES
        flights = {z: self._flights(geo, health, contact_x, z, n) for z in zones}
        # Depth 1 (greedy) decision first, so a decision exists even if the budget runs out.
        greedy = {z: sum(f[1] - rho * f[0] for f in fs) / n for z, fs in flights.items()}
        self._decisions[1] = max(zones, key=lambda z: (greedy[z], z == 0, -z))
        yield
        if len(zones) == 1 or not self.USE_VALUE_GRID:
            return
        table: list[Any] = []
        yield from self._contact_values(geo, health, rho, self.GRID_BINS, self.GRID_SAMPLES, table)
        tables = {tuple(health): table[0]}
        self._tables = tables  # kept for searches that extend this one
        scores = self._score(flights, tables, table[0], rho)
        self._decisions[2] = max(zones, key=lambda z: (scores[z], z == 0, -z))
        if 0 < sum(h for h in health if h > 0) <= self.ENDGAME_HITS:
            # Few hits left: compare zones by the exact expected time to clear the level. Solved boards
            # are kept, so a solve cut short by the budget resumes at the next contact.
            seconds = {}
            for z, fs in flights.items():
                total = 0.0
                for t, _, cleared, nx, nh in fs:
                    total += t
                    if not cleared:
                        bx, bv = yield from self._time_to_clear(geo, nh)
                        total += float(np.interp(nx, bx, bv))
                seconds[z] = total / n
            self._decisions[4] = min(zones, key=lambda z: (seconds[z], z != 0, z))
            return
        if not self.NEXT_BOARD_TABLES:
            return
        # Flights that break bricks leave a different board: value their return contacts on that board,
        # most frequent boards first. This is what sees a rebound that opens a way above the wall.
        counts: dict[tuple[int, ...], int] = {}
        for fs in flights.values():
            for f in fs:
                if f[1] and not f[2]:
                    counts[tuple(f[4])] = counts.get(tuple(f[4]), 0) + 1
        for board in sorted(counts, key=counts.__getitem__, reverse=True)[: self.NEXT_BOARD_LIMIT]:
            sub: list[Any] = []
            yield from self._contact_values(geo, list(board), rho, self.NEXT_BOARD_BINS,
                                            self.NEXT_BOARD_SAMPLES, sub)
            tables[board] = sub[0]
        scores = self._score(flights, tables, table[0], rho)
        self._decisions[3] = max(zones, key=lambda z: (scores[z], z == 0, -z))

    # ------------------------------------------------------------------ paddle

    def _positions(self, geo: Geometry, paddle_x: float, max_steps: int) -> dict[float, tuple[int, int]]:
        """Paddle positions reachable in ``max_steps`` moves: position -> (steps, first move)."""
        max_p = geo.width - geo.paddle_width
        start = round(paddle_x, 6)
        seen = {start: (0, 0)}
        frontier = deque([start])
        while frontier:
            p = frontier.popleft()
            steps, first = seen[p]
            if steps >= max_steps:
                continue
            for move in (-1, 1):
                q = round(min(max_p, max(0.0, p + move * geo.step)), 6)
                if q not in seen:
                    seen[q] = (steps + 1, first or move)
                    frontier.append(q)
        return seen

    def _zone_position(self, geo: Geometry, paddle_x: float, contact_x: float, zone: int,
                       max_steps: int) -> tuple[float, int] | None:
        """Reachable paddle position with the widest margin for ``zone``: (position, first move)."""
        m = self.ZONE_MARGIN
        best: tuple[float, int, float, int] | None = None
        for p, (steps, first) in self._positions(geo, paddle_x, max_steps).items():
            rel = contact_x - p
            if zone == -1:
                margin = min(rel + geo.radius, geo.left_limit - rel)
            elif zone == 0:
                margin = min(rel - geo.left_limit, geo.right_limit - rel)
            else:
                margin = min(rel - geo.right_limit, geo.paddle_width + geo.radius - rel)
            if margin < m:
                continue
            cand = (min(margin, 10.0), -steps, p, first)
            if best is None or cand > best:
                best = cand
        return (best[2], best[3]) if best is not None else None

    # ------------------------------------------------------------------ loop

    def _sync_level(self, state: dict[str, Any]) -> Geometry:
        bricks = state.get("bricks", [])
        key = (state.get("level"), state.get("stage_name"))
        ids = {int(b["index"]) for b in bricks}
        if self._geo is None or key != self._level_key or not ids.issubset(self._geo.slot):
            self._geo = Geometry(state)
            self._level_key = key
            self._start_hits = sum(h for h in self._geo.health_from(state) if h > 0)
            self._plan_key = None
            self._flight_cache.clear()
            self._endgame.clear()
        return self._geo

    def _step_plan(self) -> None:
        if self._plan is None:
            return
        # Spend at most ~40% of the observed time between states, so faster servers are not starved.
        budget = min(self.TICK_BUDGET_S, 0.4 * self._interval)
        deadline = None if self.unlimited_budget else time.perf_counter() + budget
        try:
            while True:
                next(self._plan)
                if deadline is not None and time.perf_counter() > deadline:
                    return
        except StopIteration:
            self._plan = None
            self.max_depth_reached.append(max(self._decisions, default=0))

    async def deliberate(self) -> dict[str, Any] | None:
        state = self.current_state
        if not state or state.get("game_over") or state.get("game_won"):
            return None
        timer = state.get("time_elapsed")
        if timer is not None and timer == self._last_timer:
            return None  # stale duplicate: our previous move may not be applied yet
        if timer is not None and self._last_timer is not None:
            # Server ticks between the states we answer: on a slow machine or an accelerated server we
            # see fewer states, so the paddle gets fewer moves per tick and must start moving earlier.
            gap = max(1, round((float(timer) - float(self._last_timer)) * 30.0))
            self._stride = 0.8 * self._stride + 0.2 * min(gap, 10)
        self._last_timer = timer
        now = time.perf_counter()
        if self._last_wall is not None:
            interval = now - self._last_wall
            self._interval = 0.9 * self._interval + 0.1 * min(interval, 0.1)
        self._last_wall = now

        geo = self._sync_level(state)
        health = geo.health_from(state)
        bx, by = float(state["ball_x"]), float(state["ball_y"])
        bvx, bvy = float(state["ball_vx"]), float(state["ball_vy"])
        paddle_x = float(state["paddle_x"])
        if by > geo.paddle_y - geo.radius + 1e-6:
            return None  # ball below the paddle plane: the life is lost, wait for the reset

        t, _, cleared, contact_x, contact_health = fly(geo, health, bx, by, bvx, bvy)
        if cleared:
            return None
        stride = math.ceil(self._stride - 0.05)  # server ticks per state we answer (1 when we keep up)
        ticks_left = int(t * geo.fps) // stride  # moves the paddle can still make before the contact
        key = tuple(contact_health)
        if key != self._plan_key or abs(contact_x - self._contact_x) > 0.01:
            self._plan_key = key
            self._decisions = {}
            self._contact_x = contact_x
            max_steps = max(0, ticks_left - 2)
            zones = tuple(z for z in self._feasible(geo, contact_x)
                          if self._zone_position(geo, paddle_x, contact_x, z, max_steps) is not None)
            if not zones:
                zones = (0,)
            self._plan = self._root(geo, contact_health, contact_x, zones, self._rho(state, health))

        # Keep searching while there is time; commit once the paddle has to start moving.
        worst_steps = int((geo.width - geo.paddle_width) / geo.step) + 2
        if ticks_left > worst_steps + self.DEADLINE_TICKS or self.unlimited_budget or not self._decisions:
            self._step_plan()
        elif self._plan is not None:
            self.max_depth_reached.append(max(self._decisions, default=0))
            self._plan = None

        zone = self._decisions[max(self._decisions)] if self._decisions else 0
        choice = self._zone_position(geo, paddle_x, contact_x, zone, max(0, ticks_left))
        for fallback in (0, -1, 1):
            if choice is None:
                choice = self._zone_position(geo, paddle_x, contact_x, fallback, max(0, ticks_left) + 1)
        if choice is None:
            # Nothing keeps a margin in time: head for the catch position closest to the paddle centre.
            target = min(geo.width - geo.paddle_width, max(0.0, contact_x - geo.paddle_width / 2))
            choice = (target, 1 if target > paddle_x else -1)
        target, first = choice
        if abs(target - paddle_x) < 1e-6 or first == 0:
            return None
        direction = "EAST" if first > 0 else "WEST"
        actions = state.get("valid_actions", state.get("actions", []))
        return next((a for a in actions if a.get("direction") == direction), None)


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Expectimax Arkanoid agent")
    parser.add_argument("--server", default="ws://localhost:8765/ws")
    parser.add_argument("--name", default="Expectimax Agent")
    parser.add_argument("--id", default=None)
    args = parser.parse_args()
    asyncio.run(ExpectimaxAgent(args.server, args.name, args.id).run())
