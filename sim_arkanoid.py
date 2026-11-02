"""Local Arkanoid simulator used to benchmark agents (not part of the project).

The physics are intentionally written independently from the agent's model:
sub-stepped circle-vs-rectangle collisions, random angle cones on the paddle
(triangular distribution inside each third), optional command latency, and a
selectable paddle-contact model. Everything the agent assumes about the server
(28 px per command, 30 ticks/s, angle cones) comes from the project's own code.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from agents.search_agent import SearchAgent
from student import StudentAgent

W, H = 600.0, 800.0
CEILING_Y = 82.0
PADDLE_Y = 740.0
PADDLE_W = 84.0
PADDLE_H = 14.0
RADIUS = 6.0
SPEED = 400.0
FPS = 30
STEP = 28.0
CONES = {0: (-45.0, 15.0), 1: (0.0, 8.0), 2: (45.0, 15.0)}


def tri(rng: random.Random, mode: float, half: float) -> float:
    return rng.triangular(mode - half, mode + half, mode)


@dataclass
class Brick:
    x: float
    y: float
    w: float
    h: float
    hits: int
    points: int
    metal: bool = False


def make_layout(kind: int, rng: random.Random) -> list[Brick]:
    bricks: list[Brick] = []
    if kind == 0:  # classic wall, points grow towards the top
        rows, cols = 6, 12
        for r in range(rows):
            for c in range(cols):
                hits = 2 if r < 2 else 1
                bricks.append(Brick(c * 50 + 1, 110 + r * 22, 48, 20, hits, 70 - r * 10))
    elif kind == 1:  # with metal blocks and holes
        for r in range(7):
            for c in range(12):
                if rng.random() < 0.12:
                    continue
                if r == 3 and c in (2, 5, 8, 11):
                    bricks.append(Brick(c * 50 + 1, 110 + r * 22, 48, 20, 99, 0, True))
                    continue
                bricks.append(Brick(c * 50 + 1, 110 + r * 22, 48, 20, 1 + (r % 3 == 0), 50 - r * 5))
    elif kind == 2:  # diamond, multi-hit core
        rows = 8
        for r in range(rows):
            half = r if r < 4 else 7 - r
            n = 2 + 2 * half
            start = (12 - n) // 2
            for c in range(n):
                core = 3 <= r <= 4 and 3 <= c <= n - 4
                bricks.append(Brick((start + c) * 50 + 1, 120 + r * 24, 48, 20, 3 if core else 1, 90 if core else 30 + r * 4))
    else:  # sparse, tall wall that reaches low
        for r in range(9):
            for c in range(12):
                if (r + c) % 3 == 0 and rng.random() < 0.6:
                    continue
                bricks.append(Brick(c * 50 + 1, 100 + r * 30, 48, 22, 1 + (r > 5), 100 - r * 8))
        for c in (1, 4, 7, 10):
            bricks.append(Brick(c * 50 + 1, 400, 48, 14, 99, 0, True))
    return bricks


@dataclass
class Game:
    layout: int
    seed: int
    latency: int = 0
    contact: str = "tick"  # "tick": first tick overlapping the paddle; "plane": exact crossing
    time_limit: float = 240.0
    remove_destroyed: bool = False
    rng: random.Random = field(init=False)
    bricks: list[Brick] = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed * 7919 + self.layout)
        self.bricks = make_layout(self.layout, self.rng)
        self.paddle_x = (W - PADDLE_W) / 2 + self.rng.choice((-84, -28, 0, 28, 56))
        self.lives = 3
        self.lives_lost = 0
        self.score = 0
        self.t = 0.0
        self.tick = 0
        self.pending: list[int] = [0] * self.latency
        self.won = False
        self.over = False
        self.timeout = False
        self.contacts = 0
        self.bricks_hit = 0
        self.max_score = sum(b.points for b in self.bricks if not b.metal)
        self._launch()

    # -- ball ---------------------------------------------------------
    def _launch(self) -> None:
        angle = math.radians(tri(self.rng, 0.0, 20.0))
        self.bx = min(max(self.paddle_x + PADDLE_W / 2, RADIUS), W - RADIUS)
        self.by = PADDLE_Y - RADIUS - 1
        self.bvx = SPEED * math.sin(angle)
        self.bvy = -SPEED * math.cos(angle)

    def observe(self) -> dict:
        bricks = []
        for b in self.bricks:
            if self.remove_destroyed and b.hits <= 0 and not b.metal:
                continue
            bricks.append({
                "x": b.x, "y": b.y, "width": b.w, "height": b.h,
                "hits_remaining": max(0, b.hits), "points": b.points, "indestructible": b.metal,
            })
        return {
            "width": W, "height": H, "ceiling_y": CEILING_Y, "paddle_y": PADDLE_Y,
            "paddle_width": PADDLE_W, "paddle_x": self.paddle_x, "ball_radius": RADIUS,
            "ball_speed": SPEED, "ball_x": self.bx, "ball_y": self.by,
            "ball_vx": self.bvx, "ball_vy": self.bvy, "lives": self.lives,
            "score": self.score, "level_score": self.score,
            "level_time_elapsed": round(self.t, 6), "level_time_limit": self.time_limit,
            "game_over": self.over, "game_won": self.won, "bricks": bricks,
            "valid_actions": [{"direction": "EAST"}, {"direction": "WEST"}],
        }

    def step(self, direction: str | None) -> None:
        if self.over or self.won:
            return
        cmd = 1 if direction == "EAST" else -1 if direction == "WEST" else 0
        self.pending.append(cmd)
        applied = self.pending.pop(0)
        self.paddle_x = min(max(self.paddle_x + applied * STEP, 0.0), W - PADDLE_W)

        sub = 3
        dt = 1.0 / FPS / sub
        for _ in range(sub):
            self._advance(dt)
            if self.over or self.won:
                break
        self.t += 1.0 / FPS
        self.tick += 1
        if not self.won and self.t >= self.time_limit:
            self.timeout = True
            self.over = True

    def _advance(self, dt: float) -> None:
        x, y = self.bx + self.bvx * dt, self.by + self.bvy * dt
        if x < RADIUS:
            x, self.bvx = 2 * RADIUS - x, abs(self.bvx)
        elif x > W - RADIUS:
            x, self.bvx = 2 * (W - RADIUS) - x, -abs(self.bvx)
        if y < CEILING_Y + RADIUS:
            y, self.bvy = 2 * (CEILING_Y + RADIUS) - y, abs(self.bvy)

        # bricks: circle vs rectangle, reflect on the dominant axis of the contact normal
        for b in self.bricks:
            if b.hits <= 0 and not b.metal:
                continue
            cx = min(max(x, b.x), b.x + b.w)
            cy = min(max(y, b.y), b.y + b.h)
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy > RADIUS * RADIUS:
                continue
            if dx == 0 and dy == 0:  # centre inside: push along shortest axis
                left, right = x - b.x, b.x + b.w - x
                up, down = y - b.y, b.y + b.h - y
                m = min(left, right, up, down)
                if m in (left, right):
                    self.bvx = -self.bvx
                else:
                    self.bvy = -self.bvy
            elif abs(dx) > abs(dy):
                self.bvx = abs(self.bvx) * (1 if dx > 0 else -1)
                x = cx + math.copysign(RADIUS + 1e-3, dx)
            else:
                self.bvy = abs(self.bvy) * (1 if dy > 0 else -1)
                y = cy + math.copysign(RADIUS + 1e-3, dy)
            self.bricks_hit += 1
            if not b.metal:
                b.hits -= 1
                if b.hits <= 0:
                    self.score += b.points
                    if all(bb.hits <= 0 for bb in self.bricks if not bb.metal):
                        self.won = True
            break

        # paddle
        if self.bvy > 0 and y + RADIUS >= PADDLE_Y and y - RADIUS <= PADDLE_Y + PADDLE_H:
            hit_x = x
            if self.contact == "plane":
                hit_x = self.bx + self.bvx * (PADDLE_Y - RADIUS - self.by) / self.bvy * 1.0
            if self.paddle_x - RADIUS <= hit_x <= self.paddle_x + PADDLE_W + RADIUS:
                rel = min(max(hit_x - self.paddle_x, 0.0), PADDLE_W)
                zone = 0 if rel < PADDLE_W / 3 else 1 if rel < 2 * PADDLE_W / 3 else 2
                mode, half = CONES[zone]
                ang = math.radians(tri(self.rng, mode, half))
                self.bvx, self.bvy = SPEED * math.sin(ang), -SPEED * math.cos(ang)
                x, y = hit_x, PADDLE_Y - RADIUS
                self.contacts += 1
        elif y - RADIUS > PADDLE_Y + PADDLE_H + 4:
            self.lives -= 1
            self.lives_lost += 1
            if self.lives <= 0:
                self.over = True
            else:
                self._launch()
                return
        self.bx, self.by = x, y


def run_episode(
    agent_name: str,
    layout: int,
    seed: int,
    latency: int = 0,
    contact: str = "tick",
    time_limit: float = 240.0,
) -> dict[str, Any]:
    """Run one agent against the local simulator and return comparable metrics."""
    if latency < 0:
        raise ValueError("latency must be non-negative")
    if contact not in {"tick", "plane"}:
        raise ValueError("contact must be 'tick' or 'plane'")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")

    game = Game(layout, seed, latency=latency, contact=contact, time_limit=time_limit)
    agent_type = {"student": StudentAgent, "search": SearchAgent}.get(agent_name)
    if agent_type is None:
        raise ValueError(f"unknown agent: {agent_name}")
    agent = agent_type(name=f"{agent_name.title()} Benchmark Agent")
    while not game.over and not game.won:
        agent.current_state = game.observe()
        decision = agent.deliberate()
        try:
            decision.send(None)
        except StopIteration as completed:
            action = completed.value
        else:
            decision.close()
            raise RuntimeError("The local simulator requires a non-blocking deliberate() method")
        direction = action.get("direction") if action else None
        game.step(direction)

    return {
        "agent": agent_name,
        "layout": layout,
        "seed": seed,
        "latency_ticks": latency,
        "contact_model": contact,
        "score": game.score,
        "elapsed_seconds": round(game.t, 3),
        "points_per_second": round(game.score / max(game.t, 1e-9), 3),
        "lives_lost": game.lives_lost,
        "lives_remaining": game.lives,
        "contacts": game.contacts,
        "bricks_hit": game.bricks_hit,
        "won": game.won,
        "timed_out": game.timeout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Arkanoid agents in a local simulator")
    parser.add_argument("--agent", choices=("student", "search"), default="student")
    parser.add_argument("--layouts", type=int, nargs="+", choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--games-per-layout", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--latency", type=int, default=0, help="Command delay in simulation ticks")
    parser.add_argument("--contact", choices=("tick", "plane"), default="tick")
    parser.add_argument("--time-limit", type=float, default=240.0)
    args = parser.parse_args()
    if args.games_per_layout < 1:
        parser.error("--games-per-layout must be at least 1")
    logging.disable(logging.CRITICAL)

    results = [
        run_episode(
            args.agent,
            layout,
            args.seed_start + index,
            latency=args.latency,
            contact=args.contact,
            time_limit=args.time_limit,
        )
        for layout in args.layouts
        for index in range(args.games_per_layout)
    ]
    summary = {
        "games": len(results),
        "wins": sum(result["won"] for result in results),
        "games_without_lost_lives": sum(result["lives_lost"] == 0 for result in results),
        "average_score": round(sum(result["score"] for result in results) / len(results), 2),
        "average_points_per_second": round(
            sum(result["points_per_second"] for result in results) / len(results), 3
        ),
        "average_lives_lost": round(
            sum(result["lives_lost"] for result in results) / len(results), 2
        ),
    }
    print(json.dumps({"summary": summary, "games": results}, indent=2))


if __name__ == "__main__":
    main()
