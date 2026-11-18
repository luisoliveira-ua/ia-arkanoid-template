"""Learn the expectimax agent's own time-to-clear, to value boards inside its search.

``ExpectimaxAgent`` values the contact a rebound returns to with tables computed on a board held
fixed, so it cannot see that opening a column pays off several rebounds later. This script plays
whole levels with the expectimax decisions on the exact flight model (no 30 Hz ticks, the zone is
always reachable, the angle is drawn from the server's distribution), records every contact
(board, contact position, flight seconds) and fits one value network per map to the seconds
still needed to clear the level from that contact under that policy.

The targets are TD(lambda) returns: the first fit uses the Monte Carlo remaining time, and every
later fit bootstraps on the previous network, which has less noise than the final clear time of
one game. The networks are stored in ``training_data/value_model_expectimax.npz`` with the format
of ``agents/value_model.py``.

    python train_expectimax_value.py --games 600 --td-rounds 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from agents.expectimax_agent import ExpectimaxAgent
from agents.fast_physics import fly, launch
from experiments.value_model import LevelNet, ValueModel, layout_of
from experiments.train_value import LEVELS, TIME_LIMIT, fit, level_setup

DATA_DIR = Path(__file__).resolve().parent / "training_data"  # this folder's data

DEFAULT_OUT = DATA_DIR / "value_model_expectimax.npz"
GAMES_DIR = DATA_DIR / "expectimax_games"


def expectimax_zone(agent: ExpectimaxAgent, geo: Any, health: list[int], contact_x: float,
                    elapsed: float, start_hits: int) -> int:
    """Zone the expectimax search picks for this contact, with its full (unlimited) search."""
    zones = agent._feasible(geo, contact_x) or (0,)
    if len(zones) == 1:
        return zones[0]
    done = start_hits - sum(h for h in health if h > 0)
    w = agent.RHO_PRIOR_WEIGHT
    rho = agent.RHO_SCALE * (done + agent.RHO_PRIOR * w) / (elapsed + w)
    agent._decisions = {}
    for _ in agent._root(geo, health, contact_x, zones, rho):
        pass
    return agent._decisions[max(agent._decisions)]


def play(job: tuple[int, int]) -> dict[str, Any]:
    """One level with expectimax decisions, from the server's launch to the last brick."""
    level, seed = job
    rng = random.Random(seed)
    geo, health = level_setup(level)
    agent = ExpectimaxAgent("ws://localhost:1/ws", "Train", unlimited_budget=True)
    start_hits = sum(h for h in health if h > 0)
    a = math.radians(rng.uniform(-30.0, 30.0))
    t, _, cleared, contact_x, health = fly(geo, health, geo.width / 2.0, geo.paddle_y - geo.radius - 2.0,
                                           geo.speed * math.sin(a), -geo.speed * math.cos(a))
    elapsed = t
    boards, xs, secs, ends = [], [], [], []
    while not cleared and elapsed < TIME_LIMIT:
        zone = expectimax_zone(agent, geo, health, contact_x, elapsed, start_hits)
        mode, half = geo.zone_angle(zone)
        boards.append(health)
        xs.append(contact_x)
        t, _, cleared, contact_x, health = fly(geo, health, *launch(geo, contact_x,
                                                                    rng.triangular(mode - half, mode + half, mode)))
        elapsed += t
        secs.append(t)
        ends.append(cleared)
    return {"level": level, "seed": seed, "seconds": elapsed, "cleared": cleared,
            "norm": 50.0 * start_hits / elapsed if cleared else 0.0,
            "boards": np.array(boards, dtype=np.int8).reshape(len(boards), -1),
            "xs": np.array(xs), "secs": np.array(secs), "ends": np.array(ends, dtype=bool)}


def td_targets(game: dict[str, Any], values: np.ndarray | None, lam: float) -> np.ndarray:
    """TD(lambda) returns of every contact; ``values`` are the network's estimates (None: Monte Carlo)."""
    n = len(game["secs"])
    out = np.zeros(n)
    g = 0.0  # lambda-return of the next contact
    for i in range(n - 1, -1, -1):
        if game["ends"][i] or i == n - 1:
            g = game["secs"][i]
        else:
            nxt = g if values is None else lam * g + (1.0 - lam) * values[i + 1]
            g = game["secs"][i] + nxt
        out[i] = g
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=600, help="expectimax games per level")
    parser.add_argument("--seed-start", type=int, default=5_000_000)
    parser.add_argument("--td-rounds", type=int, default=3, help="refits that bootstrap on the previous network")
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    games: dict[int, list[dict[str, Any]]] = {level: [] for level in LEVELS}
    todo = []
    for seed in range(args.seed_start, args.seed_start + args.games):  # levels interleaved
        for level in LEVELS:
            path = GAMES_DIR / f"{level}_{seed}.npz"
            if path.exists():  # games are kept, so an interrupted run resumes
                with np.load(path) as d:
                    games[level].append({k: d[k] for k in d.files} | {"level": level})
            else:
                todo.append((level, seed))
    started = time.time()
    with Pool(args.workers) as pool:
        for k, game in enumerate(pool.imap_unordered(play, todo, chunksize=1)):
            np.savez_compressed(GAMES_DIR / f"{game['level']}_{game['seed']}.npz",
                                **{key: game[key] for key in ("boards", "xs", "secs", "ends")},
                                norm=game["norm"], cleared=game["cleared"])
            games[game["level"]].append(game)
            if (k + 1) % 60 == 0:
                print(f"  {k + 1}/{len(todo)} games, {(time.time() - started) / 60:.1f} min", flush=True)
    for level in LEVELS:
        norms = [float(g["norm"]) for g in games[level]]
        print(f"level {level}: {len(norms)} games, expectimax {np.mean(norms):.2f} points/s", flush=True)

    nets = []
    for level in LEVELS:
        geo, health = level_setup(level)
        layout = layout_of(geo, health)
        rows = [g for g in games[level] if bool(g["cleared"])]
        net = LevelNet(layout, [])
        columns = net.columns(geo)
        feats = [net.features(g["boards"].astype(np.float64), g["xs"], columns) for g in rows]
        all_feats = np.concatenate(feats)
        weights = None
        for round_ in range(args.td_rounds + 1):
            values = None
            if weights is not None:
                scored = LevelNet(layout, weights)
                values = [scored.forward(f) for f in feats]
            targets = np.concatenate([td_targets(g, None if values is None else values[i], args.lam)
                                      for i, g in enumerate(rows)])
            print(f"level {level}, {'Monte Carlo' if round_ == 0 else f'TD round {round_}'} targets:", flush=True)
            weights = fit(all_feats, targets, weights, args.hidden, args.epochs, round_)
        nets.append(LevelNet(layout, weights))
    ValueModel(nets).save(args.out)
    print(f"saved {args.out}", flush=True)
    (DATA_DIR / "expectimax_value_training.json").write_text(json.dumps(
        {"games": args.games, "lam": args.lam, "td_rounds": args.td_rounds,
         "expectimax": {level: float(np.mean([g["norm"] for g in games[level]])) for level in LEVELS}},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
