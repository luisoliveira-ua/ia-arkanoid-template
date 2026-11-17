"""Train the learned value function (expected seconds to clear a level) by fitted policy iteration.

Games are simulated flight by flight with the exact flight model (``agents/fast_physics.py``):
the level starts with the server's launch (paddle centre, uniform +-30 degrees), every paddle
contact picks a zone, the angle is drawn from the server's triangular distribution, and the
flight runs to the next contact. That is the same game as the server's, without the 30 Hz ticks,
so thousands of games take minutes.

Iteration 0 plays the greedy ``hits - rho * seconds`` policy. Iteration k plays the one-step
lookahead policy on the network of iteration k-1 (minimize ``flight seconds + V(next board,
next contact)``) with a little exploration, then fits a new network to the Monte Carlo
remaining-time targets of those games. Every iteration is evaluated without exploration and the
best network is saved to ``training_data/value_model.npz``.

    python train_value.py --iterations 6 --games 4000 --eval-games 400
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

from agents.fast_physics import Geometry, feasible_zones, fly, launch, triangular_quantile
from agents.value_model import DEFAULT_MODEL_PATH, TARGET_SCALE, LevelNet, ValueModel, layout_of
from server.config import load_config
from server.highscores import HighScoreManager
from server.logic import Arkanoid

LEVELS = (1, 2, 3)
TIME_LIMIT = 600.0

_worker: dict[str, Any] = {}


def level_setup(level: int) -> tuple[Geometry, list[int]]:
    config = load_config()
    config.game.level_sequence = [f"level{level}.json"]
    state = Arkanoid(config, HighScoreManager(None)).get_state()
    geo = Geometry(state)
    return geo, geo.health_from(state)


def _init_worker(weights: dict[int, list[np.ndarray]] | None) -> None:
    _worker.clear()
    for level in LEVELS:
        geo, health = level_setup(level)
        net = LevelNet(layout_of(geo, health), weights[level]) if weights else None
        _worker[level] = (geo, health, net, net.columns(geo) if net else None)


def choose_zone(level: int, health: list[int], contact_x: float, rng: random.Random, samples: int,
                epsilon: float, rho: float, shift: float | None = None) -> int:
    """Zone chosen by the greedy (no network) or one-step lookahead (network) policy."""
    geo, _, net, columns = _worker[level]
    zones = feasible_zones(geo, contact_x) or (0,)
    if len(zones) == 1:
        return zones[0]
    if rng.random() < epsilon:
        return rng.choice(zones)
    u = rng.random() if shift is None else shift  # one shift for all zones: common random numbers
    outcomes = []
    for zone in zones:
        mode, half = geo.zone_angle(zone)
        for k in range(samples):
            angle = triangular_quantile((k + u) / samples, mode - half, mode + half, mode)
            outcomes.append((zone, *fly(geo, health, *launch(geo, contact_x, angle))))
    scores = dict.fromkeys(zones, 0.0)
    if net is None:
        for zone, t, hits, _, _, _ in outcomes:
            scores[zone] += hits - rho * t
        return max(zones, key=lambda z: (scores[z], z == 0))
    open_rows = [o for o in outcomes if not o[3]]
    values = np.zeros(0)
    if open_rows:
        boards = np.array([o[5] for o in open_rows], dtype=np.float64)
        xs = np.array([o[4] for o in open_rows], dtype=np.float64)
        values = net.forward(net.features(boards, xs, columns))
    k = 0
    for zone, t, _, cleared, _, _ in outcomes:
        scores[zone] += t
        if not cleared:
            scores[zone] += float(values[k])
            k += 1
    return min(zones, key=lambda z: (scores[z], z != 0))


def play(job: tuple[int, int, int, float, float, bool]) -> dict[str, Any]:
    """One level from the server's launch to the last brick; optionally records the contacts."""
    level, seed, samples, epsilon, rho, record = job
    rng = random.Random(seed)
    geo, health, _, _ = _worker[level]
    total_hits = sum(h for h in health if h > 0)
    angle = rng.uniform(-30.0, 30.0)
    a = math.radians(angle)
    x0 = (geo.width - geo.paddle_width) / 2.0 + geo.paddle_width / 2.0
    t, _, cleared, contact_x, health = fly(geo, health, x0, geo.paddle_y - geo.radius - 2.0,
                                           geo.speed * math.sin(a), -geo.speed * math.cos(a))
    elapsed = t
    boards, xs, times = [], [], []
    shift = None if record else 0.5
    while not cleared and elapsed < TIME_LIMIT:
        zone = choose_zone(level, health, contact_x, rng, samples, epsilon, rho, shift)
        if record:
            boards.append(health)
            xs.append(contact_x)
            times.append(elapsed)
        mode, half = geo.zone_angle(zone)
        angle = rng.triangular(mode - half, mode + half, mode)
        t, _, cleared, contact_x, health = fly(geo, health, *launch(geo, contact_x, angle))
        elapsed += t
    result: dict[str, Any] = {
        "level": level, "seed": seed, "seconds": elapsed, "cleared": cleared,
        "norm": 50.0 * total_hits / elapsed if cleared else 0.0,
    }
    if record:
        result["boards"] = np.array(boards, dtype=np.int8).reshape(len(boards), -1)
        result["xs"] = np.array(xs)
        result["targets"] = elapsed - np.array(times)
    return result


def fit(features: np.ndarray, targets: np.ndarray, weights: list[np.ndarray] | None, hidden: int,
        epochs: int, seed: int) -> list[np.ndarray]:
    """Mean-squared-error fit of a two-hidden-layer ReLU network with Adam (numpy only)."""
    rng = np.random.default_rng(seed)
    n_in = features.shape[1]
    if weights is None or weights[0].shape[0] != n_in:
        sizes = (n_in, hidden, hidden, 1)
        weights = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            weights += [rng.normal(0.0, math.sqrt(2.0 / a), (a, b)), np.zeros(b)]
        weights[-1][:] = float(np.mean(targets)) / TARGET_SCALE
    weights = [w.astype(np.float64) for w in weights]
    m = [np.zeros_like(w) for w in weights]
    v = [np.zeros_like(w) for w in weights]
    y = targets / TARGET_SCALE
    n = len(y)
    order = rng.permutation(n)
    split = max(1, n // 10)
    val, train = order[:split], order[split:]
    batch, step, lr = 2048, 0, 1e-3
    best, best_loss = [w.copy() for w in weights], float("inf")
    for epoch in range(epochs):
        rng.shuffle(train)
        for start in range(0, len(train), batch):
            idx = train[start:start + batch]
            x = features[idx]
            h1 = np.maximum(x @ weights[0] + weights[1], 0.0)
            h2 = np.maximum(h1 @ weights[2] + weights[3], 0.0)
            out = (h2 @ weights[4] + weights[5])[:, 0]
            g_out = (2.0 / len(idx)) * (out - y[idx])[:, None]
            g_h2 = (g_out @ weights[4].T) * (h2 > 0)
            g_h1 = (g_h2 @ weights[2].T) * (h1 > 0)
            grads = [x.T @ g_h1, g_h1.sum(0), h1.T @ g_h2, g_h2.sum(0), h2.T @ g_out, g_out.sum(0)]
            step += 1
            rate = lr * (0.3 if epoch >= epochs * 0.7 else 1.0)
            for i, g in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                weights[i] -= rate * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
        h = features[val]
        for layer in range(2):
            h = np.maximum(h @ weights[2 * layer] + weights[2 * layer + 1], 0.0)
        loss = float(np.mean(((h @ weights[4] + weights[5])[:, 0] - y[val]) ** 2))
        if loss < best_loss:
            best, best_loss = [w.copy() for w in weights], loss
    print(f"    fit: {n} samples, validation RMSE {math.sqrt(best_loss) * TARGET_SCALE:.1f} s", flush=True)
    return best


def run_games(pool_weights: dict[int, list[np.ndarray]] | None, jobs: list[Any], workers: int) -> list[dict]:
    with Pool(workers, initializer=_init_worker, initargs=(pool_weights,)) as pool:
        return pool.map(play, jobs, chunksize=4)


def summarize(results: list[dict]) -> dict[int, float]:
    out = {}
    for level in LEVELS:
        norms = [r["norm"] for r in results if r["level"] == level]
        out[level] = float(np.mean(norms))
        cleared = sum(r["cleared"] for r in results if r["level"] == level)
        print(f"    level {level}: {out[level]:6.2f} +- {np.std(norms) / math.sqrt(len(norms)):.2f} points/s "
              f"({cleared}/{len(norms)} cleared)", flush=True)
    print(f"    mean: {np.mean(list(out.values())):.2f}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--games", type=int, default=4000, help="training games per level and iteration")
    parser.add_argument("--eval-games", type=int, default=400, help="evaluation games per level")
    parser.add_argument("--samples", type=int, default=12, help="angle samples per zone while training")
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.1, help="time price of the iteration-0 greedy policy")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--resume", action="store_true", help="start from the saved model")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log", type=Path, default=Path("training_data/value_training.json"))
    args = parser.parse_args()

    setups = {level: level_setup(level) for level in LEVELS}
    layouts = {level: layout_of(geo, health) for level, (geo, health) in setups.items()}
    weights: dict[int, list[np.ndarray]] | None = None
    if args.resume:
        saved = ValueModel.load(args.out)
        if saved is not None:
            weights = {}
            for level, (geo, _) in setups.items():
                net = saved.for_geometry(geo)
                if net is not None:
                    weights[level] = net.weights
    best_mean, history = -1.0, []
    eval_seeds = range(900_000, 900_000 + args.eval_games)
    for iteration in range(args.iterations):
        started = time.time()
        print(f"iteration {iteration} ({'greedy' if weights is None else 'value lookahead'} policy)", flush=True)
        jobs = [(level, iteration * 1_000_000 + level * 100_000 + g, args.samples, args.epsilon, args.rho, True)
                for level in LEVELS for g in range(args.games)]
        results = run_games(weights, jobs, args.workers)
        print("  training games:", flush=True)
        summarize(results)
        new_weights = {}
        for level in LEVELS:
            rows = [r for r in results if r["level"] == level and r["cleared"]]
            geo, _ = setups[level]
            net = LevelNet(layouts[level], [])
            boards = np.concatenate([r["boards"] for r in rows]).astype(np.float64)
            xs = np.concatenate([r["xs"] for r in rows])
            targets = np.concatenate([r["targets"] for r in rows])
            feats = net.features(boards, xs, net.columns(geo))
            previous = weights[level] if weights else None
            new_weights[level] = fit(feats, targets, previous, args.hidden, args.epochs, iteration)
        weights = new_weights
        eval_jobs = [(level, seed, args.eval_samples, 0.0, args.rho, False) for level in LEVELS for seed in eval_seeds]
        print("  evaluation of the new value lookahead policy:", flush=True)
        scores = summarize(run_games(weights, eval_jobs, args.workers))
        mean = float(np.mean(list(scores.values())))
        history.append({"iteration": iteration, "eval": scores, "mean": mean,
                        "minutes": round((time.time() - started) / 60, 1)})
        if mean > best_mean:
            best_mean = mean
            ValueModel([LevelNet(layouts[level], weights[level]) for level in LEVELS]).save(args.out)
            print(f"  saved {args.out} (mean {mean:.2f})", flush=True)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
