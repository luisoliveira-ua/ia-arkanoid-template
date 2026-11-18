"""Measure which bricks the expectimax agent tends to leave for the end of a level.

Reads the expectimax games recorded by ``train_expectimax_value.py`` (one board per paddle
contact) and, for every map, counts in how many games each destructible brick is still standing
when only ``--late-hits`` hits are left. Those bricks make the slow tail of a level. The rates
are written to ``training_data/tail_weights.json`` keyed by brick position, for ``TailAgent``.

    python tail_weights.py --late-hits 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.train_value import LEVELS, level_setup

DATA_DIR = Path(__file__).resolve().parent / "training_data"  # this folder's data

GAMES_DIR = DATA_DIR / "expectimax_games"
DEFAULT_OUT = DATA_DIR / "tail_weights.json"


def late_rates(level: int, late_hits: int) -> list[dict[str, float]]:
    """Per destructible brick: position and share of games where it is standing at ``late_hits`` hits left."""
    geo, health = level_setup(level)
    standing = np.zeros(len(health))
    games = 0
    for path in sorted(GAMES_DIR.glob(f"{level}_*.npz")):
        with np.load(path) as d:
            boards = d["boards"].astype(np.int64)
        left = np.maximum(boards, 0).sum(axis=1)
        late = np.flatnonzero(left <= late_hits)
        if late.size == 0:
            continue
        standing += boards[late[0]] > 0
        games += 1
    return [{"x": geo.lefts[i] + geo.radius, "y": geo.tops[i] + geo.radius, "late": standing[i] / max(games, 1)}
            for i in range(len(health)) if health[i] > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--late-hits", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    maps = {str(level): late_rates(level, args.late_hits) for level in LEVELS}
    args.out.write_text(json.dumps({"late_hits": args.late_hits, "maps": maps}, indent=1), encoding="utf-8")
    for level, rows in maps.items():
        top = sorted(rows, key=lambda r: -r["late"])[:5]
        print(f"level {level}: " + ", ".join(f"({r['x']:.0f},{r['y']:.0f}) {r['late']:.2f}" for r in top))


if __name__ == "__main__":
    main()
