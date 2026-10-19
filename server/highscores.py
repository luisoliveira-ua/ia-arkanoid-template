"""High score manager: top-10 raw and avg_norm_score leaderboards backed by a plain CSV log of finished games.

The raw score is the sum over the levels played; `avg_norm_score` is the mean over the levels of the sequence of the
points per second of each level, a level that was not cleared counting 0.
"""

import csv
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

_FIELDS = ["date", "id", "player", "score", "time_elapsed", "avg_norm_score"]


@dataclass
class HighScoreEntry:
    """Single high score record."""

    score: int
    player: str = "PLAYER"
    agent_id: str = ""
    date: str = ""
    time_elapsed: float = 0.0
    avg_norm_score: float = 0.0


# Default arcade leaderboard honoring Taito Arkanoid (1986)
DEFAULT_SCORES: list[HighScoreEntry] = [
    HighScoreEntry(score=10000, player="TAI", date="1986-07-01", time_elapsed=100.0, avg_norm_score=100.0),
    HighScoreEntry(score=8500, player="DOH", date="1986-07-01", time_elapsed=100.0, avg_norm_score=85.0),
    HighScoreEntry(score=7000, player="ARK", date="1986-07-01", time_elapsed=100.0, avg_norm_score=70.0),
    HighScoreEntry(score=5500, player="VAU", date="1986-07-01", time_elapsed=100.0, avg_norm_score=55.0),
    HighScoreEntry(score=4000, player="RET", date="1986-07-01", time_elapsed=100.0, avg_norm_score=40.0),
    HighScoreEntry(score=3000, player="PIL", date="1986-07-01", time_elapsed=100.0, avg_norm_score=30.0),
    HighScoreEntry(score=2500, player="LAS", date="1986-07-01", time_elapsed=100.0, avg_norm_score=25.0),
    HighScoreEntry(score=2000, player="COM", date="1986-07-01", time_elapsed=100.0, avg_norm_score=20.0),
    HighScoreEntry(score=1500, player="BOT", date="1986-07-01", time_elapsed=100.0, avg_norm_score=15.0),
    HighScoreEntry(score=1000, player="NEW", date="1986-07-01", time_elapsed=100.0, avg_norm_score=10.0),
]


class HighScoreManager:
    """Top 10 raw and normalized high scores; finished games are appended to a CSV file (memory-only if None)."""

    def __init__(self, filepath: str | None = "highscores.csv") -> None:
        self.filepath = filepath
        self.raw_scores: list[HighScoreEntry] = []
        self.normalized_scores: list[HighScoreEntry] = []
        self._top_cache: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None
        self.load()

    @property
    def scores(self) -> list[HighScoreEntry]:
        """Alias to raw scores for backwards compatibility."""
        return self.raw_scores

    @scores.setter
    def scores(self, val: list[HighScoreEntry]) -> None:
        self.raw_scores = val
        self._top_cache = None

    @staticmethod
    def _read_rows(filepath: str) -> list[HighScoreEntry]:
        """Read the CSV log, skipping malformed rows."""
        loaded: list[HighScoreEntry] = []
        with open(filepath, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    loaded.append(
                        HighScoreEntry(
                            score=int(row["score"]),
                            player=row.get("player") or "PLAYER",
                            agent_id=row.get("id") or "",
                            date=row.get("date") or "",
                            time_elapsed=float(row.get("time_elapsed") or 0.0),
                            avg_norm_score=float(row.get("avg_norm_score") or row.get("normalized_score") or 0.0),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    logging.warning("Skipping malformed high score row.")
        return loaded

    def _rebuild(self, entries: list[HighScoreEntry]) -> None:
        """Rank the arcade defaults plus the logged games into the two top-10 boards."""
        pool = [*DEFAULT_SCORES, *entries]
        self.raw_scores = sorted(pool, key=lambda e: e.score, reverse=True)[:10]
        self.normalized_scores = sorted(pool, key=lambda e: e.avg_norm_score, reverse=True)[:10]
        self._top_cache = None

    def load(self) -> list[HighScoreEntry]:
        """Rebuild the leaderboards from the CSV log (arcade defaults only when there is none)."""
        entries: list[HighScoreEntry] = []
        if self.filepath is not None and os.path.exists(self.filepath):
            try:
                entries = self._read_rows(self.filepath)
            except (OSError, csv.Error, UnicodeDecodeError) as e:
                logging.warning("Ignoring unreadable high score file %s: %s", self.filepath, e)
        self._rebuild(entries)
        return self.raw_scores

    def _append(self, entry: HighScoreEntry) -> None:
        """Append one finished game to the CSV log (no-op in memory-only mode)."""
        if self.filepath is None:
            return
        new_file = not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(_FIELDS)
            writer.writerow(
                [entry.date, entry.agent_id, entry.player, entry.score, entry.time_elapsed, entry.avg_norm_score]
            )

    def get_highest_score(self) -> int:
        """Return the all-time highest raw score."""
        if self.raw_scores:
            return self.raw_scores[0].score
        return 0

    def get_highest_normalized_score(self) -> float:
        """Return the all-time highest normalized score (points/sec)."""
        if self.normalized_scores:
            return self.normalized_scores[0].avg_norm_score
        return 0.0

    def add_score(
        self,
        score: int,
        player: str = "PLAYER",
        agent_id: str = "",
        date: str | None = None,
        time_elapsed: float = 0.0,
        avg_norm_score: float | None = None,
    ) -> bool:
        """
        Add new score if it qualifies for raw and/or normalized top 10.
        Returns True if the score was added to at least one leaderboard.
        """
        if score <= 0:
            return False

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M")

        if avg_norm_score is None:
            avg_norm_score = 0.0  # no rate is derived from score / time: only cleared levels earn normalized score

        new_entry = HighScoreEntry(
            score=score,
            player=player,
            agent_id=agent_id,
            date=date,
            time_elapsed=time_elapsed,
            avg_norm_score=avg_norm_score,
        )

        self._append(new_entry)

        qualifies_raw = (len(self.raw_scores) < 10 or score > self.raw_scores[-1].score) and score > 0
        qualifies_norm = (
            (len(self.normalized_scores) < 10
            or avg_norm_score > self.normalized_scores[-1].avg_norm_score)
            and avg_norm_score > 0.0
        )

        if not qualifies_raw and not qualifies_norm:
            return False

        if qualifies_raw:
            self.raw_scores.append(new_entry)
            self.raw_scores.sort(key=lambda x: x.score, reverse=True)
            self.raw_scores = self.raw_scores[:10]

        if qualifies_norm:
            self.normalized_scores.append(new_entry)
            self.normalized_scores.sort(key=lambda x: x.avg_norm_score, reverse=True)
            self.normalized_scores = self.normalized_scores[:10]

        self._top_cache = None
        return True

    def _tops(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Decoded leaderboards, rebuilt only when scores change (state payloads read them every tick)."""
        if self._top_cache is None:
            self._top_cache = (
                [{"rank": i + 1, **asdict(e)} for i, e in enumerate(self.raw_scores[:10])],
                [{"rank": i + 1, **asdict(e)} for i, e in enumerate(self.normalized_scores[:10])],
            )
        return self._top_cache

    def get_top_scores(self) -> list[dict[str, Any]]:
        """Return decoded raw leaderboard list formatted for state payloads (read-only)."""
        return self._tops()[0]

    def get_top_normalized_scores(self) -> list[dict[str, Any]]:
        """Return decoded normalized leaderboard list formatted for state payloads (read-only)."""
        return self._tops()[1]
