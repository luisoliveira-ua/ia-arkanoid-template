"""Contract between the game server and the dashboard: what is sent, how, and what happens when it fails.

The dashboard (https://dashboard.av.it.pt, repo aigf-dashboard) accepts `PUT /api/v1/results` with a JSON record and an
optional bearer token. These tests pin the record (keys, types, meaning) for every kind of game end, so a change in the
server cannot silently break the upload. A live check against a real dashboard is skipped unless
`ARKANOID_HS_LIVE_URL` is set.
"""

import json
import os
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest import mock

from server.logic import Arkanoid
from server.remote import RemoteScoreSubmitter

RECORD_TYPES: dict[str, tuple[type, ...]] = {
    "name": (str,),
    "score": (int,),
    "avg_norm_score": (int, float),
    "levels_cleared": (int,),
    "setting": (str,),
    "time_elapsed": (int, float),
    "level": (int,),
    "won": (bool,),
    "timed_out": (bool,),
}


def finished_game(
    *, norms: list[float], sequence: list[str] | None = None, won: bool = False, timed_out: bool = False
) -> dict[str, Any]:
    """Finish a game with the given per-level points/s and return the record handed to the uploader."""
    game = Arkanoid()
    records: list[dict[str, Any]] = []
    game.score_listener = records.append
    game.high_score_manager.filepath = None
    game.player_name = "ann"
    game.score = 1234
    game.elapsed_time = game.level_elapsed = 80.0
    game.level_norms = list(norms)
    if sequence is not None:
        game.level_sequence = sequence
    game.game_won = won
    game.timed_out = timed_out
    game.record_high_score()
    assert len(records) == 1
    return records[0]


class TestRecord(unittest.TestCase):
    def assert_contract(self, rec: dict[str, Any]) -> None:
        self.assertEqual(set(rec), set(RECORD_TYPES))
        for key, types in RECORD_TYPES.items():
            self.assertIsInstance(rec[key], types, key)
            if types == (int,):
                self.assertNotIsInstance(rec[key], bool, key)
        json.dumps(rec)  # must be JSON serialisable as it is

    def test_cleared_level_record(self) -> None:
        rec = finished_game(norms=[20.0], sequence=["maps/level1.json"], won=True)
        self.assert_contract(rec)
        self.assertEqual((rec["avg_norm_score"], rec["levels_cleared"]), (20.0, 1))
        self.assertTrue(rec["won"])

    def test_nothing_cleared_scores_zero_but_is_still_sent(self) -> None:
        rec = finished_game(norms=[])
        self.assert_contract(rec)
        self.assertEqual((rec["avg_norm_score"], rec["levels_cleared"]), (0.0, 0))
        self.assertFalse(rec["won"])

    def test_uncleared_levels_count_as_zero_in_the_mean(self) -> None:
        """One level cleared at 30 points/s out of three played gives 10."""
        rec = finished_game(norms=[30.0])
        self.assertEqual(rec["levels_cleared"], 1)
        self.assertAlmostEqual(rec["avg_norm_score"], 10.0)
        self.assertEqual(rec["setting"], "1-3")

    def test_time_limit_is_flagged(self) -> None:
        rec = finished_game(norms=[5.0], timed_out=True)
        self.assert_contract(rec)
        self.assertTrue(rec["timed_out"])

    def test_setting_label_is_canonical(self) -> None:
        """Order and repeats do not change the setting the dashboard groups by."""
        a = finished_game(norms=[], sequence=["maps/level3.json", "maps/level1.json"])
        b = finished_game(norms=[], sequence=["maps/level1.json", "maps/level3.json"])
        self.assertEqual((a["setting"], b["setting"]), ("1,3", "1,3"))

    def test_record_is_sent_once(self) -> None:
        game = Arkanoid()
        sent: list[dict[str, Any]] = []
        game.score_listener = sent.append
        game.high_score_manager.filepath = None
        game.record_high_score()
        game.record_high_score()
        self.assertEqual(len(sent), 1)


class _Capture(BaseHTTPRequestHandler):
    seen: list[dict[str, Any]] = []
    status = 201

    def do_PUT(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.seen.append(
            {
                "path": self.path,
                "method": self.command,
                "type": self.headers.get("Content-Type"),
                "auth": self.headers.get("Authorization"),
                "body": json.loads(body),
            }
        )
        self.send_response(self.status)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


class TestWire(unittest.TestCase):
    def setUp(self) -> None:
        _Capture.seen = []
        _Capture.status = 201
        self.httpd = HTTPServer(("127.0.0.1", 0), _Capture)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_full_path_from_game_end_to_http_put(self) -> None:
        """A finished game becomes exactly one authenticated JSON PUT to /api/v1/results."""
        sub = RemoteScoreSubmitter(f"{self.base}/api/v1/results", token="tok")
        game = Arkanoid()
        game.high_score_manager.filepath = None

        def blocking(record: dict[str, Any]) -> None:
            sub.send(record)  # blocking variant of `submit`

        game.score_listener = blocking
        game.player_name, game.score = "ann", 10
        game.record_high_score()
        (req,) = _Capture.seen
        self.assertEqual((req["method"], req["path"]), ("PUT", "/api/v1/results"))
        self.assertEqual(req["type"], "application/json")
        self.assertEqual(req["auth"], "Bearer tok")
        self.assertEqual(req["body"]["player_name"], "ann")
        self.assertEqual(set(req["body"]), {"player_name", "scores", "play_time", "setting", "details"})
        self.assertEqual(set(req["body"]["scores"]), {"levels_cleared", "avg_norm_score", "highscore"})

    def test_server_error_is_logged_and_does_not_stop_the_game(self) -> None:
        for status in (401, 422, 500):
            _Capture.status = status
            with self.subTest(status=status), self.assertLogs(level="WARNING"):
                self.assertFalse(RemoteScoreSubmitter(f"{self.base}/api/v1/results", token="x").send({"name": "a"}))

    def test_timeout_is_bounded(self) -> None:
        """A dashboard that never answers costs the game at most the timeout, in a background thread."""
        sub = RemoteScoreSubmitter(f"{self.base}/api/v1/results", timeout=0.5)
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("slow")) as op:
            self.assertFalse(sub.send({"name": "a"}))
        self.assertEqual(op.call_args.kwargs["timeout"], 0.5)

    def test_submit_returns_immediately(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow(*_: Any, **__: Any) -> bool:
            started.set()
            release.wait(2.0)
            return True

        sub = RemoteScoreSubmitter(f"{self.base}/api/v1/results")
        with mock.patch.object(sub, "send", side_effect=slow):
            sub.submit({"name": "a"})
            self.assertTrue(started.wait(1.0))
            release.set()


@unittest.skipUnless(os.environ.get("ARKANOID_HS_LIVE_URL"), "set ARKANOID_HS_LIVE_URL to test a real dashboard")
class TestLiveDashboard(unittest.TestCase):
    """Read-only checks of a real dashboard (no upload: the dashboard has no way to remove test rows yet)."""

    def test_health_and_ranking_are_reachable(self) -> None:
        base = os.environ["ARKANOID_HS_LIVE_URL"].rstrip("/")
        with urllib.request.urlopen(f"{base}/healthz", timeout=10) as r:  # noqa: S310
            self.assertEqual(json.load(r), {"status": "ok"})
        with urllib.request.urlopen(f"{base}/api/v1/ranking", timeout=10) as r:  # noqa: S310
            self.assertEqual(r.status, 200)


@unittest.skipUnless(
    os.environ.get("ARKANOID_HS_LIVE_URL") and os.environ.get("ARKANOID_HS_TOKEN"),
    "set ARKANOID_HS_LIVE_URL and ARKANOID_HS_TOKEN to run a dry-run upload against a real dashboard",
)
class TestLiveDryRun(unittest.TestCase):
    """One dry-run upload to a real dashboard: the body is validated there and nothing is stored."""

    def test_game_end_record_is_accepted_and_not_stored(self) -> None:
        base = os.environ["ARKANOID_HS_LIVE_URL"].rstrip("/")

        def games() -> int:
            with urllib.request.urlopen(f"{base}/api/v1/stats", timeout=10) as r:  # noqa: S310
                return int(json.load(r)["games"])

        before = games()
        record = finished_game(norms=[12.5], sequence=["maps/level1.json"], won=True)
        record["name"] = "contract-test"
        sub = RemoteScoreSubmitter(f"{base}/api/v1/results", dry_run=True)
        self.assertTrue(sub.send(record))
        self.assertEqual(games(), before)


if __name__ == "__main__":
    unittest.main()
