import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest import mock

import server.server as server_module
from server.remote import DEFAULT_TOKEN, RemoteScoreSubmitter
from server.server import ArkanoidGameServer

received: list[dict[str, Any]] = []


class _Handler(BaseHTTPRequestHandler):
    def do_PUT(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        received.append({"body": json.loads(body), "auth": self.headers.get("Authorization")})
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


class TestRemoteSubmitter(unittest.TestCase):
    def setUp(self) -> None:
        received.clear()
        self.httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/scores"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_put_with_token(self) -> None:
        sub = RemoteScoreSubmitter(self.url, token="s3cret")
        self.assertEqual(sub.build_request({}).get_method(), "PUT")
        self.assertTrue(sub.send({"name": "ann", "score": 5}))
        self.assertEqual(received[0]["body"]["player_name"], "ann")
        self.assertEqual(received[0]["body"]["scores"]["highscore"], 5)
        self.assertEqual(received[0]["auth"], "Bearer s3cret")

    def test_unreachable_server_does_not_raise(self) -> None:
        self.httpd.shutdown()
        self.assertFalse(RemoteScoreSubmitter(self.url, token=None, timeout=1.0).send({"player": "x"}))

    def test_invalid_url(self) -> None:
        with self.assertRaises(ValueError):
            RemoteScoreSubmitter("ftp://nope")

    def test_game_end_uploads_avg_norm_score(self) -> None:
        with mock.patch.object(server_module, "HIGHSCORE_URL", self.url):
            server = ArkanoidGameServer(share_highscore=True)
        server.game.high_score_manager.filepath = None
        server.game.player_name = "ann"
        server.game.score = 1000
        server.game.elapsed_time = server.game.level_elapsed = 50.0
        server.game.level_norms = [20.0]  # one level cleared at 20 points/s ...
        server.game.level_sequence = server.game.level_sequence[:1]  # ... in a one-level sequence
        server.game.record_high_score()
        for _ in range(50):
            if received:
                break
            threading.Event().wait(0.1)
        self.assertEqual(
            received[0]["body"],
            {
                "player_name": "ann",
                "scores": {"levels_cleared": 1, "avg_norm_score": 20.0, "highscore": 1000},
                "play_time": 50.0,
                "setting": "1",
                "details": {"level": 1, "won": False, "timed_out": False},
            },
        )

    def test_no_listener_without_share(self) -> None:
        self.assertIsNone(ArkanoidGameServer().game.score_listener)


class TestDashboardEndpoint(unittest.TestCase):
    """The uploader targets the course dashboard unless the environment says otherwise."""

    def _reload(self, env: dict[str, str]) -> Any:
        import importlib
        import os

        import server.remote as remote

        with mock.patch.dict(os.environ, env, clear=False):
            for name in ("ARKANOID_HS_URL", "ARKANOID_HS_TOKEN"):
                if name not in env:
                    os.environ.pop(name, None)
            return importlib.reload(remote).HIGHSCORE_URL

    def tearDown(self) -> None:
        import importlib

        import server.remote as remote

        importlib.reload(remote)

    def test_default_is_the_course_dashboard(self) -> None:
        url = self._reload({})
        self.assertEqual(url, "https://dashboard.av.it.pt/api/v1/results")
        self.assertTrue(url.startswith("https://"))

    def test_env_overrides_the_default(self) -> None:
        self.assertEqual(
            self._reload({"ARKANOID_HS_URL": "http://localhost:8000/api/v1/results"}),
            "http://localhost:8000/api/v1/results",
        )

    def test_token_comes_from_the_environment_with_a_shipped_default(self) -> None:
        import os

        with mock.patch.dict(os.environ, {"ARKANOID_HS_TOKEN": "tok"}):
            sub = RemoteScoreSubmitter("https://dashboard.av.it.pt/api/v1/results")
        self.assertEqual(sub.build_request({}).get_header("Authorization"), "Bearer tok")
        with mock.patch.dict(os.environ, clear=True):
            default = RemoteScoreSubmitter("https://dashboard.av.it.pt/api/v1/results")
        self.assertEqual(default.build_request({}).get_header("Authorization"), f"Bearer {DEFAULT_TOKEN}")
        self.assertIsNone(RemoteScoreSubmitter("http://x/y", token="").build_request({}).get_header("Authorization"))

    def test_dry_run_adds_the_query_flag(self) -> None:
        url = "https://dashboard.av.it.pt/api/v1/results"
        self.assertTrue(RemoteScoreSubmitter(url, dry_run=True).url.endswith("/api/v1/results?dry_run=true"))
        self.assertTrue(RemoteScoreSubmitter(url + "?x=1", dry_run=True).url.endswith("?x=1&dry_run=true"))
        self.assertEqual(RemoteScoreSubmitter(url, dry_run=False).url, url)

    def test_dry_run_comes_from_the_environment(self) -> None:
        import os

        url = "https://dashboard.av.it.pt/api/v1/results"
        for value, expected in (("1", True), ("0", False), ("", False)):
            with mock.patch.dict(os.environ, {"ARKANOID_HS_DRY_RUN": value}):
                self.assertEqual(RemoteScoreSubmitter(url).dry_run, expected)

    def test_rejected_upload_is_reported_not_raised(self) -> None:
        """A 401 (wrong or missing token) makes `send` return False and never raises."""

        class Reject(BaseHTTPRequestHandler):
            def do_PUT(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(401)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Reject)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            sub = RemoteScoreSubmitter(f"http://127.0.0.1:{httpd.server_port}/api/v1/results", token="bad")
            self.assertFalse(sub.send({"name": "a1"}))
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()


class TestDashboardFields(unittest.TestCase):
    """`is_test` and `player_name` follow the dashboard's ResultIn rules."""

    def test_is_test_flag_and_name_limits(self) -> None:
        from server.remote import display_name

        sub = RemoteScoreSubmitter("http://x/api/v1/results", is_test=True)
        body = json.loads(sub.build_request({"name": "Zoë " + "x" * 40}).data)  # type: ignore[arg-type]
        self.assertTrue(body["is_test"])
        self.assertEqual(body["player_name"], "Zo " + "x" * 29)
        plain = json.loads(RemoteScoreSubmitter("http://x/y", is_test=False).build_request({}).data)  # type: ignore[arg-type]
        self.assertNotIn("is_test", plain)
        self.assertIsNone(display_name("é"))
