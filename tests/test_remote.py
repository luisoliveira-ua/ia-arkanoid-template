import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest import mock

import server.server as server_module
from server.remote import RemoteScoreSubmitter
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
        self.assertEqual(received[0]["body"], {"name": "ann", "score": 5})
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
        server.game.agent_id = "a1"
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
                "id": "a1",
                "name": "ann",
                "score": 1000,
                "avg_norm_score": 20.0,
                "levels_cleared": 1,
                "levels_played": 1,
                "setting": "1",
                "time_elapsed": 50.0,
                "level": 1,
                "won": False,
                "timed_out": False,
            },
        )

    def test_no_listener_without_share(self) -> None:
        self.assertIsNone(ArkanoidGameServer().game.score_listener)


if __name__ == "__main__":
    unittest.main()
