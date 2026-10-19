"""Optional submission of finished-game results to a remote highscore service via HTTP PUT."""

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any

TOKEN_ENV = "ARKANOID_HS_TOKEN"
URL_ENV = "ARKANOID_HS_URL"
# Leaderboard endpoint (the aigf-dashboard `/api/highscores`); the default is a placeholder (.invalid never resolves).
HIGHSCORE_URL = os.environ.get(URL_ENV, "https://leaderboard.example.invalid/api/highscores")


class RemoteScoreSubmitter:
    """PUTs a JSON score record (id, name, score, avg_norm_score, levels_cleared, ...) off-thread."""

    def __init__(self, url: str, token: str | None = None, timeout: float = 5.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid highscore URL (must be http/https): {url!r}")
        self.url = url
        self.token = token if token is not None else os.environ.get(TOKEN_ENV)
        self.timeout = timeout

    def build_request(self, record: dict[str, Any]) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return urllib.request.Request(
            self.url, data=json.dumps(record).encode("utf-8"), headers=headers, method="PUT"
        )

    def send(self, record: dict[str, Any]) -> bool:
        """Blocking PUT. Returns True on a 2xx response; failures are logged, never raised."""
        try:
            with urllib.request.urlopen(self.build_request(record), timeout=self.timeout) as resp:  # noqa: S310
                ok = 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError) as e:
            logging.warning("Highscore upload for %s failed: %s", record.get("name"), e)
            return False
        if not ok:
            logging.warning("Highscore upload for %s rejected: HTTP %s", record.get("name"), resp.status)
        return ok

    def submit(self, record: dict[str, Any]) -> None:
        """Fire-and-forget send on a daemon thread."""
        threading.Thread(target=self.send, args=(record,), daemon=True).start()
