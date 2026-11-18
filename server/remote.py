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
DRY_RUN_ENV = "ARKANOID_HS_DRY_RUN"  # "1": the dashboard validates the upload and stores nothing
TEST_ENV = "ARKANOID_HS_TEST"  # "1": stored but marked `is_test`, so it is hidden from ranking and histogram
# Course leaderboard (aigf-dashboard `PUT /api/v1/results`, TLS ends at the DETI proxy). ARKANOID_HS_URL overrides it.
DEFAULT_HIGHSCORE_URL = "https://dashboard.av.it.pt/api/v1/results"
HIGHSCORE_URL = os.environ.get(URL_ENV, DEFAULT_HIGHSCORE_URL)
# Upload-only token of the course dashboard, shipped on purpose so every student can submit (decision 2026-11-11: the
# page is only reachable from the UA network and the token cannot read or delete anything).
# ARKANOID_HS_TOKEN overrides it.
DEFAULT_TOKEN = "a1c5e3be164d11e572c5b221547b644681271d1cf658cb34"


def display_name(name: Any) -> str | None:
    """Dashboard `player_name`: 1-32 printable ASCII characters, else None (the field is optional)."""
    text = "".join(ch for ch in str(name or "") if " " <= ch <= "~").strip()[:32]
    return text or None


def to_result(record: dict[str, Any]) -> dict[str, Any]:
    """Map the game-end record to the dashboard body (`player_id`, `scores`, `play_time`, `setting`, `details`)."""
    return {
        # no player_id: the dashboard derives it from the name (a hash), the game never handles an id
        "player_name": display_name(record.get("name")) or "PLAYER",
        "scores": {
            "levels_cleared": record.get("levels_cleared", 0),
            "avg_norm_score": record.get("avg_norm_score", 0.0),
            "highscore": record.get("score", 0),
        },
        "play_time": record.get("time_elapsed"),
        "setting": record.get("setting"),
        "details": {k: record.get(k) for k in ("level", "won", "timed_out")},
    }


class RemoteScoreSubmitter:
    """PUTs a game-end record, mapped by `to_result`, to the dashboard off-thread."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        timeout: float = 5.0,
        dry_run: bool | None = None,
        is_test: bool | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid highscore URL (must be http/https): {url!r}")
        if dry_run is None:
            dry_run = os.environ.get(DRY_RUN_ENV, "") not in ("", "0")
        if dry_run:
            url += ("&" if "?" in url else "?") + "dry_run=true"
        if is_test is None:
            is_test = os.environ.get(TEST_ENV, "") not in ("", "0")
        self.is_test = is_test
        self.url = url
        self.dry_run = dry_run
        self.token = token if token is not None else os.environ.get(TOKEN_ENV, DEFAULT_TOKEN)
        self.timeout = timeout

    def build_request(self, record: dict[str, Any]) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return urllib.request.Request(
            self.url,
            data=json.dumps({**to_result(record), **({"is_test": True} if self.is_test else {})}).encode("utf-8"),
            headers=headers,
            method="PUT",
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
