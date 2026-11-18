"""Tests package for Arkanoid game.

A git-ignored `.env` beside `pyproject.toml` (lines `NAME=value`) fills the environment for the live dashboard tests;
variables that are already set win. The template has no `.env`, so there the live tests are skipped.
"""

import os
from pathlib import Path

_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.is_file():
    for _line in _ENV.read_text(encoding="utf-8").splitlines():
        _name, _sep, _value = _line.partition("=")
        if _sep and not _name.lstrip().startswith("#"):
            os.environ.setdefault(_name.strip(), _value.strip())
