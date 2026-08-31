"""Structured logging.

Normal CLI output goes through Rich in ``cli.py``. Logging is for the machine-readable
audit trail: data refreshes, source failures, unresolved player mappings, draft syncs,
and recommendations. It is quiet by default (WARNING) and written both to stderr and to
``data/logs/fantasy_draft.log``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line, keeping any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | int | None = None, log_dir: Path | None = None) -> None:
    """Install handlers. Idempotent — safe to call from every CLI entry point."""
    global _CONFIGURED
    resolved = level if level is not None else os.environ.get("FF_LOG_LEVEL", "WARNING")
    root = logging.getLogger("fantasy_draft")

    if _CONFIGURED:
        root.setLevel(resolved)
        return

    root.setLevel(resolved)
    root.propagate = False

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(stream)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "fantasy_draft.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(JsonFormatter())
            root.addHandler(file_handler)
        except OSError:
            # A read-only data dir must not stop the engine from running.
            root.warning("could not open log file in %s", log_dir)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``fantasy_draft`` namespace."""
    suffix = name.removeprefix("fantasy_draft.").removeprefix("fantasy_draft")
    return logging.getLogger(f"fantasy_draft.{suffix}" if suffix else "fantasy_draft")
