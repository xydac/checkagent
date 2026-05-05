"""Scan result persistence and delta tracking for checkagent scan.

Saves scan results to .checkagent/history/ and computes score deltas
between consecutive scans of the same target.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HISTORY_DIR = ".checkagent" + "/" + "history"
_MAX_HISTORY_PER_TARGET = 20


def _target_id(target: str) -> str:
    """Return a short stable identifier for a scan target."""
    return hashlib.sha256(target.encode()).hexdigest()[:12]


def _history_dir(base: Path, target: str) -> Path:
    return base / _HISTORY_DIR / _target_id(target)


def save_scan_result(
    target: str,
    *,
    passed: int,
    failed: int,
    errors: int,
    total: int,
    elapsed: float,
    timestamp: float | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Persist a scan result to .checkagent/history/.

    Returns the path the result was written to.
    """
    base = base_dir or Path.cwd()
    tdir = _history_dir(base, target)
    tdir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or time.time()
    score = passed / total if total > 0 else 0.0
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)

    record: dict[str, Any] = {
        "target": target,
        "timestamp": ts,
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S UTC"),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "score": round(score, 4),
            "elapsed_seconds": round(elapsed, 3),
        },
    }

    # Timestamped file (for history listing)
    ts_name = dt.strftime("%Y%m%d-%H%M%S") + ".json"
    ts_path = tdir / ts_name
    ts_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # latest.json — always points to the most recent result
    latest_path = tdir / "latest.json"
    latest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # Prune old timestamped files (keep most recent N)
    _prune_history(tdir)

    return ts_path


def load_previous_result(
    target: str,
    *,
    base_dir: Path | None = None,
    before_timestamp: float | None = None,
) -> dict[str, Any] | None:
    """Load the most recent scan result for *target*, or None if no history exists.

    If *before_timestamp* is given, returns the most recent result BEFORE
    that timestamp (used to avoid comparing a result against itself).
    """
    base = base_dir or Path.cwd()
    tdir = _history_dir(base, target)

    if before_timestamp is None:
        # Simple case: just read latest.json
        latest = tdir / "latest.json"
        if not latest.exists():
            return None
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # Walk timestamped files in reverse order to find the one before threshold
    try:
        files = sorted(
            (f for f in tdir.iterdir() if f.name != "latest.json" and f.suffix == ".json"),
            key=lambda p: p.stem,
            reverse=True,
        )
    except OSError:
        return None

    for f in files:
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("timestamp", 0) < before_timestamp:
            return record

    return None


def compute_delta(
    current_passed: int,
    current_total: int,
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Compute the delta between the current scan and a previous result."""
    prev_summary = previous.get("summary", {})
    prev_passed = prev_summary.get("passed", 0)
    prev_total = prev_summary.get("total", 0)
    prev_score = prev_summary.get("score", 0.0)

    current_score = current_passed / current_total if current_total > 0 else 0.0
    score_delta = current_score - prev_score

    prev_failed = prev_summary.get("failed", 0)

    return {
        "previous_date": previous.get("date", "?"),
        "previous_score": prev_score,
        "current_score": round(current_score, 4),
        "score_delta": round(score_delta, 4) + 0.0,
        "previous_failed": prev_failed,
        "previous_passed": prev_passed,
        "previous_total": prev_total,
    }


def format_delta_line(delta: dict[str, Any]) -> str:
    """Format a one-line delta summary for display in the scan output."""
    prev_date = delta["previous_date"]
    score_delta = delta["score_delta"]
    prev_pct = int(round(delta["previous_score"] * 100))
    _curr_pct = int(round(delta["current_score"] * 100))

    if score_delta > 0.005:
        arrow = "↑"
        change = f"+{int(round(score_delta * 100))}%"
        style = "green"
    elif score_delta < -0.005:
        arrow = "↓"
        change = f"{int(round(score_delta * 100))}%"
        style = "red"
    else:
        arrow = "→"
        change = "no change"
        style = "dim"

    return (
        f"[{style}]{arrow} {change} from last scan[/{style}]"
        f" [dim](was {prev_pct}% on {prev_date})[/dim]"
    )


def list_history(
    target: str,
    *,
    limit: int = 10,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return a list of past scan results for *target*, newest first."""
    base = base_dir or Path.cwd()
    tdir = _history_dir(base, target)

    if not tdir.exists():
        return []

    results: list[dict[str, Any]] = []
    try:
        files = sorted(
            (f for f in tdir.iterdir() if f.name != "latest.json" and f.suffix == ".json"),
            key=lambda p: p.stem,
            reverse=True,
        )
    except OSError:
        return []

    for f in files[:limit]:
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue

    return results


def _prune_history(tdir: Path) -> None:
    """Remove oldest timestamped history files beyond the retention limit."""
    try:
        files = sorted(
            (f for f in tdir.iterdir() if f.name != "latest.json" and f.suffix == ".json"),
            key=lambda p: p.stem,
        )
    except OSError:
        return

    for old in files[:-_MAX_HISTORY_PER_TARGET]:
        with contextlib.suppress(OSError):
            old.unlink()
