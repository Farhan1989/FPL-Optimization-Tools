"""
Live FPL API reads for the console — bounded, cached, and always optional.

Why this exists
---------------
The archive snapshot is captured *before* a deadline, which is correct for
solving but means that once the deadline passes its `picks_gw*.json` is the
*previous* gameweek's squad. Offering that as "your team" is how a solve gets
seeded with a squad the user no longer owns.

So the console asks the FPL API two questions on page load:
  1. which gameweek is live right now?  (bootstrap-static)
  2. what is the locked squad for it?   (entry/{id}/event/{gw}/picks/)

Rules this module is built around
---------------------------------
* **Never block the console.** `utils.cached_request` at the repo root calls
  `requests.get(url)` with no timeout (a blackholed host hung for 75s in
  testing) and is upstream, so it is not used here. Every request made here
  carries an explicit connect/read timeout and the whole call runs under a
  wall-clock budget; the caller adds a hard `asyncio.wait_for` on top.
* **Failure is data, not an exception.** Every path returns the same dict.
  Offline, DNS-poisoned, rate-limited and 404 all come back as a status the
  caller can label honestly, and the archived squad is offered regardless.
* **Before a deadline the picks endpoint 404s.** That is the *normal* case
  when solving on time, not an error, and is reported as `no_picks`.

Returned shape (always these keys)::

    {"status": "ok" | "no_picks" | "unavailable" | "no_team_id" | "disabled",
     "current_gw": int | None,   # the newest gameweek that has locked
     "gw": int | None,           # gameweek the picks below belong to
     "ids": [int, ...],          # 15 element ids, or []
     "team_id": int | None,
     "detail": str,              # one short human-readable line
     "elapsed": float}           # seconds spent on the network
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:  # requests is a repo dependency, but the console must not die without it
    import requests
except ImportError:  # pragma: no cover - exercised only in a broken env
    requests = None  # type: ignore[assignment]

# Overridable so the offline path can be exercised against a blackhole host.
API_BASE = os.environ.get("FPL_API_BASE", "https://fantasy.premierleague.com/api").rstrip("/")

# Per-request ceilings. Deliberately tight: this runs on page load.
CONNECT_TIMEOUT = float(os.environ.get("FPL_API_CONNECT_TIMEOUT", "2.5"))
READ_TIMEOUT = float(os.environ.get("FPL_API_READ_TIMEOUT", "3.5"))
# Whole-call ceiling across both requests.
BUDGET = float(os.environ.get("FPL_API_BUDGET", "6.0"))

# A page reload should not re-pay for the network. Successes are worth holding
# longer than failures — a failure is usually transient and a retry is cheap.
TTL_OK = float(os.environ.get("FPL_API_TTL", "120"))
TTL_FAIL = float(os.environ.get("FPL_API_TTL_FAIL", "20"))

SQUAD_SIZE = 15

_CACHE: Dict[Any, Any] = {}
_CACHE_LOCK = threading.Lock()


def _result(status: str, detail: str, **kw: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": status,
        "detail": detail,
        "current_gw": None,
        "gw": None,
        "ids": [],
        "team_id": None,
        "elapsed": 0.0,
    }
    out.update(kw)
    return out


def _parse_deadline(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def live_gameweek(events: List[Dict[str, Any]], now: Optional[datetime] = None) -> Optional[int]:
    """The newest gameweek whose squads are locked.

    `is_current` is the API's own answer and is trusted first, but it lags at
    the rollover — between a deadline passing and the first kickoff the flag
    can still point at the previous gameweek while the new squad is already
    locked. Deadlines settle that, so take whichever is further ahead."""
    now = now or datetime.now(timezone.utc)
    flagged = [int(e["id"]) for e in events if e.get("is_current") and e.get("id") is not None]
    passed = [int(e["id"]) for e in events if e.get("id") is not None and (d := _parse_deadline(e.get("deadline_time"))) is not None and d <= now]
    candidates = flagged + passed
    return max(candidates) if candidates else None


def _get_json(url: str, budget_left: float) -> Any:
    """GET with explicit timeouts. Returns the decoded body, the int 404, or None."""
    if requests is None:
        return None
    timeout = (min(CONNECT_TIMEOUT, max(0.5, budget_left)), max(0.5, min(READ_TIMEOUT, budget_left)))
    try:
        res = requests.get(url, timeout=timeout, headers={"User-Agent": "fpl-solver-console/1.0"})
    except Exception:  # noqa: BLE001 - connection, DNS, TLS, timeout: all the same answer here
        return None
    if res.status_code == 404:
        return 404
    if res.status_code != 200:
        return None
    try:
        return res.json()
    except ValueError:
        return None


def _fetch(team_id: int, gw: Optional[int], now: Optional[datetime]) -> Dict[str, Any]:
    started = time.monotonic()

    def left() -> float:
        return BUDGET - (time.monotonic() - started)

    if requests is None:
        return _result("unavailable", "the `requests` package is not installed", team_id=team_id)

    boot = _get_json(f"{API_BASE}/bootstrap-static/", left())
    if not isinstance(boot, dict) or not isinstance(boot.get("events"), list):
        return _result(
            "unavailable",
            "could not reach the FPL API",
            team_id=team_id,
            elapsed=round(time.monotonic() - started, 3),
        )

    current = live_gameweek(boot["events"], now=now)
    target = gw if gw is not None else current
    if target is None:
        return _result(
            "no_picks",
            "no gameweek has locked yet this season",
            team_id=team_id,
            current_gw=current,
            elapsed=round(time.monotonic() - started, 3),
        )

    if left() <= 0:
        return _result(
            "unavailable",
            "ran out of time budget before reading the squad",
            team_id=team_id,
            current_gw=current,
            elapsed=round(time.monotonic() - started, 3),
        )

    picks = _get_json(f"{API_BASE}/entry/{team_id}/event/{target}/picks/", left())
    elapsed = round(time.monotonic() - started, 3)

    # 404 is the ordinary pre-deadline answer: the squad for an upcoming
    # gameweek is not public until it locks. Not an error, nothing to surface.
    if picks == 404:
        return _result(
            "no_picks",
            f"GW{target} squad is not published yet (still before the deadline)",
            team_id=team_id,
            current_gw=current,
            gw=target,
            elapsed=elapsed,
        )
    if not isinstance(picks, dict):
        return _result(
            "unavailable",
            f"could not read the GW{target} squad from the FPL API",
            team_id=team_id,
            current_gw=current,
            elapsed=elapsed,
        )

    ids: List[int] = []
    for pick in picks.get("picks") or []:
        try:
            ids.append(int(pick["element"]))
        except (KeyError, TypeError, ValueError):
            continue
    if len(ids) < SQUAD_SIZE:
        return _result(
            "unavailable",
            f"the FPL API returned {len(ids)} picks for GW{target}, expected {SQUAD_SIZE}",
            team_id=team_id,
            current_gw=current,
            gw=target,
            elapsed=elapsed,
        )

    return _result(
        "ok",
        f"locked GW{target} squad for team {team_id}",
        team_id=team_id,
        current_gw=current,
        gw=target,
        ids=ids[:SQUAD_SIZE],
        elapsed=elapsed,
    )


def live_squad(team_id: Optional[int], gw: Optional[int] = None, now: Optional[datetime] = None, use_cache: bool = True) -> Dict[str, Any]:
    """Live gameweek + locked squad for `team_id`. Never raises."""
    if team_id is None:
        return _result("no_team_id", "no team_id in user_settings.json")
    if os.environ.get("FPL_API_OFFLINE"):
        return _result("disabled", "live FPL lookups disabled (FPL_API_OFFLINE)", team_id=team_id)

    key = (API_BASE, int(team_id), gw)
    if use_cache:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit and hit[0] > time.monotonic():
            return dict(hit[1])

    try:
        out = _fetch(int(team_id), gw, now)
    except Exception as exc:  # noqa: BLE001 - a live lookup must never take the console down
        out = _result("unavailable", f"live lookup failed: {type(exc).__name__}", team_id=team_id)

    if use_cache:
        ttl = TTL_OK if out["status"] == "ok" else TTL_FAIL
        with _CACHE_LOCK:
            _CACHE[key] = (time.monotonic() + ttl, dict(out))
    return out


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
