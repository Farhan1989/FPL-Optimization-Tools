"""
Adapter layer: solver output in, canonical plan shape out.

THIS IS THE FILE TO EDIT if your solver writes something the UI misreads.
Everything else in the app speaks the canonical shape below and nothing else.

Canonical plan
--------------
{
  "variant":  "cvar_a20",              # stable id, used as the table row key
  "label":    "CVaR alpha=0.2",        # what a human reads
  "source":   "results/cvar_a20.csv",
  "gameweeks": [
    {"gw": 3, "chip": null, "itb": 0.4, "ft": 1, "hits": 0, "xpts": 62.4,
     "transfers": [{"out": {...player...}, "in": {...player...}}],
     "squad": [ ...player... ]}
  ],
  "totals": {"xpts": 300.2, "hits": 4, "transfers": 6}
}

player = {"name", "team", "pos", "price", "xpts", "starting", "captain",
          "vice", "bench_order"}

Two readers are tried, in order:
  1. *.plan.json  — the canonical shape, written straight through.
  2. *.csv        — long format, one row per player per gameweek. Column names
                    are matched loosely (see COLUMN_ALIASES).

If you'd rather not fight CSV column matching, have each solver variant dump a
`<variant>.plan.json` in the canonical shape and reader 1 takes over.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Plans within this many xPts of the leader are treated as tied rather than
# ranked. Grounded in the project's noise-band finding: candidate plans
# separated by ~0.4 pts over a long horizon are inside projection noise.
NOISE_BAND_XPTS = 0.4

COLUMN_ALIASES: Dict[str, List[str]] = {
    "gw": ["week", "gw", "gameweek", "event"],
    "name": ["name", "player", "web_name", "player_name"],
    "team": ["team", "team_name", "club", "team_short"],
    "pos": ["pos", "position", "element_type", "type"],
    "price": ["buy_price", "price", "now_cost", "value", "sell_price"],
    "xpts": ["xp", "xpts", "points", "xpoints", "predicted_points", "ep"],
    "lineup": ["lineup", "starting", "start", "is_starter"],
    "bench": ["bench", "bench_order", "bench_position"],
    "captain": ["captain", "is_captain", "c"],
    "vice": ["vicecaptain", "vice_captain", "vice", "is_vice", "vc"],
    "transfer_in": ["transfer_in", "in", "bought", "buy"],
    "transfer_out": ["transfer_out", "out", "sold", "sell"],
    "chip": ["chip", "chip_played", "active_chip"],
    "itb": ["itb", "in_the_bank", "bank", "remaining_budget"],
    "ft": ["ft", "free_transfers", "free_transfer"],
    "hits": ["hits", "hit", "penalised_transfers", "point_hit"],
}

POSITION_NAMES = {
    "1": "GKP",
    "2": "DEF",
    "3": "MID",
    "4": "FWD",
    "gk": "GKP",
    "gkp": "GKP",
    "goalkeeper": "GKP",
    "def": "DEF",
    "defender": "DEF",
    "mid": "MID",
    "midfielder": "MID",
    "fwd": "FWD",
    "forward": "FWD",
}

TRUTHY = {"1", "true", "yes", "y", "t"}


# ------------------------------------------------------------ small helpers


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")


def _resolve(header: List[str]) -> Dict[str, str]:
    """Map canonical field -> the actual column name present in this CSV."""
    normed = {_norm(h): h for h in header}
    found: Dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normed:
                found[field] = normed[alias]
                break
    return found


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in TRUTHY or (_num(text, 0.0) > 0 and text not in {"", "0"})


def _position(value: Any) -> str:
    return POSITION_NAMES.get(str(value).strip().lower(), str(value).strip().upper())


def classify_source(path: Path) -> str:
    """Best-effort label so the upload list says something useful."""
    stem = path.stem.lower()
    if "review" in stem or "fplreview" in stem:
        return "FPL Review"
    if "solio" in stem:
        return "Solio"
    if "eleven" in stem:
        return "elevenify"
    if "blend" in stem or "merged" in stem:
        return "Blend"
    if path.suffix.lower() == ".json":
        return "JSON"
    return "CSV"


def count_rows(path: Path) -> Optional[int]:
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except OSError:
        return None


def _label_from(variant: str) -> str:
    pretty = variant.replace("_", " ").strip()
    pretty = re.sub(r"\ba(\d+)\b", lambda m: f"alpha={int(m.group(1)) / 100:g}", pretty)
    pretty = re.sub(r"\bcvar\b", "CVaR", pretty, flags=re.I)
    pretty = re.sub(r"\bev\b|\bxp\b", "Expected points", pretty, flags=re.I)
    return pretty[:1].upper() + pretty[1:]


# ------------------------------------------------------------ reader 1: json


def read_plan_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict) or "gameweeks" not in blob:
        return None
    variant = blob.get("variant") or path.stem.replace(".plan", "")
    plan = {
        "variant": variant,
        "label": blob.get("label") or _label_from(variant),
        "source": path.name,
        "gameweeks": blob["gameweeks"],
        "notes": blob.get("notes"),
    }
    plan["totals"] = blob.get("totals") or _totals(plan["gameweeks"])
    return plan


# ------------------------------------------------------------ reader 2: csv


def read_plan_csv(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    if not rows:
        return None

    cols = _resolve(list(rows[0].keys()))
    if "gw" not in cols or "name" not in cols:
        return None  # not a plan file — probably a projections input

    def cell(row: Dict[str, str], field: str, default: Any = "") -> Any:
        key = cols.get(field)
        return row.get(key, default) if key else default

    by_gw: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_gw[int(_num(cell(row, "gw"), 0))].append(row)

    gameweeks: List[Dict[str, Any]] = []
    for gw in sorted(k for k in by_gw if k > 0):
        rows_gw = by_gw[gw]
        squad, moves_in, moves_out = [], [], []

        for row in rows_gw:
            player = {
                "name": str(cell(row, "name")).strip(),
                "team": str(cell(row, "team")).strip(),
                "pos": _position(cell(row, "pos")),
                "price": round(_num(cell(row, "price")), 1),
                "xpts": round(_num(cell(row, "xpts")), 2),
                "starting": _flag(cell(row, "lineup", "0")),
                "captain": _flag(cell(row, "captain", "0")),
                "vice": _flag(cell(row, "vice", "0")),
                "bench_order": int(_num(cell(row, "bench", 0))) or None,
            }
            if _flag(cell(row, "transfer_in", "0")):
                moves_in.append(player)
            if _flag(cell(row, "transfer_out", "0")):
                moves_out.append(player)
            else:
                squad.append(player)

        # Gameweek-level fields are often repeated on every row, but some
        # writers only stamp them on the row that triggered them. Take the
        # first row that actually carries a value.
        def gw_field(field: str, default: Any = "") -> Any:
            for row in rows_gw:
                value = str(cell(row, field, "")).strip()
                if value and value.lower() not in {"none", "nan", "null"}:
                    return value
            return default

        chip = gw_field("chip") or None
        gameweeks.append(
            {
                "gw": gw,
                "chip": chip,
                "itb": round(_num(gw_field("itb")), 1),
                "ft": int(_num(gw_field("ft"), 1)),
                "hits": int(_num(gw_field("hits"), 0)),
                "xpts": round(sum(p["xpts"] * (2 if p["captain"] else 1) for p in squad if p["starting"]), 2),
                "transfers": _pair_transfers(moves_out, moves_in),
                "squad": squad,
            }
        )

    variant = path.stem
    return {
        "variant": variant,
        "label": _label_from(variant),
        "source": path.name,
        "gameweeks": gameweeks,
        "totals": _totals(gameweeks),
    }


def _pair_transfers(outs: List[Dict], ins: List[Dict]) -> List[Dict[str, Any]]:
    """Pair sales with buys, matching position where possible so the arrows read
    the way a manager thinks about the move."""
    remaining = list(ins)
    pairs = []
    for gone in outs:
        match = next((p for p in remaining if p["pos"] == gone["pos"]), None)
        if match is None:
            match = remaining[0] if remaining else None
        if match is not None:
            remaining.remove(match)
        pairs.append({"out": gone, "in": match})
    pairs.extend({"out": None, "in": p} for p in remaining)
    return pairs


def _totals(gameweeks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "xpts": round(sum(_num(g.get("xpts")) for g in gameweeks), 2),
        "hits": sum(int(_num(g.get("hits"), 0)) for g in gameweeks),
        "transfers": sum(len(g.get("transfers") or []) for g in gameweeks),
        "gameweeks": len(gameweeks),
    }


# ------------------------------------------------------------ collection

# ------------------------------------------------- reader 3: solver stdout

GW_RE = re.compile(r"^\s+\*\* GW (\d+):")
PLAYER_RE = re.compile(r"([^,()]+?)\s*\(([\d.]+)(?:,\s*([CV]))?\)")
SOLUTION_RE = re.compile(r"^Solution (\d+)\s*$")
MOVE_RE = re.compile(r"^\s+(Buy|Sell) (\d+) - (.+?)\s*$")
META_RE = re.compile(r"ITB=([\d.]+)->([\d.]+), FT=(\d+), PT=(\d+)")
CHIP_RE = re.compile(r"^\s+CHIP (\w+)")
XPTS_RE = re.compile(r"Lineup xPts:\s*([\d.]+)")


def read_plan_stdout(path: Path, variant_prefix: str) -> List[Dict[str, Any]]:
    """
    Parse the stock solver's printed output. One plan per 'Solution N' block,
    which is exactly what `num_iterations` produces and exactly what the
    comparison table is for: near-tied candidates, ranked honestly.

    Format verified against real solver output — the same shapes chip_planner
    parses. Anything unrecognised is skipped rather than guessed at.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if "** GW " not in text:
        return []  # not a solver log (enrich, validate, scenarios...)

    solutions: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    gw: Optional[int] = None
    section: Optional[str] = None

    def close() -> None:
        if cur and cur["gameweeks"]:
            solutions.append(cur)

    for line in text.splitlines():
        m = SOLUTION_RE.match(line.strip()) if line.strip().startswith("Solution") else None
        if m:
            close()
            n = m.group(1)
            cur = {"variant": f"{variant_prefix}_s{n}", "label": f"{variant_prefix} · solution {n}", "source": path.name, "gameweeks": []}
            gw, section = None, None
            continue
        if cur is None:
            continue

        gm = GW_RE.match(line)
        if gm:
            gw = int(gm.group(1))
            cur["gameweeks"].append(
                {"gw": gw, "chip": None, "itb": 0.0, "ft": 1, "hits": 0, "xpts": 0.0, "transfers": [], "squad": [], "_in": [], "_out": []}
            )
            section = None
            continue
        if not cur["gameweeks"]:
            continue
        entry = cur["gameweeks"][-1]

        cm = CHIP_RE.match(line)
        if cm:
            entry["chip"] = cm.group(1).upper()
            continue
        mm = META_RE.search(line)
        if mm:
            entry["itb"] = round(float(mm.group(2)), 1)
            entry["ft"] = int(mm.group(3))
            entry["hits"] = int(mm.group(4))
            continue
        mv = MOVE_RE.match(line)
        if mv:
            entry["_in" if mv.group(1) == "Buy" else "_out"].append({"name": mv.group(3).strip(), "team": "", "pos": "", "price": 0.0})
            continue
        xm = XPTS_RE.search(line)
        if xm:
            entry["xpts"] = round(float(xm.group(1)), 2)
            section = None
            continue
        if line.strip().startswith("Lineup:"):
            section = "lineup"
            continue
        if line.strip().startswith("Bench:"):
            section = "bench"
            continue
        if section:
            for name, xp, flag in PLAYER_RE.findall(line):
                entry["squad"].append(
                    {
                        "name": name.strip(),
                        "team": "",
                        "pos": "",
                        "price": 0.0,
                        "xpts": round(float(xp), 2),
                        "starting": section == "lineup",
                        "captain": flag == "C",
                        "vice": flag == "V",
                        "bench_order": None,
                    }
                )
    close()

    for plan in solutions:
        for entry in plan["gameweeks"]:
            entry["transfers"] = _pair_transfers(entry.pop("_out"), entry.pop("_in"))
        plan["totals"] = _totals(plan["gameweeks"])
    return solutions


def load_plans(results_dir: Path, logs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    plans: Dict[str, Dict[str, Any]] = {}

    for path in sorted(results_dir.glob("*.plan.json")):
        plan = read_plan_json(path)
        if plan:
            plans[plan["variant"]] = plan

    for path in sorted(results_dir.glob("*.csv")):
        if path.stem in plans:
            continue
        plan = read_plan_csv(path)
        if plan and plan["gameweeks"]:
            plans[plan["variant"]] = plan

    # Solver stdout captured by the UI. This is the path that actually works
    # today: cvar_solver and stochastic_solver print rather than writing plan
    # files, and the stock solver's Solution blocks are the variants to compare.
    if logs_dir and logs_dir.is_dir():
        # The same solve reaches us twice: once as the solver's own results
        # file and once as the stdout we captured. Same numbers, two rows,
        # which reads as two candidate plans when it is one. Keep the results
        # file (it carries team, position and price) and drop the echo.
        def signature(plan: Dict[str, Any]) -> tuple:
            t = plan.get("totals", {})
            return (round(float(t.get("xpts", 0)), 1), len(plan.get("gameweeks", [])), int(t.get("moves", 0)), int(t.get("hits", 0)))

        seen = {signature(p) for p in plans.values()}
        for path in sorted(logs_dir.glob("*.latest.log")):
            prefix = path.name.replace(".latest.log", "")
            for plan in read_plan_stdout(path, prefix):
                sig = signature(plan)
                if sig in seen:
                    continue
                seen.add(sig)
                plans.setdefault(plan["variant"], plan)

    ordered = sorted(plans.values(), key=lambda p: -p["totals"].get("xpts", 0))
    return ordered


def build_comparison(plans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score table: one row per variant, one column per gameweek, plus totals
    and a gap-to-leader that is explicitly marked when it falls inside the
    noise band."""
    if not plans:
        return {"gameweeks": [], "rows": [], "noise_band": NOISE_BAND_XPTS}

    gws = sorted({g["gw"] for p in plans for g in p["gameweeks"]})
    best = max(p["totals"].get("xpts", 0) for p in plans)

    rows = []
    for plan in plans:
        per_gw = {g["gw"]: g.get("xpts") for g in plan["gameweeks"]}
        total = plan["totals"].get("xpts", 0)
        gap = round(best - total, 2)
        rows.append(
            {
                "variant": plan["variant"],
                "label": plan["label"],
                "cells": [per_gw.get(gw) for gw in gws],
                "total": total,
                "hits": plan["totals"].get("hits", 0),
                "transfers": plan["totals"].get("transfers", 0),
                "gap": gap,
                "tied": gap <= NOISE_BAND_XPTS,
                "leader": gap == 0,
            }
        )

    return {
        "gameweeks": gws,
        "rows": rows,
        "best": best,
        "noise_band": NOISE_BAND_XPTS,
        "tied_count": sum(1 for r in rows if r["tied"]),
    }


# ------------------------------------------------- squad ID auto-discovery

STAGE1_HEADER = "Stage-1 squad"
ID_ROW_RE = re.compile(r"^\s*(\d{1,4})\s+\S")
BUY_RE = re.compile(r"^\s+Buy (\d+) - ")
GW_HEAD_RE = re.compile(r"^\s+\*\* GW (\d+):")
SQUAD_SIZE = 15


def _ids_from_stage1(text: str) -> List[int]:
    """The stage-1 table printed by stochastic_solver.py --preseason."""
    if STAGE1_HEADER not in text:
        return []
    out: List[int] = []
    for line in text.split(STAGE1_HEADER, 1)[1].splitlines():
        if line.strip().startswith("ID"):
            continue
        m = ID_ROW_RE.match(line)
        if m:
            out.append(int(m.group(1)))
        elif out:
            break  # table ended
    return out[:SQUAD_SIZE]


def _ids_from_first_gw_buys(text: str) -> List[int]:
    """Preseason EV solve: every player is a Buy in the first gameweek."""
    out: List[int] = []
    started = False
    for line in text.splitlines():
        head = GW_HEAD_RE.match(line)
        if head:
            if started:
                break
            started = True
            continue
        if started:
            m = BUY_RE.match(line)
            if m:
                out.append(int(m.group(1)))
    return out[:SQUAD_SIZE]


SELL_RE = re.compile(r"^\s+Sell (\d+) - ")


def _apply_first_gw_transfers(text: str, base: List[int]) -> List[int]:
    """Mid-season: the solve prints only the moves, not the squad.

    The post-transfer squad is what B5a needs to score, so derive it by
    applying the first gameweek's Buy/Sell to your current team rather than
    making you reconstruct it by hand."""
    ins: List[int] = []
    outs: List[int] = []
    started = False
    for line in text.splitlines():
        head = GW_HEAD_RE.match(line)
        if head:
            if started:
                break
            started = True
            continue
        if not started:
            continue
        mi, mo = BUY_RE.match(line), SELL_RE.match(line)
        if mi:
            ins.append(int(mi.group(1)))
        elif mo:
            outs.append(int(mo.group(1)))
    if not ins or len(ins) != len(outs):
        return []
    squad = [p for p in base if p not in outs] + ins
    return squad if len(squad) == SQUAD_SIZE else []


PICKS_GW_RE = re.compile(r"picks_gw0*(\d+)$", re.IGNORECASE)


def _archive_squad(archive_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Newest `picks_gw*.json` in the snapshot, by gameweek *number*.

    Sorting these by filename puts `picks_gw9` after `picks_gw10`, which would
    quietly pick the wrong file from GW10 onwards — so parse the number."""
    if not archive_dir or not archive_dir.is_dir():
        return None
    try:
        candidates = list(archive_dir.glob("picks_gw*.json"))
    except OSError:
        return None

    best: Optional[Dict[str, Any]] = None
    for path in candidates:
        match = PICKS_GW_RE.match(path.stem)
        if not match:
            continue
        gw = int(match.group(1))
        if best and gw <= best["gw"]:
            continue
        try:
            data = json.loads(path.read_text())
            ids = [int(p["element"]) for p in data.get("picks", [])]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if ids:
            best = {"gw": gw, "ids": ids[:SQUAD_SIZE], "path": path}
    return best


def _squads_for_your_team(archive_dir: Optional[Path], live: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The "your team" entries, told truthfully.

    The archive snapshot is taken *before* a deadline — correct for solving,
    but it means that once the deadline passes its newest picks file is last
    gameweek's squad. Seeding a solve with it produces advice to sell players
    already sold, and the old `Your team (GW04)` label read as provenance
    rather than as the warning it actually was.

    So: the live locked squad is offered first when the API gives us one, and
    an archived squad that predates the live gameweek is labelled STALE in the
    dropdown text itself. When the API is unreachable the archived squad is
    still offered — never removed — but is labelled unverified, because we
    genuinely do not know whether it is current."""
    live = live or {}
    live_ids: List[int] = [int(i) for i in (live.get("ids") or [])][:SQUAD_SIZE]
    live_gw = live.get("gw")
    current_gw = live.get("current_gw")
    team_id = live.get("team_id")
    owner = f"team {team_id}" if team_id else "your team"

    arch = _archive_squad(archive_dir)
    out: List[Dict[str, Any]] = []

    # Live and archive agree: one entry, and say so — a confirmed squad is
    # more useful than two identical-looking options.
    if arch and live_ids and set(live_ids) == set(arch["ids"]):
        gw = live_gw if live_gw is not None else arch["gw"]
        return [
            {
                "key": "current",
                "label": f"Your team (GW{gw:02d}) · confirmed live",
                "ids": live_ids,
                "note": f"archive snapshot for GW{arch['gw']:02d}, confirmed against the live FPL API as still your squad",
                "source": "live",
                "gw": gw,
                "stale": False,
            }
        ]

    if live_ids and live_gw is not None:
        out.append(
            {
                "key": "live",
                "label": f"Your team (GW{live_gw:02d}) · live from FPL",
                "ids": live_ids,
                "note": f"locked GW{live_gw:02d} squad for {owner}, read from the FPL API just now — includes transfers already made",
                "source": "live",
                "gw": live_gw,
                "stale": False,
            }
        )

    if not arch:
        return out

    arch_gw = arch["gw"]
    stale = current_gw is not None and arch_gw < current_gw
    if stale:
        label = f"STALE · archived GW{arch_gw:02d} squad (GW{current_gw:02d} is live)"
        note = (
            f"captured before the GW{arch_gw:02d} deadline, so it predates the live GW{current_gw:02d} gameweek. "
            "Any transfers made since are missing — solving from this can recommend selling a player you no longer own."
        )
    elif current_gw is None:
        label = f"Your team (GW{arch_gw:02d}) · unverified"
        note = "from the FPL API snapshot in this archive; the live FPL API was unreachable, so whether it is still current is unknown"
    else:
        label = f"Your team (GW{arch_gw:02d})"
        note = f"from the FPL API snapshot in this archive; GW{arch_gw:02d} is the live gameweek"

    out.append(
        {
            "key": "current",
            "label": label,
            "ids": arch["ids"],
            "note": note,
            "source": "archive",
            "gw": arch_gw,
            "stale": stale,
        }
    )
    # Freshest first, by gameweek. Live is normally newest, but if a snapshot
    # were ever taken for a later gameweek than the API reports as live, the
    # ordering should follow the evidence rather than the source.
    out.sort(key=lambda e: e.get("gw") or 0, reverse=True)
    return out


def discover_squads(
    logs_dir: Optional[Path],
    archive_dir: Optional[Path],
    live: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Squad ID lists worth offering as one-click fills.

    Typing fifteen IDs by hand before every solve is both slow and an easy
    place to make a silent mistake — a wrong ID is a valid solve of the wrong
    problem. Every source here is something an earlier step already produced.

    `live` is the (optional) result of `fpl_live.live_squad()`. This function
    does no networking itself: it stays offline, pure and testable, and simply
    labels what it is handed. `live=None` means "nothing was looked up", which
    is treated identically to "the lookup failed".
    """
    found: List[Dict[str, Any]] = _squads_for_your_team(archive_dir, live)

    if logs_dir and logs_dir.is_dir():
        for path in sorted(logs_dir.glob("*.latest.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            step = path.name.replace(".latest.log", "")

            ids = _ids_from_stage1(text)
            if len(ids) == SQUAD_SIZE:
                found.append(
                    {
                        "key": f"{step}_stage1",
                        "label": f"{step} · stage-1 squad",
                        "ids": ids,
                        "note": "committed decision under uncertainty",
                        "source": "log",
                        "stale": False,
                    }
                )
                continue

            ids = _ids_from_first_gw_buys(text)
            if len(ids) == SQUAD_SIZE:
                found.append(
                    {
                        "key": f"{step}_gw1",
                        "label": f"{step} · EV solve squad",
                        "ids": ids,
                        "note": "first-gameweek squad from the stock solver",
                        "source": "log",
                        "stale": False,
                    }
                )
                continue

            # Apply the solve's own moves to the freshest squad we have. The
            # live squad is tried first; if the solve was run against the
            # archived one its Sells won't all be present in the live squad,
            # `_apply_first_gw_transfers` returns [], and the archived squad
            # is tried instead.
            for base in (b for b in (_entry_ids(found, "live"), _entry_ids(found, "current")) if b):
                ids = _apply_first_gw_transfers(text, base)
                if ids:
                    found.append(
                        {
                            "key": f"{step}_post",
                            "label": f"{step} · after this week's move",
                            "ids": ids,
                            "note": "your team with the first-gameweek transfers applied",
                            "source": "log",
                            "stale": False,
                        }
                    )
                    break
    return found


def _entry_ids(found: List[Dict[str, Any]], key: str) -> List[int]:
    return next((f["ids"] for f in found if f["key"] == key), [])
