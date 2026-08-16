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
        for path in sorted(logs_dir.glob("*.latest.log")):
            prefix = path.name.replace(".latest.log", "")
            for plan in read_plan_stdout(path, prefix):
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
