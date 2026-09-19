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
import shlex
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


STAGE1_IDS_RE = re.compile(r"^\[2stage\] stage-1 squad IDs:\s*([\d,\s]+?)\s*$", re.M)


def _ids_from_stage1_ids_line(text: str) -> List[int]:
    """The authoritative line stochastic_solver.py prints for BOTH arms.

    Preferred over every other reader here: the solver states its own fifteen
    as ids, so there is no base squad to reconstruct and no name to resolve.
    The other readers remain for logs written before this line existed."""
    m = STAGE1_IDS_RE.search(text)
    if not m:
        return []
    ids = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
    return ids if len(ids) == SQUAD_SIZE and len(set(ids)) == SQUAD_SIZE else []


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


# ------------------------------------------------- how far to trust an entry

# Three states, and the middle one is the whole point. "We found no problem"
# and "we checked and there is no problem" are different claims, and the bug
# this file has already been bitten by was the first rendered as the second.
# An entry we cannot check is offered — never removed — but never called fresh.
TRUST_FRESH = "fresh"  # checked against the live squad, and it agrees
TRUST_UNKNOWN = "unknown"  # not checkable: no provenance, or no live squad
TRUST_STALE = "stale"  # checked, and it is not your current squad

# Sort order for the dropdown: a trustworthy entry is never below an
# untrustworthy one, and the option a hurried click lands on is the safest.
TRUST_RANK = {TRUST_FRESH: 0, TRUST_UNKNOWN: 1, TRUST_STALE: 2}


def _worst(*states: str) -> str:
    """The least trustworthy of several verdicts. Evidence of staleness from
    any one source is evidence of staleness, whatever the others say."""
    return max(states, key=lambda s: TRUST_RANK.get(s, 1))


def _decorate(trust: str, label: str) -> str:
    """Staleness belongs in the dropdown text.

    `note` is rendered by static/app.js as a hover tooltip only, and a warning
    you have to hover to see is not a warning. The label is the one string
    that is always on screen while the option is being chosen."""
    if trust == TRUST_STALE:
        return f"STALE · {label}"
    if trust == TRUST_UNKNOWN:
        return f"{label} · unverified"
    return label


# The `#` provenance block app.py writes above captured output: `# command`
# carries the full argv, and the argv carries the `--squad` the solve was
# seeded with. Only the *leading* run of `#` lines is read, so a `#` further
# down in solver output can never be mistaken for provenance.
HEADER_ROW_RE = re.compile(r"^#\s+([a-z][a-z ]*?)\s{2,}(.+?)\s*$")
LOG_COMMENT = "#"

# Flags that mean "there is no squad to be seeded with" rather than "the seed
# was omitted". stochastic_solver.py requires one of --preseason or --squad,
# and refuses to start without either, so the distinction is never ambiguous.
SQUAD_FREE_FLAGS = {"--preseason", "--wildcard", "--wc", "--free-hit", "--freehit"}


def _parse_ids(value: str) -> List[int]:
    out: List[int] = []
    for chunk in re.split(r"[,\s]+", (value or "").strip()):
        try:
            out.append(int(chunk))
        except ValueError:
            continue
    return out


def seed_from_command(command: str) -> Dict[str, Any]:
    """What squad, if any, this argv seeded its solve with.

    Split with shlex rather than a regex so a quoted value is one token, the
    same way app.py builds the argv in the first place."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return {"seed": None, "squad_free": False}

    seed: Optional[List[int]] = None
    squad_free = False
    for i, token in enumerate(argv):
        if token in SQUAD_FREE_FLAGS:
            squad_free = True
        elif token == "--squad" and i + 1 < len(argv):
            seed = _parse_ids(argv[i + 1])
        elif token.startswith("--squad="):
            seed = _parse_ids(token.split("=", 1)[1])
    # An explicit --squad wins: a run handed a squad solved from that squad,
    # whatever else was on the command line.
    return {"seed": seed or None, "squad_free": squad_free and not seed}


def read_log_provenance(text: str) -> Dict[str, Any]:
    """The run record app.py writes above the output, or an empty one."""
    rows: Dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith(LOG_COMMENT):
            break
        match = HEADER_ROW_RE.match(line)
        if match:
            rows.setdefault(match.group(1).strip(), match.group(2))

    prov: Dict[str, Any] = {
        "header": bool(rows),
        "step": rows.get("step"),
        "command": rows.get("command"),
        "finished": rows.get("finished"),
        "seed": None,
        "squad_free": False,
    }
    if prov["command"]:
        prov.update(seed_from_command(prov["command"]))
    return prov


HORIZON_RE = re.compile(r"\bGWs?\s(\d+)\s*-\s*\d+\b")


def _solve_horizon(text: str) -> Optional[int]:
    """The first gameweek this solve was built for.

    Two shapes, because two families of tool print it: the stochastic and CVaR
    banners say `GWs 5-9`, the stock solver heads its first block `** GW 5:`."""
    match = HORIZON_RE.search(text)
    if match:
        return int(match.group(1))
    for line in text.splitlines():
        head = GW_HEAD_RE.match(line)
        if head:
            return int(head.group(1))
    return None


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
                "trust": TRUST_FRESH,
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
                "trust": TRUST_FRESH,
            }
        )

    if not arch:
        return out

    arch_gw = arch["gw"]
    stale = current_gw is not None and arch_gw < current_gw
    trust = TRUST_STALE if stale else (TRUST_FRESH if current_gw is not None else TRUST_UNKNOWN)
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
            "trust": trust,
        }
    )
    # Freshest first, by gameweek. Live is normally newest, but if a snapshot
    # were ever taken for a later gameweek than the API reports as live, the
    # ordering should follow the evidence rather than the source.
    out.sort(key=lambda e: e.get("gw") or 0, reverse=True)
    return out


def _seed_basis(prov: Dict[str, Any]) -> str:
    """How we know — or don't — what squad a log's solve started from."""
    if prov.get("squad_free"):
        return "confirmed by the squad-free flag in its recorded command line"
    if prov.get("header"):
        return "its recorded command line carries no `--squad`"
    return "this log predates run provenance headers, so this rests on the log's shape alone"


def _seed_verdict(prov: Dict[str, Any], live_ids: List[int]) -> tuple[str, str]:
    """What the recorded argv says about the squad the solve started from."""
    seed = prov.get("seed")
    if seed and live_ids:
        if set(seed) == set(live_ids):
            return TRUST_FRESH, "its recorded `--squad` matches your live squad exactly"
        gone = len([i for i in seed if i not in live_ids])
        return TRUST_STALE, f"its recorded `--squad` holds {gone} player(s) your live squad no longer contains"
    if seed:
        return TRUST_UNKNOWN, "the log records the `--squad` it used, but the live squad could not be read to compare against"
    if not prov.get("header"):
        return TRUST_UNKNOWN, "this log has no provenance header, so the squad it was seeded with was never recorded"
    return TRUST_UNKNOWN, "its recorded command line carries no `--squad`, so the seed is not recoverable from the log"


def _stage1_ids_entry(
    step: str,
    ids: List[int],
    prov: Dict[str, Any],
    horizon: Optional[int],
    current_gw: Optional[int],
    live_ids: List[int],
) -> Dict[str, Any]:
    """The solver's own statement of its stage-1 fifteen.

    Unlike `_from_scratch_entry`, a recorded `--squad` here is NOT a
    contradiction: mid-season this squad is precisely "your team after the
    move", so a seed is expected. What matters is whether that seed was the
    squad you actually hold — a solve seeded with last week's team produces a
    plan for a team you no longer own, which is the failure this whole grading
    pass exists to catch."""
    seed = prov.get("seed")
    label = f"{step} · stage-1 squad"

    if current_gw is None or horizon is None:
        trust = TRUST_UNKNOWN
        label = f"{step} · stage-1 squad · unverified"
        why = "the solve's gameweeks or the live gameweek could not be read, so whether this plan is still actionable is unknown"
    elif horizon < current_gw:
        trust = TRUST_STALE
        label = f"STALE · {step} · stage-1 squad for GW{horizon:02d}+ (GW{current_gw:02d} is live)"
        why = f"built for GW{horizon} onwards, and GW{current_gw} is live — those gameweeks have been played"
    elif seed and live_ids and set(seed) != set(live_ids):
        missing = len(set(seed) - set(live_ids))
        trust = TRUST_STALE
        label = f"STALE · {step} · stage-1 squad (seeded with an old squad)"
        why = (
            f"its recorded `--squad` holds {missing} player(s) your live squad no longer contains, so the move was planned for a team you do not own"
        )
    elif seed and not live_ids:
        trust = TRUST_UNKNOWN
        label = f"{step} · stage-1 squad · unverified"
        why = "the live squad could not be read, so whether this was seeded with your current team is unknown"
    elif not prov.get("header"):
        # `prov` is always a dict — it carries header=False — so test the flag,
        # not the dict. No header means the seed was never recorded, and an
        # unverifiable seed is UNKNOWN, never fresh: the point of this grading
        # is that "could not check" is a different claim from "fine".
        trust = TRUST_UNKNOWN
        label = f"{step} · stage-1 squad · unverified"
        why = "this log has no provenance header, so the squad it was seeded with was never recorded"
    elif prov.get("squad_free"):
        trust = TRUST_FRESH
        why = "a squad-free solve (confirmed by its recorded command) built for the live gameweek"
    else:
        trust = TRUST_FRESH
        why = "the solver stated these ids itself, seeded with your current squad, for the live gameweek"

    return {
        "key": f"{step}_stage1_ids",
        "label": label,
        "ids": ids[:SQUAD_SIZE],
        "note": f"read from {step}.latest.log — {why}",
        "trust": trust,
        "stale": trust == TRUST_STALE,
        "gw": horizon,
        "source": "stage1_ids",
    }


def _from_scratch_entry(
    step: str,
    suffix: str,
    kind: str,
    ids: List[int],
    prov: Dict[str, Any],
    horizon: Optional[int],
    current_gw: Optional[int],
) -> Dict[str, Any]:
    """A squad the solve built from nothing — preseason, or a wildcard.

    There is no seed here to be stale, and calling it a stale seed would be a
    lie about a legitimate solve. Both readers that reach this function only
    match output a squad-free solve prints: `stochastic_solver.py` prints its
    "Stage-1 squad" table under `if args.preseason` and nowhere else, and a
    first gameweek in which all fifteen players are Buys is a squad bought
    from scratch. Where a header exists the argv confirms it.

    What *can* be wrong is different, and the old label hid it. These fifteen
    were chosen for the solve's own gameweeks. Once those have been played the
    squad is a historical answer, and filling it into `--squad` seeds the next
    solve with a team that was never yours."""
    if prov.get("seed"):
        # Reader and argv disagree. Report the disagreement, don't pick one.
        trust = TRUST_UNKNOWN
        why = "the log's shape says this squad was built from scratch, but its recorded command passed a `--squad`; the two disagree"
    elif current_gw is None:
        trust = TRUST_UNKNOWN
        why = "the live gameweek could not be read, so whether its gameweeks have been played is unknown"
    elif horizon is None:
        trust = TRUST_UNKNOWN
        why = "the log does not record which gameweeks this squad was built for"
    elif horizon < current_gw:
        trust = TRUST_STALE
        why = f"it was built for GW{horizon:02d} onwards and GW{current_gw:02d} is live, so those gameweeks have been played"
    else:
        trust = TRUST_FRESH
        why = f"it was built for GW{horizon:02d} onwards, level with or ahead of the live GW{current_gw:02d}"

    label = f"{step} · {kind}"
    if trust == TRUST_STALE and horizon is not None and current_gw is not None:
        label = f"{step} · {kind} for GW{horizon:02d}+ (GW{current_gw:02d} is live)"
    note = f"a squad built from scratch, not a move from your team — no `--squad` was involved ({_seed_basis(prov)}). {why[:1].upper() + why[1:]}."
    return {
        "key": f"{step}_{suffix}",
        "label": _decorate(trust, label),
        "ids": ids,
        "note": note,
        "source": "log",
        "gw": horizon,
        "stale": trust == TRUST_STALE,
        "trust": trust,
        "squad_free": True,
    }


def _post_entry(
    step: str,
    ids: List[int],
    base: Dict[str, Any],
    prov: Dict[str, Any],
    live_ids: List[int],
    current_gw: Optional[int],
    horizon: Optional[int],
) -> Dict[str, Any]:
    """Your team with the solve's first-gameweek moves applied.

    This is the entry that used to inherit staleness in silence: apply a stale
    solve's transfers to a stale squad and the result looks like any other
    fifteen IDs. Three independent pieces of evidence, least trusting wins:

    1. the argv, when the log has a header — the recorded `--squad` against
       the live squad is a direct answer;
    2. which base the derivation succeeded against — if the solve's Sells
       could only be satisfied by the *archived* squad, then it demonstrably
       sold players the live squad no longer holds. That settles it with no
       header at all, which matters because every log written before today
       has none;
    3. the solve's own first gameweek — a move planned for a gameweek that
       has already been played can no longer be made."""
    seed_trust, seed_why = _seed_verdict(prov, live_ids)
    base_trust = base.get("trust", TRUST_UNKNOWN)
    trust = _worst(seed_trust, base_trust)
    reasons = [seed_why]
    reasons.append(
        {
            TRUST_STALE: "and its first-gameweek sells are players your live squad no longer holds, so it was solved from the archived squad",
            TRUST_UNKNOWN: "and the squad its moves were applied to could not itself be verified",
            TRUST_FRESH: "and its first-gameweek sells are all players you still own",
        }[base_trust]
    )
    if current_gw is not None and horizon is not None and horizon < current_gw:
        trust = _worst(trust, TRUST_STALE)
        reasons.append(f"and it plans a GW{horizon:02d} move while GW{current_gw:02d} is already live")

    label = f"{step} · after this week's move"
    if base_trust == TRUST_STALE and base.get("gw") is not None and current_gw is not None:
        label = f"{step} · move applied to the GW{base['gw']:02d} squad (GW{current_gw:02d} is live)"
    return {
        "key": f"{step}_post",
        "label": _decorate(trust, label),
        "ids": ids,
        "note": "your team with the first-gameweek transfers applied — " + " ".join(reasons) + ".",
        "source": "log",
        "gw": horizon,
        "stale": trust == TRUST_STALE,
        "trust": trust,
        "squad_free": False,
        "base": base.get("key"),
    }


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

    Every entry carries a `trust` of fresh / unknown / stale and is sorted by
    it, so no trustworthy entry sits below an untrustworthy one. Nothing is
    ever dropped: an entry that cannot be checked is offered and labelled
    unverified, because a squad we cannot vouch for is still often the right
    one and removing it would only send the user back to typing IDs by hand.

    `live` is the (optional) result of `fpl_live.live_squad()`. This function
    does no networking itself: it stays offline, pure and testable, and simply
    labels what it is handed. `live=None` means "nothing was looked up", which
    is treated identically to "the lookup failed".
    """
    live = live or {}
    live_ids: List[int] = [int(i) for i in (live.get("ids") or [])][:SQUAD_SIZE]
    current_gw = live.get("current_gw")

    found: List[Dict[str, Any]] = _squads_for_your_team(archive_dir, live)

    if logs_dir and logs_dir.is_dir():
        for path in sorted(logs_dir.glob("*.latest.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            step = path.name.replace(".latest.log", "")
            prov = read_log_provenance(text)
            horizon = _solve_horizon(text)

            ids = _ids_from_stage1_ids_line(text)
            if ids:
                found.append(_stage1_ids_entry(step, ids, prov, horizon, current_gw, live_ids))
                continue

            ids = _ids_from_stage1(text)
            if len(ids) == SQUAD_SIZE:
                found.append(_from_scratch_entry(step, "stage1", "stage-1 squad", ids, prov, horizon, current_gw))
                continue

            ids = _ids_from_first_gw_buys(text)
            if len(ids) == SQUAD_SIZE:
                found.append(_from_scratch_entry(step, "gw1", "EV solve squad", ids, prov, horizon, current_gw))
                continue

            # Apply the solve's own moves to the freshest squad we have. The
            # live squad is tried first; if the solve was run against the
            # archived one its Sells won't all be present in the live squad,
            # `_apply_first_gw_transfers` returns [], and the archived squad
            # is tried instead — and *which* base succeeded is itself the
            # evidence `_post_entry` grades the result on.
            for base in (b for b in (_entry(found, "live"), _entry(found, "current")) if b and b.get("ids")):
                ids = _apply_first_gw_transfers(text, base["ids"])
                if ids:
                    found.append(_post_entry(step, ids, base, prov, live_ids, current_gw, horizon))
                    break

    # Trustworthy first. A stable sort, so inside a band the order this
    # function already established — live squad, then archive, then logs by
    # filename — survives untouched.
    found.sort(key=lambda e: TRUST_RANK.get(e.get("trust", TRUST_UNKNOWN), 1))
    return found


def _entry(found: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    return next((f for f in found if f["key"] == key), None)


def _entry_ids(found: List[Dict[str, Any]], key: str) -> List[int]:
    entry = _entry(found, key)
    return entry["ids"] if entry else []
