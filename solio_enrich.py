#!/usr/bin/env python3
"""
solio_enrich.py — pull Solio's public 4-hourly data page and turn it into an
enrichment file for scenario_generator.py.

Why: the generator INFERS team clean-sheet probability and per-player DefCon
trigger probability from residual decomposition with position priors — its
two weakest links. Solio PUBLISHES both, free, no auth, refreshed 4-hourly:

    https://fpl.solioanalytics.com/api/data/latest.md

What is extracted (current gameweek only — that is all the page carries):
  - team goals for / against for ALL 20 teams: the attacking-fixtures table
    lists (G For, G Against) per listed team, and each opponent's numbers
    are the mirror (ARS 2.62/0.50 vs COV  =>  COV 0.50/2.62).
  - team clean-sheet %, direct where listed, exp(-G_against) for the rest.
  - per-player DefCon trigger % for the listed players (the exact
    cheap-defender archetypes the pipeline's differentials live on).

Output: enrich.json consumed by  scenario_generator.py --enrich enrich.json
The generator applies it to the FIRST gameweek of the horizon only, with
EV conservation (component shifts, player totals unchanged).

Usage
-----
    uv run python solio_enrich.py --out data/enrich.json          # live fetch
    uv run python solio_enrich.py --from-file page.md --out ...   # offline
"""
# ruff: noqa: PLR2004, PLC0415, PLR0912

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

URL = "https://fpl.solioanalytics.com/api/data/latest.md"

# endpoint full names -> the short codes used in review.csv / scenario meta
TEAM_CODE = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Chelsea": "CHE",
    "Coventry": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull": "HUL",
    "Ipswich": "IPS",
    "Leeds": "LEE",
    "Liverpool": "LIV",
    "Man City": "MCI",
    "Man Utd": "MUN",
    "Newcastle": "NEW",
    "Nott'm Forest": "NFO",
    "Sunderland": "SUN",
    "Spurs": "TOT",
    "Tottenham": "TOT",
    "West Ham": "WHU",
    "Wolves": "WOL",
    "Burnley": "BUR",
    "Leicester": "LEI",
    "Southampton": "SOU",
}

GW_RE = re.compile(r"Gameweek (\d+)", re.IGNORECASE)
ROW_RE = re.compile(r"^\|\s*\d+\s*\|(.+)\|\s*$")


def cells(line: str) -> list[str]:
    m = ROW_RE.match(line.strip())
    return [c.strip() for c in m.group(1).split("|")] if m else []


def section_rows(text: str, heading_key: str) -> list[list[str]]:
    rows, active = [], False
    for line in text.splitlines():
        if line.startswith("##"):
            active = heading_key.lower() in line.lower()
            continue
        if active:
            c = cells(line)
            if c:
                rows.append(c)
    return rows


def parse_page(text: str) -> dict:
    gw_m = GW_RE.search(text)
    gw = int(gw_m.group(1)) if gw_m else None

    teams: dict[str, dict] = {}

    def team_entry(code):
        return teams.setdefault(code, {})

    # attacking fixtures table: Team | Fixtures | G For | G Against
    fixture_re = re.compile(r"(vs|@)\s+([A-Z]{3})")
    for c in section_rows(text, "attacking fixtures"):
        if len(c) < 4:
            continue
        code = TEAM_CODE.get(c[0])
        fm = fixture_re.search(c[1])
        try:
            gf, ga = float(c[2]), float(c[3])
        except ValueError:
            continue
        if code:
            team_entry(code).update({"gf": gf, "ga": ga})
        if fm:  # mirror onto the opponent
            opp = fm.group(2)
            team_entry(opp).setdefault("gf", ga)
            team_entry(opp).setdefault("ga", gf)

    # clean sheet table: Team | Fixtures | G Against | CS %
    for c in section_rows(text, "clean sheet odds"):
        if len(c) < 4:
            continue
        code = TEAM_CODE.get(c[0])
        if not code:
            continue
        try:
            team_entry(code)["cs"] = float(c[3].rstrip("%")) / 100
            team_entry(code).setdefault("ga", float(c[2]))
        except ValueError:
            continue

    # fill CS from Poisson(G against) where not listed
    import math

    for t in teams.values():
        if "cs" not in t and "ga" in t:
            t["cs"] = round(math.exp(-t["ga"]), 4)

    # DefCon table: Player | Team | Pos | Price | DefCon % | ...
    defcon = {}
    for c in section_rows(text, "projected DefCon"):
        if len(c) < 5:
            continue
        try:
            pct = float(c[4].rstrip("%")) / 100
        except ValueError:
            continue
        defcon[f"{c[0]}|{c[1]}"] = pct  # keyed name|TEAMCODE

    gen_m = re.search(r"Generated\s+([\d.:\s]+UTC)", text)
    return {"gameweek": gw, "generated": gen_m.group(1).strip() if gen_m else None, "teams": teams, "defcon": defcon}


def main() -> int:
    ap = argparse.ArgumentParser(description="Solio public-endpoint enrichment.")
    ap.add_argument("--out", type=Path, default=Path("data/enrich.json"))
    ap.add_argument("--from-file", type=Path, default=None, help="parse a saved copy instead of fetching")
    args = ap.parse_args()

    if args.from_file:
        text = args.from_file.read_text(encoding="utf-8")
    else:
        import requests

        r = requests.get(URL, timeout=30, headers={"User-Agent": "fpl-pipeline/1.0"})
        r.raise_for_status()
        text = r.text

    data = parse_page(text)
    n_cs = sum(1 for t in data["teams"].values() if "cs" in t)
    if data["gameweek"] is None or n_cs < 10 or not data["defcon"]:
        print(
            f"[enrich] page parsed thin (gw={data['gameweek']}, "
            f"teams-with-cs={n_cs}, defcon={len(data['defcon'])}) — "
            "layout may have changed; NOT writing output",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1))
    print(
        f"[enrich] GW{data['gameweek']} ({data['generated']}): "
        f"{len(data['teams'])} teams ({n_cs} with CS), "
        f"{len(data['defcon'])} DefCon players -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
