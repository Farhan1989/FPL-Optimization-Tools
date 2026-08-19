#!/usr/bin/env python3
"""
solio_enrich.py — pull Solio's public 4-hourly data feed and turn it into an
enrichment file for scenario_generator.py.

Why: the generator INFERS team clean-sheet probability and per-player DefCon
trigger probability from residual decomposition with position priors — its
two weakest links. Solio PUBLISHES both, free, no auth, refreshed 4-hourly:

    https://fpl.solioanalytics.com/api/data/latest.json   (preferred)
    https://fpl.solioanalytics.com/api/data/latest.md     (legacy, --md)

The JSON feed carries the same figures as typed fields rather than markdown
table cells, so it is both unrounded (csProb 0.5982 where the table prints
60%) and immune to layout changes — the markdown parser's standing
fragility. The .md path is kept for offline files and as an explicit escape
hatch; there is deliberately NO silent fallback, because a quiet downgrade
would hide the JSON endpoint breaking.

What is extracted (current gameweek only — that is all the feed carries):
  - team goals for / against for ALL 20 teams: the attacking-fixtures and
    clean-sheet tables list (G For, G Against) per listed team, and each
    opponent's numbers are the mirror (ARS 2.57/0.51 vs COV  =>  COV
    0.51/2.57).
  - team clean-sheet probability, direct where listed, exp(-G_against) for
    the rest.
  - per-player DefCon trigger probability for the listed players (the exact
    cheap-defender archetypes the pipeline's differentials live on).

Output: enrich.json consumed by  scenario_generator.py --enrich enrich.json
The generator applies it to the FIRST gameweek of the horizon only, with
EV conservation (component shifts, player totals unchanged).

Usage
-----
    uv run python solio_enrich.py --out data/enrich.json          # live fetch
    uv run python solio_enrich.py --md --out data/enrich.json     # legacy feed
    uv run python solio_enrich.py --from-file page.json --out ... # offline
    uv run python solio_enrich.py --from-file page.md --out ...   # offline
"""
# ruff: noqa: PLR2004, PLC0415, PLR0912

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

URL_JSON = "https://fpl.solioanalytics.com/api/data/latest.json"
URL_MD = "https://fpl.solioanalytics.com/api/data/latest.md"

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


def team_code(name: str | None) -> str | None:
    """Solio mixes conventions: full names in `team`, 3-letter codes in
    `opponent`. Accept either; return None for anything unrecognised so an
    unmapped side is dropped rather than silently polluting the team map."""
    if not name:
        return None
    if len(name) == 3 and name.isupper():
        return name
    return TEAM_CODE.get(name)


def fill_cs(teams: dict[str, dict]) -> None:
    """Teams that appear only as someone else's opponent have goals but no
    published CS; Poisson(G against) is the same fallback the markdown path
    has always used."""
    for t in teams.values():
        if "cs" not in t and "ga" in t:
            t["cs"] = round(math.exp(-t["ga"]), 4)


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

    fill_cs(teams)

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

    # the page emits ISO-8601 ("Generated: 2026-08-19T23:29:26.663Z"); an
    # older "... UTC" pattern silently captured nothing
    gen_m = re.search(r"Generated:?\s*(\S+)", text)
    return {"gameweek": gw, "generated": gen_m.group(1).strip() if gen_m else None, "teams": teams, "defcon": defcon}


def parse_feed(d: dict) -> dict:
    """Parse the JSON feed into the same contract parse_page() emits."""
    teams: dict[str, dict] = {}

    # Both tables carry the identical per-team fixture record and each is a
    # top-10; taking the union widens coverage before opponent mirroring.
    for section in ("bestAttackingFixtures", "bestCleanSheets"):
        for row in d.get(section) or []:
            code = team_code(row.get("team"))
            gf, ga = row.get("prGoalsFor"), row.get("prGoalsAgainst")
            if code is None or gf is None or ga is None:
                continue
            gf, ga = round(float(gf), 4), round(float(ga), 4)
            entry = teams.setdefault(code, {})
            entry["gf"], entry["ga"] = gf, ga
            if row.get("csProb") is not None:
                entry["cs"] = round(float(row["csProb"]), 4)
            for fx in row.get("fixtures") or []:  # mirror onto the opponent
                opp = team_code(fx.get("opponent"))
                if opp:
                    mirrored = teams.setdefault(opp, {})
                    mirrored.setdefault("gf", ga)
                    mirrored.setdefault("ga", gf)

    fill_cs(teams)

    defcon = {}
    for row in d.get("topDefCon") or []:
        code, name, p = team_code(row.get("team")), row.get("name"), row.get("prDefConProb")
        if code and name and p is not None:
            defcon[f"{name}|{code}"] = round(float(p), 4)

    gw = d.get("gameweek")
    return {
        "gameweek": int(gw) if gw is not None else None,
        "generated": d.get("generatedAt"),
        "teams": teams,
        "defcon": defcon,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Solio public-endpoint enrichment.")
    ap.add_argument("--out", type=Path, default=Path("data/enrich.json"))
    ap.add_argument("--from-file", type=Path, default=None, help="parse a saved copy instead of fetching")
    ap.add_argument("--md", action="store_true", help="use the legacy markdown feed instead of JSON")
    args = ap.parse_args()

    if args.from_file:
        text = args.from_file.read_text(encoding="utf-8")
        use_md = args.md or args.from_file.suffix.lower() == ".md"
    else:
        import requests

        use_md = args.md
        r = requests.get(URL_MD if use_md else URL_JSON, timeout=30, headers={"User-Agent": "fpl-pipeline/1.0"})
        r.raise_for_status()
        text = r.text

    if use_md:
        data = parse_page(text)
    else:
        try:
            data = parse_feed(json.loads(text))
        except json.JSONDecodeError as exc:
            print(f"[enrich] JSON feed did not parse ({exc}); retry with --md — NOT writing output", file=sys.stderr)
            return 1

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
        f"[enrich] {'md' if use_md else 'json'} feed, GW{data['gameweek']} ({data['generated']}): "
        f"{len(data['teams'])} teams ({n_cs} with CS), "
        f"{len(data['defcon'])} DefCon players -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
