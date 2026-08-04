#!/usr/bin/env python3
"""
validate_sources.py — pre-solve sanity checks on projection CSVs.

Catches the failure modes actually found in the 25/26 data:

  1. DECAY baked into a source. Decay belongs in `decay_base` only. A source
     that arrives pre-decayed silently reweights a blend by horizon distance.
  2. FIXTURE MISALIGNMENT between sources (blank/double disagreement). Blending
     across different fixture assumptions manufactures half-existing players.
  3. HALVED GAMEWEEKS. Gameweeks present in only one source can come out of
     the blend at half value if weights are not normalised.
  4. SINGLE-SOURCE PLAYERS with real projections, which get down-weighted.

Exit code 0 = pass, 1 = at least one hard failure. Designed to gate a solve.

Usage
-----
    python validate_sources.py --sources solio review
    python validate_sources.py --sources solio review --mixed mixed.csv
    python validate_sources.py --sources solio --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SINGLE_MIN, SINGLE_MAX = 60, 100  # xMins window for a single fixture
DOUBLE_XMINS = 100  # above this implies two fixtures
BLANK_XMINS = 5  # below this implies no fixture
DECAY_FLOOR = 0.990  # implied per-GW factor below this = decay
DISAGREE_LIMIT = 20  # blank-status disagreements per GW

MIN_GWS_FOR_DECAY = 4  # need this many GWs to fit a trend
MIN_PLAYERS_PER_GW = 30  # per-GW sample floor for pts/90
MIN_PAIRED_PLAYERS = 20  # sample floor for the paired test
MIN_PTS_FOR_RATIO = 0.5  # ignore near-zero projections in ratios
MIN_SOLO_SAMPLE = 10  # sample floor for halving check
HALVED_RATIO = 0.75  # below this, blend has halved a GW
INFLATED_RATIO = 1.25  # above this, blend has inflated a GW
REAL_PTS_THRESHOLD = 5.0  # horizon Pts that counts as "real"
MIN_SOURCES_TO_COMPARE = 2
UNIFORM_SPREAD = 0.02  # per-player ratio spread below this = multiplier
TEAM_BLANK_XMINS = 45  # team max xMins below this = no fixture
PLACEHOLDER_ID_FLOOR = 10000  # IDs at/above this are source-internal, not FPL
UNAVAILABLE_STATUS = {"u"}  # u = unavailable/left the league.
# NOT "s" (suspended) or "i" (injured):
# those are temporary and correctly projected.
GHOST_PTS_THRESHOLD = 1.0  # projected pts that make a ghost worth flagging
RISING_XMINS = 1.02  # ratio above this counts as rising
DECLINE_XMINS = 0.99  # median ratio below this = declining


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  FAIL  {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  WARN  {msg}")

    def ok(self, msg: str) -> None:
        print(f"  ok    {msg}")


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def gameweeks(df: pd.DataFrame) -> list[int]:
    return sorted(int(c.split("_")[0]) for c in df.columns if c.endswith("_Pts"))


# ---------------------------------------------------------------- check 1


def check_decay(name: str, df: pd.DataFrame, rep: Report) -> None:
    """Within-source test: is each player's pts/90 declining geometrically?"""
    gws = gameweeks(df)
    if len(gws) < MIN_GWS_FOR_DECAY:
        rep.warn(f"{name}: only {len(gws)} gameweeks, decay test skipped")
        return

    pts90, idx = [], []
    for i, gw in enumerate(gws):
        sub = df[df[f"{gw}_xMins"].between(SINGLE_MIN, SINGLE_MAX)]
        if len(sub) < MIN_PLAYERS_PER_GW or sub[f"{gw}_xMins"].sum() == 0:
            continue
        pts90.append(90 * sub[f"{gw}_Pts"].sum() / sub[f"{gw}_xMins"].sum())
        idx.append(i)

    if len(pts90) < MIN_GWS_FOR_DECAY:
        rep.warn(f"{name}: too few usable gameweeks for decay test")
        return

    slope, _ = np.polyfit(idx, np.log(pts90), 1)
    factor = float(np.exp(slope))

    # Paired per-player corroboration, first vs last usable gameweek.
    a_gw, b_gw = gws[idx[0]], gws[idx[-1]]
    p = df[df[f"{a_gw}_xMins"].between(SINGLE_MIN, SINGLE_MAX) & df[f"{b_gw}_xMins"].between(SINGLE_MIN, SINGLE_MAX)].copy()
    p = p[(p[f"{a_gw}_Pts"] > MIN_PTS_FOR_RATIO) & (p[f"{b_gw}_Pts"] > MIN_PTS_FOR_RATIO)]
    paired = np.nan
    if len(p) >= MIN_PAIRED_PLAYERS:
        r_a = 90 * p[f"{a_gw}_Pts"] / p[f"{a_gw}_xMins"]
        r_b = 90 * p[f"{b_gw}_Pts"] / p[f"{b_gw}_xMins"]
        paired = float((r_b / r_a).median())

    msg = f"{name}: implied per-GW factor {factor:.4f} (paired GW{b_gw}/GW{a_gw} ratio {paired:.3f}, n={len(p)})"
    if factor < DECAY_FLOOR:
        span = len(gws) - 1
        rep.fail(msg + f"  -> looks pre-decayed, ~{100 * (1 - factor**span):.0f}% over {span} GWs. Remove source decay.")
    else:
        rep.ok(msg)


def check_xmins_multiplier(name: str, df: pd.DataFrame, rep: Report) -> None:
    """
    A decay applied to BOTH Pts and xMins leaves pts/90 flat, so check_decay
    cannot see it. Genuine minutes modelling is dispersed across players and
    moves in both directions; a blanket multiplier is not.
    """
    gws = gameweeks(df)
    if len(gws) < MIN_GWS_FOR_DECAY:
        return
    a, b = gws[0], gws[-1]
    sub = df[df[f"{a}_xMins"] >= SINGLE_MIN].copy()
    if len(sub) < MIN_PAIRED_PLAYERS:
        rep.warn(f"{name}: too few players for xMins-multiplier test")
        return
    ratio = sub[f"{b}_xMins"] / sub[f"{a}_xMins"]
    spread, med = float(ratio.std()), float(ratio.median())
    rose = int((ratio > RISING_XMINS).sum())
    msg = f"{name}: xMins GW{b}/GW{a} median {med:.3f}, spread {spread:.3f}, {rose} players rising"
    if spread < UNIFORM_SPREAD and med < DECLINE_XMINS:
        rep.fail(msg + "  -> uniform multiplier on minutes, not modelling")
    elif med < DECLINE_XMINS:
        rep.ok(msg + "  (dispersed = genuine minutes modelling)")
    else:
        rep.ok(msg)


# ---------------------------------------------------------------- check 2


def team_blank_double(df: pd.DataFrame, gw: int) -> tuple[pd.Series, pd.Series]:
    """Per-player flags for whether that player's TEAM blanks/doubles this GW."""
    grp = df.groupby("Team")[f"{gw}_xMins"].max()
    return (df["Team"].map(grp) < TEAM_BLANK_XMINS, df["Team"].map(grp) > DOUBLE_XMINS)


def check_fixtures(names: list[str], dfs: list[pd.DataFrame], rep: Report) -> None:
    """
    Compare fixture STRUCTURE, not individual minutes. A blank means the whole
    team has no fixture; fringe players projected near zero are a modelling
    difference, not a calendar one.
    """
    a_name, b_name = names[0], names[1]
    a, b = dfs[0], dfs[1]
    common = sorted(set(gameweeks(a)) & set(gameweeks(b)))
    if not common:
        rep.warn("no overlapping gameweeks between sources")
        return

    keep_a = ["ID", "Team", *[f"{g}_xMins" for g in common]]
    keep_b = ["ID", "Team", *[f"{g}_xMins" for g in common]]
    m = a[keep_a].merge(b[keep_b], on="ID", suffixes=("_a", "_b"))
    a_side = m.rename(columns={"Team_a": "Team", **{f"{g}_xMins_a": f"{g}_xMins" for g in common}})
    b_side = m.rename(columns={"Team_b": "Team", **{f"{g}_xMins_b": f"{g}_xMins" for g in common}})

    bad, player_noise = [], 0
    for gw in common:
        blank_a, dbl_a = team_blank_double(a_side, gw)
        blank_b, dbl_b = team_blank_double(b_side, gw)
        n_blank = int((blank_a != blank_b).sum())
        n_dbl = int((dbl_a != dbl_b).sum())
        if n_blank or n_dbl:
            bad.append((gw, n_blank, n_dbl))
        player_noise += int(((m[f"{gw}_xMins_a"] < BLANK_XMINS) != (m[f"{gw}_xMins_b"] < BLANK_XMINS)).sum())

    if bad:
        rep.fail(f"{a_name} vs {b_name}: fixture structure differs in {len(bad)}/{len(common)} gameweeks")
        for gw, nb, nd in bad:
            print(f"          GW{gw}: {nb} players with team blank/no-blank mismatch, {nd} with double mismatch")
    else:
        rep.ok(f"{a_name} vs {b_name}: fixture structure matches across {len(common)} gameweeks (n={len(m)} players)")
    avg = player_noise / max(len(common), 1)
    print(f"  info  mean {avg:.0f} players/GW differ on near-zero minutes (squad-depth modelling, not fixtures)")


def check_availability(names: list[str], dfs: list[pd.DataFrame], boot_path: Path, rep: Report) -> None:
    """
    Cross-check projections against FPL availability status. A source that lags
    the transfer window will keep projecting departed players, or miss new
    arrivals. Uses the bootstrap snapshot the archive wrapper already captures.
    """
    with boot_path.open() as fh:
        boot = json.load(fh)
    els = {e["id"]: e for e in boot.get("elements", [])}
    teams = {t["id"]: t["short_name"] for t in boot.get("teams", [])}
    if not els:
        rep.warn("bootstrap snapshot has no elements")
        return

    shared = set(gameweeks(dfs[0]))
    for d2 in dfs[1:]:
        shared &= set(gameweeks(d2))
    cols = [f"{g}_Pts" for g in sorted(shared)]

    for name, df in zip(names, dfs, strict=True):
        ghosts = []
        for _, row in df.iterrows():
            e = els.get(int(row["ID"]))
            if e and e.get("status") in UNAVAILABLE_STATUS:
                pts = float(row[cols].sum())
                if pts > GHOST_PTS_THRESHOLD:
                    ghosts.append((e["web_name"], teams.get(e["team"], "?"), e["status"], pts))
        if ghosts:
            ghosts.sort(key=lambda x: -x[3])
            rep.warn(f"{name}: projects {len(ghosts)} player(s) the FPL API marks unavailable — source lags the transfer window")
            for n, tm, st, p in ghosts[:5]:
                print(f"          {n} ({tm}) status={st}: {p:.1f} pts")
        else:
            rep.ok(f"{name}: no projections for unavailable players")

        missing = [i for i in els if els[i].get("status") == "a" and i not in set(df.ID)]
        if missing:
            rep.warn(f"{name}: {len(missing)} available FPL player(s) absent ({', '.join(els[i]['web_name'] for i in sorted(missing)[:5])})")


# ---------------------------------------------------------------- check 3+4


def check_blend(  # noqa: PLR0912
    names: list[str], dfs: list[pd.DataFrame], mixed: pd.DataFrame, rep: Report
) -> None:
    mixed_gws = set(gameweeks(mixed))

    # 3. gameweeks present in only one source must not be halved
    for name, df in zip(names, dfs, strict=True):
        others: set[int] = set()
        for n2, d2 in zip(names, dfs, strict=True):
            if n2 != name:
                others |= set(gameweeks(d2))
        solo = sorted((set(gameweeks(df)) - others) & mixed_gws)
        if not solo:
            continue
        m = df.merge(mixed, on="ID", suffixes=("_s", "_m"))
        for gw in solo:
            sub = m[m[f"{gw}_xMins_s"].between(SINGLE_MIN, SINGLE_MAX) & (m[f"{gw}_Pts_s"] > 1.0)]
            if len(sub) < MIN_SOLO_SAMPLE:
                continue
            ratio = float((sub[f"{gw}_Pts_m"] / sub[f"{gw}_Pts_s"]).median())
            label = f"GW{gw} exists only in {name}: mixed/source ratio {ratio:.3f}"
            if ratio < HALVED_RATIO:
                rep.fail(label + "  -> HALVED, weights not normalised")
            elif ratio > INFLATED_RATIO:
                rep.fail(label + "  -> inflated")
            else:
                rep.ok(label)

    # 4. single-source players carrying real projections
    all_ids = [set(d.ID) for d in dfs]
    union = set().union(*all_ids)
    inter = set.intersection(*all_ids)
    shared_gws = set(gameweeks(dfs[0]))
    for d2 in dfs[1:]:
        shared_gws &= set(gameweeks(d2))
    for name, df in zip(names, dfs, strict=True):
        gws = sorted(shared_gws) or gameweeks(df)
        solo_ids = set(df.ID) - inter
        if not solo_ids:
            continue
        sub = df[df.ID.isin(solo_ids)].copy()
        sub["tot"] = sub[[f"{gw}_Pts" for gw in gws]].sum(axis=1)
        real = sub[sub["tot"] > REAL_PTS_THRESHOLD]
        if len(real):
            worst = real.nlargest(3, "tot")
            rep.warn(
                f"{name}: {len(real)} players projected by {name} ONLY. "
                f"Not halved (the blend normalises), but their projections "
                f"rest on one model with no cross-check"
            )
            for _, w in worst.iterrows():
                print(f"          {w['Name']} ({w['Team']}): {w['tot']:.1f} pts over the shared horizon")
        else:
            rep.ok(f"{name}: {len(solo_ids)} single-source players, all ~0 Pts (harmless)")
    print(f"  info  {len(union)} unique players, {len(inter)} in every source")
    placeholders = {i for i in union if i >= PLACEHOLDER_ID_FLOOR}
    if placeholders:
        print(f"  info  {len(placeholders)} of those use placeholder IDs >={PLACEHOLDER_ID_FLOOR} (not yet registered in the FPL API)")


# ------------------------------------------------------------------- main


def main() -> int:  # noqa: PLR0912
    ap = argparse.ArgumentParser(description="Validate FPL projection sources pre-solve.")
    ap.add_argument("--sources", nargs="+", required=True, help="source names without .csv, e.g. --sources solio review")
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--mixed", type=str, default=None, help="blended csv filename to check (e.g. mixed.csv)")
    ap.add_argument("--bootstrap", type=Path, default=None, help="bootstrap_slim.json from the archive; enables availability cross-checks")
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures too")
    args = ap.parse_args()

    names, dfs = [], []
    for s in args.sources:
        path = args.data_dir / f"{s}.csv"
        if not path.exists():
            print(f"  FAIL  missing source file: {path}")
            return 1
        names.append(s)
        dfs.append(load(path))

    rep = Report()

    print("\n[1] source decay")
    for n, d in zip(names, dfs, strict=True):
        check_decay(n, d, rep)
        check_xmins_multiplier(n, d, rep)

    print("\n[2] fixture alignment")
    if len(dfs) >= MIN_SOURCES_TO_COMPARE:
        check_fixtures(names, dfs, rep)
    else:
        rep.ok("single source, nothing to align")

    if args.bootstrap:
        print("\n[2b] availability vs FPL API")
        if args.bootstrap.exists():
            check_availability(names, dfs, args.bootstrap, rep)
        else:
            rep.warn(f"{args.bootstrap} not found, availability checks skipped")

    print("\n[3/4] blend integrity")
    if args.mixed:
        mpath = args.data_dir / args.mixed
        if mpath.exists():
            check_blend(names, dfs, load(mpath), rep)
        else:
            rep.warn(f"{mpath} not found, blend checks skipped")
    elif len(dfs) >= MIN_SOURCES_TO_COMPARE:
        check_blend(names, dfs, dfs[0].iloc[0:0], rep)
    else:
        rep.ok("single source, no blend to check")

    n_f, n_w = len(rep.failures), len(rep.warnings)
    print(f"\n{'=' * 60}")
    if n_f or (args.strict and n_w):
        print(f"VALIDATION FAILED — {n_f} failure(s), {n_w} warning(s)")
        return 1
    print(f"VALIDATION PASSED — {n_w} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
