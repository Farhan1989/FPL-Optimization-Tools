#!/usr/bin/env python3
"""
chip_planner.py — first-set chip planning (all four chips expire at GW19).

Three subcommands:

  enumerate  Run the stock solver once per chip-timing combination (parallel),
             saving each plan's full printed output to plans/<combo>.log.
             Wraps the same solve_regular pattern as run/run_parallel.py.

  score      Parse those plan logs (or any stock-solver stdout capture) and
             evaluate every plan under the scenario set:
               - win share: fraction of scenarios in which each plan is best
               - regret: mean and worst-case shortfall vs the ex-post best plan
               - E / CVaR vs the effective-ownership field (optional bootstrap)
             Chip effects applied in scoring: TC -> captain x3 that week,
             BB -> all 15 count that week. WC/FH already shape the squads in
             the plan itself. Chips are read from the combo manifest written
             by `enumerate` (not parsed from text - robust by construction).

  tc-table   Per-gameweek captaincy TAIL metrics from scenarios: P(haul>=12),
             P(>=15), upside-CVaR. TC is an option on the right tail; ranking
             TC weeks by mean is close to meaningless.

Why win-share/regret instead of a ranked xPts column: candidate plans in this
pipeline separate by fractions of a point, inside model noise. "BB1 wins 41%,
BB2 wins 37%" is the honest answer; a ranking implies a difference that may
not exist. Decide coin flips on team news, not decimals.

Community-consensus candidates can be injected directly, e.g.:
  --candidates "bb:1,fh:3,wc:7" "bb:2,wc:4,fh:8"

Usage
-----
  uv run python chip_planner.py enumerate --config data/user_settings.json \
      --bb 1 2 3 --fh 3 4 8 --wc 4 7 --candidates "bb:1,fh:3,wc:7"

  uv run python chip_planner.py score --plans plans/ \
      --scenario-dir scenarios/ --bootstrap <ARCHIVE>/bootstrap_slim.json

  uv run python chip_planner.py tc-table --scenario-dir scenarios/
"""
# ruff: noqa: PLR2004, PLC0415, N806, PLR0913, PLR0917, PLW2901

from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_SET_DEADLINE_GW = 19  # chips unused before the GW19 deadline are lost
HAUL_THRESHOLDS = (12, 15)
CHIPS = ("wc", "bb", "fh", "tc")


# ---------------------------------------------------------------- enumerate


def combo_name(combo: dict) -> str:
    parts = [f"{c}{w}" for c, w in sorted(combo.items()) if w]
    return "_".join(parts) if parts else "nochip"


def parse_candidate(text: str) -> dict:
    out = {}
    for part in text.split(","):
        c, w = part.strip().split(":")
        if c not in CHIPS:
            raise SystemExit(f"unknown chip '{c}' in --candidates")
        out[c] = int(w)
    return out


def cmd_enumerate(args) -> int:
    import os
    import sys
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import redirect_stdout

    run_dir = Path(__file__).resolve().parent / "run"
    sys.path.insert(0, str(run_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.chdir(run_dir)
    from solve import solve_regular

    grids = {"bb": [None, *args.bb], "fh": [None, *args.fh], "wc": [None, *args.wc], "tc": [None, *args.tc]}
    combos = []
    for vals in itertools.product(*grids.values()):
        combo = dict(zip(grids.keys(), vals, strict=True))
        used = [w for w in combo.values() if w]
        if len(used) == len(set(used)):  # one chip per GW
            combos.append(combo)
    for cand in args.candidates or []:
        c = dict.fromkeys(CHIPS) | parse_candidate(cand)
        if c not in combos:
            combos.append(c)

    plans_dir = Path(args.plans)
    plans_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    print(f"[chip] {len(combos)} combinations to solve")

    def one(combo):
        name = combo_name(combo)
        opts = {
            "verbose": False,
            "print_result_table": False,
            "print_decay_metrics": False,
            "print_transfer_chip_summary": False,
            "print_squads": True,
            **{f"use_{c}": ([w] if w else []) for c, w in combo.items()},
        }
        log = plans_dir / f"{name}.log"
        with log.open("w") as fh, redirect_stdout(fh):
            solve_regular(opts)
        return name, combo

    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, combo in ex.map(one, combos):
            manifest[name] = {c: w for c, w in combo.items() if w}
            print(f"  done {name}")
    (plans_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[chip] plans + manifest written to {plans_dir}/")
    return 0


# ------------------------------------------------------------------ parsing

GW_RE = re.compile(r"^\s+\*\* GW (\d+):")
PLAYER_RE = re.compile(r"([^,()]+?)\s*\(([\d.]+)(?:,\s*([CV]))?\)")


def parse_plan_log(path: Path) -> dict[int, dict]:
    """Parse the stock solver's printed plan into
    {gw: {lineup: [names], bench: [names], captain: name}}.
    Only the FIRST solution block is read."""
    plan: dict[int, dict] = {}
    gw, section = None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Solution 2"):
            break
        m = GW_RE.match(line)
        if m:
            gw = int(m.group(1))
            plan[gw] = {"lineup": [], "bench": [], "captain": None, "chip": None}
            section = None
            continue
        if gw is not None and line.strip().startswith("CHIP "):
            plan[gw]["chip"] = line.strip().split()[1].lower()
            continue
        if gw is None:
            continue
        s = line.strip()
        if s.startswith("Lineup:"):
            section = "lineup"
            continue
        if s.startswith("Bench:"):
            section = "bench"
            continue
        if s.startswith("Lineup xPts"):
            section = None
            continue
        if section:
            for name, _xp, flag in PLAYER_RE.findall(line):
                name = name.strip()
                plan[gw][section].append(name)
                if flag == "C":
                    plan[gw]["captain"] = name
    return {g: v for g, v in plan.items() if v["lineup"]}


# ------------------------------------------------------------------- scoring


def load_scenarios(scen_dir: Path):
    files = sorted(glob.glob(str(scen_dir / "scenario_*.csv")))
    if not files:
        raise SystemExit(f"no scenarios in {scen_dir}")
    first = pd.read_csv(files[0])
    gws = sorted(int(c.split("_")[0]) for c in first.columns if c.endswith("_Pts"))
    meta = first[["ID", "Name", "Pos", "Team"]].copy()
    pts = np.zeros((len(files), len(meta), len(gws)))
    for s, f in enumerate(files):
        df = pd.read_csv(f)
        if not (df.ID.values == meta.ID.values).all():
            df = df.set_index("ID").loc[meta.ID].reset_index()
        pts[s] = df[[f"{g}_Pts" for g in gws]].to_numpy()
    return meta, gws, pts


def name_index(meta: pd.DataFrame) -> dict[str, int]:
    idx = {}
    for i, nm in enumerate(meta.Name):
        idx.setdefault(str(nm), i)
    return idx


def score_plan(plan, chips, meta, gws, pts, decay):
    """Per-scenario decayed totals for one plan, chip effects applied."""
    S = pts.shape[0]
    nidx = name_index(meta)
    gpos = {g: k for k, g in enumerate(gws)}
    total = np.zeros(S)
    missing: set[str] = set()
    # manifest is authoritative; the plan's own CHIP lines are the fallback
    bb_w = chips.get("bb") or next((g for g, e in plan.items() if e.get("chip") == "bb"), None)
    tc_w = chips.get("tc") or next((g for g, e in plan.items() if e.get("chip") == "tc"), None)
    for g, entry in plan.items():
        if g not in gpos:
            continue
        d = decay ** gpos[g]
        rows = []
        for nm in entry["lineup"]:
            if nm in nidx:
                rows.append(nidx[nm])
            else:
                missing.add(nm)
        cap = nidx.get(entry["captain"]) if entry["captain"] else None
        v = pts[:, rows, gpos[g]].sum(axis=1)
        if cap is not None:
            mult = 2.0 if g == tc_w else 1.0  # lineup already counts him once
            v = v + mult * pts[:, cap, gpos[g]]
        if g == bb_w:
            brows = [nidx[nm] for nm in entry["bench"] if nm in nidx]
            v = v + pts[:, brows, gpos[g]].sum(axis=1)
        total += d * v
    return total, missing


def load_field(bootstrap, meta, pts, gws, decay):
    if not bootstrap:
        return np.zeros(pts.shape[0])
    boot = json.loads(Path(bootstrap).read_text())
    own = {e["id"]: float(e["selected_by_percent"]) / 100 for e in boot["elements"]}
    w = np.array([own.get(int(i), 0.0) for i in meta.ID])
    proj = pts.mean(axis=0).sum(axis=1)
    score = w * np.maximum(proj, 0)
    top = np.argsort(-score)[:12]
    cap = np.zeros_like(w)
    if score[top].sum() > 0:
        cap[top] = score[top] / score[top].sum()
    d = np.array([decay**i for i in range(len(gws))])
    return np.einsum("snw,n,w->s", pts, w + cap, d)


def cmd_score(args) -> int:
    plans_dir = Path(args.plans)
    man_path = plans_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    logs = sorted(plans_dir.glob("*.log"))
    if not logs:
        raise SystemExit(f"no .log plans in {plans_dir}")
    meta, gws, pts = load_scenarios(Path(args.scenario_dir))
    F = load_field(args.bootstrap, meta, pts, gws, args.decay)
    S = pts.shape[0]

    names, totals = [], []
    for log in logs:
        plan = parse_plan_log(log)
        if not plan:
            print(f"  [skip] {log.name}: no parseable plan")
            continue
        chips = manifest.get(log.stem, {})
        t, missing = score_plan(plan, chips, meta, gws, pts, args.decay)
        if missing:
            print(f"  [warn] {log.name}: {len(missing)} names not in scenarios (e.g. {sorted(missing)[:3]})")
        names.append(log.stem)
        totals.append(t - F)
    T = np.vstack(totals)  # plans x scenarios (vs field)
    best = T.max(axis=0)
    regret = best[None, :] - T
    win = (T == best[None, :]).mean(axis=1)
    k = max(1, int(0.2 * S))
    rows = []
    for i, nm in enumerate(names):
        tail = np.sort(T[i])[:k]
        rows.append(
            {
                "plan": nm,
                "E": T[i].mean(),
                "sd": T[i].std(),
                "CVaR20": tail.mean(),
                "win%": 100 * win[i],
                "meanRegret": regret[i].mean(),
                "maxRegret": regret[i].max(),
            }
        )
    df = pd.DataFrame(rows).sort_values("win%", ascending=False)
    print(f"\n[chip] {len(names)} plans x {S} scenarios ({'vs EO field' if args.bootstrap else 'raw totals'}):\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    top2 = df.head(2)
    if len(top2) == 2 and abs(top2.iloc[0]["win%"] - top2.iloc[1]["win%"]) < args.noise_band:
        print(
            f"\n[chip] VERDICT: '{top2.iloc[0].plan}' vs '{top2.iloc[1].plan}' is "
            f"inside the noise band ({args.noise_band}pp win-share). "
            f"This is a coin flip — decide on team news and fixture "
            f"certainty, not on this table."
        )
    else:
        print(f"\n[chip] VERDICT: '{df.iloc[0].plan}' separates from the field.")
    print(f"[chip] deadline guard: first-set chips expire at the GW{FIRST_SET_DEADLINE_GW} deadline (13:30 GMT, Sat 2 Jan).")
    return 0


# ------------------------------------------------------------------ tc-table


def cmd_tc(args) -> int:
    meta, gws, pts = load_scenarios(Path(args.scenario_dir))
    proj = pts.mean(axis=0)
    rows = []
    for k, g in enumerate(gws):
        cand = np.argsort(-proj[:, k])[: args.top]
        for p in cand:
            v = pts[:, p, k]
            tail = np.sort(v)[-max(1, int(0.1 * len(v))) :]
            rows.append(
                {
                    "GW": g,
                    "player": meta.Name.iloc[p],
                    "mean": v.mean(),
                    **{f"P>={t}": np.mean(v >= t) for t in HAUL_THRESHOLDS},
                    "upCVaR10": tail.mean(),
                }
            )
    df = pd.DataFrame(rows)
    print("\n[chip] Triple Captain tail table — extra points from TC are the captain's own score, so rank by tail mass, not mean:\n")
    for g in gws:
        sub = df[df.GW == g].sort_values("P>=15", ascending=False).head(3)
        line = " | ".join(
            f"{r['player']}: mean {r['mean']:.1f}, P>=12 {r['P>=12']:.0%}, P>=15 {r['P>=15']:.0%}, top10% {r['upCVaR10']:.1f}"
            for _, r in sub.iterrows()
        )
        print(f"  GW{g:2d}  {line}")
    top = df.sort_values("P>=15", ascending=False).head(5)
    print("\n  Best TC weeks by P(haul>=15):")
    print(top.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    return 0


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="First-set chip planner.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enumerate")
    e.add_argument("--bb", type=int, nargs="*", default=[])
    e.add_argument("--fh", type=int, nargs="*", default=[])
    e.add_argument("--wc", type=int, nargs="*", default=[])
    e.add_argument("--tc", type=int, nargs="*", default=[])
    e.add_argument("--candidates", nargs="*", default=[], help='e.g. "bb:1,fh:3,wc:7"')
    e.add_argument("--plans", default="plans")
    e.add_argument("--workers", type=int, default=None)

    s = sub.add_parser("score")
    s.add_argument("--plans", default="plans")
    s.add_argument("--scenario-dir", required=True)
    s.add_argument("--bootstrap", default=None)
    s.add_argument("--decay", type=float, default=0.87)
    s.add_argument("--noise-band", type=float, default=8.0, help="win-share gap (pp) below which top-2 is a coin flip")

    t = sub.add_parser("tc-table")
    t.add_argument("--scenario-dir", required=True)
    t.add_argument("--top", type=int, default=5)

    args = ap.parse_args()
    return {"enumerate": cmd_enumerate, "score": cmd_score, "tc-table": cmd_tc}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
