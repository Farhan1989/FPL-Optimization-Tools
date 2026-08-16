#!/usr/bin/env python3
"""
stochastic_solver.py — step 4 of the roadmap.

Two-stage stochastic program over scenario files.

    Stage 1 (here-and-now): this gameweek's transfers, lineup, captain.
        ONE copy, shared by every scenario — this is non-anticipativity by
        construction: you must commit before you know which world you're in.
    Stage 2 (recourse): every later gameweek's transfers, lineups, captains
        are scenario-indexed and adapt to that scenario's outcomes.

This answers the question the deterministic solver cannot: not "which plan
has the best expected total" but "which move NOW is best given that I will
respond optimally to whatever happens next".

Within-scenario dynamics are a simplified replica of the stock solver:
free-transfer banking (cap 5), -4 hits, budget flow. Simplifications, all
documented: single-price transfers (exact preseason, approximate mid-season
where the 50% sell rule bites), no chips, no bench-order weights (a flat
epsilon keeps bench players sane). Chip weeks should be planned with the
stock solver; this tool is for the ordinary weekly decision.

Objective: (1-lam) * E[score] + lam * CVaR_alpha[score], optionally against
the effective-ownership field from a bootstrap snapshot (step 3's model).
lam=0, no bootstrap -> pure stochastic EV.

Also computes, with --vss:
    VSS = (stochastic solution's objective)
        - (objective when stage 1 is FIXED to the deterministic-on-the-mean
           solution and only recourse re-optimises)
i.e. the measured value of planning under uncertainty rather than certainty-
equivalent planning. If VSS ~ 0 on your data, the deterministic solver was
already enough — worth knowing either way.

Usage
-----
  # preseason: pick the initial 15 under uncertainty
  uv run python stochastic_solver.py --scenario-dir scenarios/ --preseason \
      --weeks 4 --use-scenarios 16 --lam 0.3 --alpha 0.2 \
      --bootstrap ../fpl-archive/.../bootstrap_slim.json --vss

  # mid-season: current squad, 2 FTs, 0.5 ITB
  uv run python stochastic_solver.py --scenario-dir scenarios/ \
      --squad 411,426,... --fts 2 --itb 0.5 --weeks 5 --use-scenarios 16
"""
# ruff: noqa: PLR0915, PLR2004, PLC0415, N803, N806, PLR0912

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

POS_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
LINEUP_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
LINEUP_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE, LINEUP_SIZE, CLUB_LIMIT = 15, 11, 3
BUDGET = 100.0
HIT_COST = 4.0
FT_CAP = 5
BENCH_EPS = 0.08  # flat bench weight: keeps the 4 bench slots honest
CAPTAIN_POOL = 12
POOL_TOP_EV = 90  # tighter pool than step 3: model is S times bigger
POOL_TOP_VALUE = 40
POOL_MIN_OWN = 10.0


# ------------------------------------------------------------------ loading


def load_scenarios(scen_dir: Path, weeks: int):
    files = sorted(glob.glob(str(scen_dir / "scenario_*.csv")))
    if not files:
        raise SystemExit(f"no scenario files in {scen_dir}")
    first = pd.read_csv(files[0])
    gws = sorted(int(c.split("_")[0]) for c in first.columns if c.endswith("_Pts"))[:weeks]
    meta = first[["ID", "Name", "Pos", "Team", "BV"]].copy()
    pts = np.zeros((len(files), len(meta), len(gws)))
    for s, f in enumerate(files):
        df = pd.read_csv(f)
        if not (df.ID.values == meta.ID.values).all():
            df = df.set_index("ID").loc[meta.ID].reset_index()
        pts[s] = df[[f"{g}_Pts" for g in gws]].to_numpy()
    return meta, gws, pts


def stratified_pick(pts, k, rng_seed=0):
    """Pick k scenarios spanning the distribution of total sampled points,
    so the reduced set keeps the spread rather than clustering at the mode."""
    totals = pts.sum(axis=(1, 2))
    order = np.argsort(totals)
    idx = np.linspace(0, len(order) - 1, k).round().astype(int)
    return sorted(order[i] for i in idx)


def load_field(bootstrap: Path | None, meta, proj_total):
    if bootstrap is None:
        return np.zeros(len(meta))
    boot = json.loads(bootstrap.read_text())
    own = {e["id"]: float(e["selected_by_percent"]) / 100 for e in boot["elements"]}
    w = np.array([own.get(int(i), 0.0) for i in meta.ID])
    score = w * np.maximum(proj_total, 0)
    top = np.argsort(-score)[:CAPTAIN_POOL]
    cap = np.zeros_like(w)
    if score[top].sum() > 0:
        cap[top] = score[top] / score[top].sum()
    return w + cap


def build_pool(meta, pts, own_ids, current_squad):
    proj = pts.mean(axis=0).sum(axis=1)
    value = proj / np.maximum(meta.BV.to_numpy(float), 3.5)
    keep = set(np.argsort(-proj)[:POOL_TOP_EV])
    keep |= set(np.argsort(-value)[:POOL_TOP_VALUE])
    keep |= set(np.where(own_ids >= POOL_MIN_OWN / 100)[0])
    keep |= set(current_squad)
    for p, q in POS_QUOTA.items():
        idx = np.where(meta.Pos.to_numpy() == p)[0]
        if sum(1 for i in idx if i in keep) < q + 2:
            keep |= set(idx[np.argsort(-proj[idx])][: q + 2])
    return sorted(keep)


# ------------------------------------------------------------------- model


def build_and_solve(meta, pts, F, d, cfg):
    """
    cfg: dict with lam, alpha, secs, gap, preseason, squad0 (indices),
         itb, fts, fixed_stage1 (None | dict), forced_in (indices)
    Returns dict with stage-1 decisions, per-scenario branches, objective.
    """
    import highspy
    import sasoptpy as so

    S, n, W = pts.shape
    players = list(range(n))
    scens = list(range(S))
    pos = meta.Pos.to_numpy()
    team = meta.Team.to_numpy()
    price = meta.BV.to_numpy(float)
    lam, alpha = cfg["lam"], cfg["alpha"]

    m = so.Model(name="two_stage")

    # ---- stage 1 (shared) ----
    sq1 = m.add_variables(players, name="sq1", vartype=so.BIN)
    y1 = m.add_variables(players, name="y1", vartype=so.BIN)
    c1 = m.add_variables(players, name="c1", vartype=so.BIN)
    tin1 = m.add_variables(players, name="tin1", vartype=so.BIN)
    tout1 = m.add_variables(players, name="tout1", vartype=so.BIN)
    pt1 = m.add_variable(name="pt1", lb=0)

    # ---- stage 2 (scenario-indexed), weeks 1..W-1 ----
    sq = m.add_variables(players, range(1, W), scens, name="sq", vartype=so.BIN)
    y = m.add_variables(players, range(1, W), scens, name="y", vartype=so.BIN)
    c = m.add_variables(players, range(1, W), scens, name="c", vartype=so.BIN)
    tin = m.add_variables(players, range(1, W), scens, name="tin", vartype=so.BIN)
    tout = m.add_variables(players, range(1, W), scens, name="tout", vartype=so.BIN)
    pt = m.add_variables(range(1, W), scens, name="pt", lb=0)
    fts = m.add_variables(range(W + 1), scens, name="fts", lb=0, ub=FT_CAP, vartype=so.INT)

    # eta is the VaR level, so it lies inside the range the score can take.
    # A placeholder bound of -1e6 let the feasibility-jump heuristic open with
    # an incumbent near 1e6 and then climb back, wasting most of the time
    # limit. Bounding it to the achievable range costs nothing and removes
    # that whole detour.
    best_case = np.zeros(S)
    for w_ in range(W):
        top = np.sort(pts[:, :, w_], axis=1)[:, -LINEUP_SIZE:]
        best_case += d[w_] * (top.sum(axis=1) + top[:, -1])  # XI + captain
    eta_hi = float((best_case - F).max())
    worst_case = -HIT_COST * FT_CAP * float(np.sum(d))
    eta_lo = float((worst_case - F).min())
    m_pad = 0.05 * max(abs(eta_hi), abs(eta_lo), 1.0)
    eta = m.add_variable(name="eta", lb=eta_lo - m_pad, ub=eta_hi + m_pad)
    z = m.add_variables(scens, name="z", lb=0)

    def squad_at(p, w, s):
        return sq1[p] if w == 0 else sq[p, w, s]

    # ---- stage-1 structure ----
    if cfg["preseason"]:
        for p in players:
            m.add_constraint(tin1[p] == 0, name=f"ps_ti_{p}")
            m.add_constraint(tout1[p] == 0, name=f"ps_to_{p}")
        m.add_constraint(pt1 == 0, name="ps_pt")
        m.add_constraint(so.expr_sum(price[p] * sq1[p] for p in players) <= BUDGET, name="ps_budget")
    else:
        sq0 = set(cfg["squad0"])
        for p in players:
            in0 = 1 if p in sq0 else 0
            m.add_constraint(sq1[p] == in0 + tin1[p] - tout1[p], name=f"flow1_{p}")
            if in0:
                m.add_constraint(tin1[p] == 0, name=f"noti_{p}")
            else:
                m.add_constraint(tout1[p] == 0, name=f"noto_{p}")
        spend = so.expr_sum(price[p] * (tin1[p] - tout1[p]) for p in players)
        m.add_constraint(spend <= cfg["itb"], name="budget1")
        tc1 = so.expr_sum(tin1[p] for p in players)
        m.add_constraint(pt1 >= tc1 - cfg["fts"], name="hits1")
        for s in scens:
            m.add_constraint(fts[1, s] <= cfg["fts"] - tc1 + pt1 + 1, name=f"ftflow1_{s}")

    m.add_constraint(so.expr_sum(sq1[p] for p in players) == SQUAD_SIZE, name="sq15_1")
    for q, k in POS_QUOTA.items():
        m.add_constraint(so.expr_sum(sq1[p] for p in players if pos[p] == q) == k, name=f"pos1_{q}")
    for t in sorted(set(team)):
        m.add_constraint(so.expr_sum(sq1[p] for p in players if team[p] == t) <= CLUB_LIMIT, name=f"club1_{t}")
    m.add_constraint(so.expr_sum(y1[p] for p in players) == LINEUP_SIZE, name="xi1")
    m.add_constraint(so.expr_sum(c1[p] for p in players) == 1, name="cap1")
    for q in POS_QUOTA:
        e = so.expr_sum(y1[p] for p in players if pos[p] == q)
        m.add_constraint(e >= LINEUP_MIN[q], name=f"f1min_{q}")
        m.add_constraint(e <= LINEUP_MAX[q], name=f"f1max_{q}")
    for p in players:
        m.add_constraint(y1[p] <= sq1[p], name=f"yx1_{p}")
        m.add_constraint(c1[p] <= y1[p], name=f"cy1_{p}")

    if cfg.get("forced_in"):
        for p in cfg["forced_in"]:
            m.add_constraint(sq1[p] == 1, name=f"force_{p}")
    if cfg.get("fixed_stage1") is not None:
        fx = cfg["fixed_stage1"]
        for p in players:
            m.add_constraint(sq1[p] == (1 if p in fx["squad"] else 0), name=f"fx_{p}")

    # ---- stage-2 structure per scenario ----
    for s in scens:
        for w in range(1, W):
            for p in players:
                m.add_constraint(sq[p, w, s] == squad_at(p, w - 1, s) + tin[p, w, s] - tout[p, w, s], name=f"flow_{p}_{w}_{s}")
                m.add_constraint(tin[p, w, s] + tout[p, w, s] <= 1, name=f"io_{p}_{w}_{s}")
                m.add_constraint(y[p, w, s] <= sq[p, w, s], name=f"yx_{p}_{w}_{s}")
                m.add_constraint(c[p, w, s] <= y[p, w, s], name=f"cy_{p}_{w}_{s}")
            m.add_constraint(so.expr_sum(sq[p, w, s] for p in players) == SQUAD_SIZE, name=f"sq15_{w}_{s}")
            for q, k in POS_QUOTA.items():
                m.add_constraint(so.expr_sum(sq[p, w, s] for p in players if pos[p] == q) == k, name=f"pos_{q}_{w}_{s}")
            for t in sorted(set(team)):
                m.add_constraint(so.expr_sum(sq[p, w, s] for p in players if team[p] == t) <= CLUB_LIMIT, name=f"club_{t}_{w}_{s}")
            m.add_constraint(so.expr_sum(y[p, w, s] for p in players) == LINEUP_SIZE, name=f"xi_{w}_{s}")
            m.add_constraint(so.expr_sum(c[p, w, s] for p in players) == 1, name=f"cap_{w}_{s}")
            for q in POS_QUOTA:
                e = so.expr_sum(y[p, w, s] for p in players if pos[p] == q)
                m.add_constraint(e >= LINEUP_MIN[q], name=f"fmin_{q}_{w}_{s}")
                m.add_constraint(e <= LINEUP_MAX[q], name=f"fmax_{q}_{w}_{s}")
            tc = so.expr_sum(tin[p, w, s] for p in players)
            m.add_constraint(tc <= cfg["max_rt"], name=f"rtcap_{w}_{s}")
            m.add_constraint(pt[w, s] >= tc - fts[w, s], name=f"hits_{w}_{s}")
            m.add_constraint(fts[w + 1, s] <= fts[w, s] - tc + pt[w, s] + 1, name=f"ftflow_{w}_{s}")
            # budget: cumulative spend within scenario cannot exceed itb
            cum = so.expr_sum(price[p] * (tin1[p] - tout1[p]) for p in players) if not cfg["preseason"] else 0
            cum = cum + so.expr_sum(price[p] * (tin[p, ww, s] - tout[p, ww, s]) for p in players for ww in range(1, w + 1))
            m.add_constraint(cum <= (cfg["itb"] if not cfg["preseason"] else 0.0), name=f"budget_{w}_{s}")
        if cfg["preseason"]:
            m.add_constraint(fts[1, s] <= 1 + 1, name=f"ftinit_{s}")  # 1 FT banked into GW2

    # ---- objective ----
    score = {}
    for s in scens:
        expr = so.expr_sum(d[0] * pts[s, p, 0] * (y1[p] + c1[p] + BENCH_EPS * (sq1[p] - y1[p])) for p in players)
        expr = expr - d[0] * HIT_COST * pt1
        for w in range(1, W):
            expr = expr + so.expr_sum(d[w] * pts[s, p, w] * (y[p, w, s] + c[p, w, s] + BENCH_EPS * (sq[p, w, s] - y[p, w, s])) for p in players)
            expr = expr - d[w] * HIT_COST * pt[w, s]
        score[s] = expr - float(F[s])
        m.add_constraint(z[s] >= eta - score[s], name=f"cvar_{s}")

    ev = so.expr_sum(score[s] for s in scens) / S
    cvar = eta - so.expr_sum(z[s] for s in scens) / (alpha * S)
    m.set_objective(-((1 - lam) * ev + lam * cvar), name="obj", sense="N")

    m.export_mps("two_stage.mps")
    h = highspy.Highs()
    h.setOptionValue("time_limit", cfg["secs"])
    h.setOptionValue("mip_rel_gap", cfg["gap"])
    h.readModel("two_stage.mps")
    h.run()
    Path("two_stage.mps").unlink(missing_ok=True)
    info = h.getInfo()
    vals = np.array(h.getSolution().col_value)
    names = [h.getColName(i)[1] for i in range(h.getNumCol())]
    idx = {nm: i for i, nm in enumerate(names)}

    def val(name):
        return vals[idx[name]]

    squad1 = [p for p in players if val(f"sq1[{p}]") > 0.5]
    res = {
        "objective": -float(info.objective_function_value),
        "gap": float(info.mip_gap),
        "squad1": squad1,
        "lineup1": [p for p in players if val(f"y1[{p}]") > 0.5],
        "captain1": next(p for p in players if val(f"c1[{p}]") > 0.5),
        "tin1": [p for p in players if val(f"tin1[{p}]") > 0.5],
        "tout1": [p for p in players if val(f"tout1[{p}]") > 0.5],
        "hits1": float(val("pt1")),
        "branches": [],
    }
    for s in scens:
        moves = []
        for w in range(1, W):
            ins = [p for p in players if val(f"tin[{p},{w},{s}]") > 0.5]
            outs = [p for p in players if val(f"tout[{p},{w},{s}]") > 0.5]
            if ins or outs:
                moves.append((w, ins, outs))
        res["branches"].append(moves)
    return res


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-stage stochastic FPL solver.")
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=Path, default=None)
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--use-scenarios", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--decay", type=float, default=0.87)
    ap.add_argument("--secs", type=float, default=900)
    ap.add_argument("--gap", type=float, default=0.005)
    ap.add_argument("--preseason", action="store_true")
    ap.add_argument("--squad", type=str, default=None, help="15 comma-sep FPL IDs")
    ap.add_argument("--itb", type=float, default=0.0)
    ap.add_argument("--fts", type=int, default=1)
    ap.add_argument("--force", type=str, default=None)
    ap.add_argument(
        "--max-recourse-transfers",
        type=int,
        default=1,
        help="cap on stage-2 transfers per week; tempers the within-scenario foresight that two-stage models have",
    )
    ap.add_argument("--vss", action="store_true", help="also solve on the mean and price the difference")
    args = ap.parse_args()

    if not args.preseason and not args.squad:
        raise SystemExit("either --preseason or --squad is required")

    meta, gws, pts_all = load_scenarios(args.scenario_dir, args.weeks)
    proj_total = pts_all.mean(axis=0).sum(axis=1)
    field_w = load_field(args.bootstrap, meta, proj_total)

    squad0_full = np.where(meta.ID.isin([int(i) for i in args.squad.split(",")]))[0].tolist() if args.squad else []
    pool = build_pool(meta, pts_all, field_w, squad0_full)
    meta_p = meta.iloc[pool].reset_index(drop=True)
    pts_pool = pts_all[:, pool, :]
    pick = stratified_pick(pts_all, args.use_scenarios)
    pts = pts_pool[pick]
    # moment matching: stratified reduction keeps the aggregate spread but
    # leaves per-player means noisy at small S; shift each player-GW cell so
    # the reduced-set mean equals the full-set mean. Correlation structure
    # within scenarios is untouched.
    pts = pts + (pts_pool.mean(axis=0) - pts.mean(axis=0))[None, :, :]
    S, n, W = pts.shape
    d = np.array([args.decay**i for i in range(W)])
    F = np.einsum("snw,n,w->s", pts_all[pick], field_w, d) if args.bootstrap else np.zeros(S)
    squad0 = [pool.index(i) for i in squad0_full]
    forced = [pool.index(i) for i in np.where(meta.ID.isin([int(x) for x in args.force.split(",")]))[0]] if args.force else None

    print(f"[2stage] {S}/{pts_all.shape[0]} scenarios (stratified), pool {n}, GWs {gws[0]}-{gws[-1]}, lam={args.lam}")

    cfg = {
        "lam": args.lam,
        "alpha": args.alpha,
        "secs": args.secs,
        "gap": args.gap,
        "preseason": args.preseason,
        "squad0": squad0,
        "itb": args.itb,
        "fts": args.fts,
        "fixed_stage1": None,
        "forced_in": forced,
        "max_rt": args.max_recourse_transfers,
    }
    res = build_and_solve(meta_p, pts, F, d, cfg)

    print(f"\n[2stage] objective {res['objective']:.2f}  (gap {res['gap'] * 100:.2f}%)")
    if args.preseason:
        rows = meta_p.iloc[res["squad1"]][["ID", "Name", "Pos", "Team", "BV"]]
        rows = rows.assign(
            XI=["*" if p in res["lineup1"] else "" for p in res["squad1"]], C=["C" if p == res["captain1"] else "" for p in res["squad1"]]
        )
        print("\nStage-1 squad (the committed decision):")
        print(rows.sort_values(["Pos", "BV"], ascending=[True, False]).to_string(index=False))
    else:
        tin = meta_p.iloc[res["tin1"]].Name.tolist()
        tout = meta_p.iloc[res["tout1"]].Name.tolist()
        print(f"\nStage-1 transfers: {tout} -> {tin}  (hits: {res['hits1']:.0f})")
        print(f"Captain: {meta_p.iloc[res['captain1']].Name}")

    # branch report: what does the recourse do, and how often?
    move_counter = Counter()
    for moves in res["branches"]:
        for w, ins, outs in moves:
            key = (w, ", ".join(sorted(meta_p.iloc[o].Name for o in outs)), ", ".join(sorted(meta_p.iloc[i].Name for i in ins)))
            move_counter[key] += 1
    print("\nRecourse branches (move, share of scenarios making it):")
    for (w, o, i), cnt in move_counter.most_common(12):
        print(f"  GW+{w}: [{o}] -> [{i}]   {cnt}/{S} scenarios")

    if args.vss:
        print("\n[2stage] VSS: solving certainty-equivalent (mean) problem ...")
        mean_pts = pts.mean(axis=0, keepdims=True)
        cfg_m = dict(cfg)
        res_m = build_and_solve(meta_p, mean_pts, np.array([F.mean()]), d, cfg_m)
        print("[2stage] evaluating its stage-1 under the full scenario set ...")
        cfg_f = dict(cfg)
        cfg_f["fixed_stage1"] = {"squad": set(res_m["squad1"])}
        res_f = build_and_solve(meta_p, pts, F, d, cfg_f)
        vss = res["objective"] - res_f["objective"]
        print(f"\n[2stage] stochastic obj {res['objective']:.2f} | mean-plan-with-recourse {res_f['objective']:.2f} | VSS = {vss:+.2f}")
        a = set(meta_p.iloc[res["squad1"]].ID)
        b = set(meta_p.iloc[res_m["squad1"]].ID)
        names = meta_p[meta_p.ID.isin(a ^ b)].Name.tolist()
        print(f"[2stage] stage-1 squad difference vs mean-plan: {len(a ^ b) // 2} swaps ({', '.join(names) if names else 'identical'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
