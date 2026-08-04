#!/usr/bin/env python3
"""
scenario_generator.py — step 2 of the roadmap.

Turns point-estimate projections (review + solio blend) into S sampled outcome
paths per player per gameweek, for consumption by the CVaR objective (step 3)
and the two-stage stochastic reformulation (step 4).

Design principles
-----------------
1. OUTCOME variance dominates. Each projection is decomposed into components
   (appearance, goals, assists, clean sheets, DefCon, saves, bonus) and the
   EVENTS are resampled, not the point estimate. A 5.0-projected midfielder
   can score 1 or 15 in a scenario, as in reality.

2. TEAM-LEVEL correlation. Scenarios first sample each team's goals scored and
   conceded per gameweek; player outcomes are conditioned on those. Teammates'
   clean sheets are perfectly correlated (as in reality), goals are allocated
   from the team total, so stacking one team's defence is correlated risk —
   exactly what a rank-protection objective must see.

3. MODEL disagreement = parameter uncertainty. Where review and solio disagree
   on a player, a per-scenario multiplicative tilt widens that player's rates.
   The tilt is drawn ONCE per scenario per player and persists across
   gameweeks: epistemic uncertainty does not re-randomise weekly.

4. MEAN PRESERVATION. The analytic expectation of each player's sampled points
   is calibrated back to the blended projection, so scenarios are a mean-
   preserving spread. The solver's EV view is unchanged; only tails are added.

Component priors are documented constants, calibratable from the archive once
realised 26/27 data accumulates.

Usage
-----
    python scenario_generator.py --sources review solio --data-dir data \
        --scenarios 200 --out scenarios/ [--seed 42] [--horizon 12]

Outputs
-------
    scenarios/scenario_000.csv ... scenario_S-1.csv   (solver schema: {gw}_Pts,
        {gw}_xMins per player — SAMPLED outcomes, not expectations)
    scenarios/summary.csv       per-player mean/sd/quantiles across scenarios
    scenarios/manifest.json     settings, seed, calibration diagnostics
"""
# ruff: noqa: PLR0915

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ scoring

GOAL_PTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_PTS = 3
DEFCON_PTS = 2
POS_ALIASES = {"G": "GKP", "D": "DEF", "M": "MID", "F": "FWD"}

# Residual-share priors: how a starter's above-appearance expectation splits.
# attack = goals+assists share; the rest is cs / defcon / saves / bonus.
# Calibratable from archive; sources: FPL scoring composition, 25/26 season.
SHARE = {
    "GKP": {"attack": 0.02, "cs": 0.52, "defcon": 0.00, "saves": 0.28, "bonus": 0.18},
    "DEF": {"attack": 0.22, "cs": 0.42, "defcon": 0.20, "saves": 0.00, "bonus": 0.16},
    "MID": {"attack": 0.62, "cs": 0.09, "defcon": 0.11, "saves": 0.00, "bonus": 0.18},
    "FWD": {"attack": 0.78, "cs": 0.00, "defcon": 0.04, "saves": 0.00, "bonus": 0.18},
}
ASSIST_FRACTION = 0.45  # of attack EV, share coming from assists (rest goals)
GOALS_WITH_ASSIST = 0.72  # P(a sampled team goal carries an assist)
TEAM_GOAL_FLOOR = 0.55  # min team goals lambda (very defensive teams)
UNLISTED_GOAL_SHARE = 0.06  # share of team goals scored by players outside pool
DISAGREE_TILT_SCALE = 1.0  # multiplier on disagreement-implied parameter sd
MIN_START_XMINS = 45.0
EPS = 1e-9
CS_FULL = 4  # cs_pts value identifying GK/DEF
CAMEO_XMINS_FLOOR = 5
SIXTY = 60
BENCH_APPEAR_PROB = 0.35  # P(a sub with xMins in (5,45) gets on the pitch)
SIXTY_PLUS_KNEE = 62.0  # xMins above this => starter plays 60+ if starts


def canon_pos(p: str) -> str:
    return POS_ALIASES.get(str(p), str(p))


# ------------------------------------------------------------------- loading


def load_blend(data_dir: Path, sources: list[str], horizon: int | None) -> pd.DataFrame:
    """50/50 blend (normalised by availability), keeping per-source Pts for
    the disagreement signal."""
    dfs = {}
    for s in sources:
        df = pd.read_csv(data_dir / f"{s}.csv", encoding="utf-8-sig")
        df["Pos"] = df["Pos"].map(canon_pos)
        dfs[s] = df

    gw_sets = [{int(c.split("_")[0]) for c in d.columns if c.endswith("_Pts")} for d in dfs.values()]
    shared = sorted(set.intersection(*gw_sets))
    if horizon:
        shared = shared[:horizon]
    if not shared:
        raise SystemExit("no overlapping gameweeks between sources")

    base_name = sources[0]
    base = dfs[base_name][["ID", "Name", "Pos", "Team", "BV"]].copy()
    for s, d in dfs.items():
        cols = ["ID"] + [f"{g}_{t}" for g in shared for t in ("Pts", "xMins")]
        base = base.merge(d[cols], on="ID", how="inner", suffixes=("", f"__{s}"))
    # after merge, first source's cols are unsuffixed; rename for uniformity
    ren = {f"{g}_{t}": f"{g}_{t}__{base_name}" for g in shared for t in ("Pts", "xMins")}
    base = base.rename(columns=ren)

    for g in shared:
        base[f"{g}_Pts"] = np.mean([base[f"{g}_Pts__{s}"] for s in sources], axis=0)
        base[f"{g}_xMins"] = np.mean([base[f"{g}_xMins__{s}"] for s in sources], axis=0)
        # relative model disagreement -> parameter sd
        spread = np.std([base[f"{g}_Pts__{s}"] for s in sources], axis=0)
        base[f"{g}_disagree"] = spread / np.maximum(base[f"{g}_Pts"], 0.5)

    base.attrs["gameweeks"] = shared
    return base


# -------------------------------------------------------------- decomposition


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per player-GW component rates whose analytic expectation reproduces the
    blended Pts. Team clean-sheet probability is pooled from GK+DEF residuals.
    """
    gws = df.attrs["gameweeks"]
    out = df.copy()

    for g in gws:
        xm = out[f"{g}_xMins"].to_numpy(float)
        pts = out[f"{g}_Pts"].to_numpy(float)
        pos = out["Pos"].to_numpy()

        p_start = np.clip(xm / 90.0, 0, 1) * (xm >= MIN_START_XMINS)
        p_cameo = np.where((xm > CAMEO_XMINS_FLOOR) & (xm < MIN_START_XMINS), BENCH_APPEAR_PROB, 0.0)
        p_play = np.clip(p_start + p_cameo, 0, 1)
        p_60 = np.where(xm >= SIXTY_PLUS_KNEE, p_start, p_start * (xm / 90.0))
        app_ev = 2 * p_60 + 1 * np.clip(p_play - p_60, 0, None)

        # Defensive players' projections are NET of the goals-conceded
        # penalty; sampling re-subtracts it, so budget it back in here.
        # E[floor(C/2)] for C ~ Poisson(lam_conceded), computed per team below
        # after team_cs is known — first pass uses a placeholder, corrected
        # at the end of this loop iteration.
        resid = np.maximum(pts - app_ev, 0.0)

        att_ev = np.zeros_like(resid)
        cs_ev = np.zeros_like(resid)
        dc_ev = np.zeros_like(resid)
        sv_ev = np.zeros_like(resid)
        bn_ev = np.zeros_like(resid)
        for p, sh in SHARE.items():
            k = pos == p
            att_ev[k] = resid[k] * sh["attack"]
            cs_ev[k] = resid[k] * sh["cs"]
            dc_ev[k] = resid[k] * sh["defcon"]
            sv_ev[k] = resid[k] * sh["saves"]
            bn_ev[k] = resid[k] * sh["bonus"]

        goal_pts = np.vectorize(GOAL_PTS.get)(pos).astype(float)
        cs_pts = np.vectorize(CS_PTS.get)(pos).astype(float)

        lam_goal = att_ev * (1 - ASSIST_FRACTION) / goal_pts
        lam_assist = att_ev * ASSIST_FRACTION / ASSIST_PTS
        # conditional-on-playing rates (EVs above are unconditional)
        safe_p = np.maximum(p_play, 1e-6)
        out[f"{g}_lam_goal"] = lam_goal / safe_p
        out[f"{g}_lam_assist"] = lam_assist / safe_p
        out[f"{g}_p_defcon"] = np.clip(dc_ev / DEFCON_PTS / np.maximum(p_60, 1e-6), 0, 0.85)
        out[f"{g}_saves_ev"] = sv_ev / safe_p
        out[f"{g}_bonus_ev"] = bn_ev / safe_p
        out[f"{g}_p_play"] = p_play
        out[f"{g}_p_60"] = p_60
        out[f"{g}_app_ev"] = app_ev

        # team clean-sheet prob: pool defensive players' cs EV
        with np.errstate(invalid="ignore", divide="ignore"):
            cs_prob_player = np.where(cs_pts > 0, cs_ev / cs_pts / np.maximum(p_60, 1e-6), np.nan)
        tmp = pd.DataFrame({"Team": out["Team"], "csp": cs_prob_player, "w": (cs_pts > 0) & (xm >= SIXTY)})
        team_cs = (tmp[tmp.w].groupby("Team").csp.median()).clip(0.02, 0.75)
        out[f"{g}_team_cs"] = out["Team"].map(team_cs).fillna(0.25)

        # EV CONSERVATION: pooling forces each player onto the TEAM cs prob.
        # The difference between a player's implied cs EV and the pooled one
        # must flow back into his other components, or premium defenders
        # silently lose EV (and cheap ones gain it).
        cs_ev_pooled = out[f"{g}_team_cs"].to_numpy(float) * p_60 * cs_pts
        leftover = cs_ev - cs_ev_pooled  # can be +/-
        cs_ev = cs_ev_pooled
        att_ev = np.maximum(att_ev + 0.7 * leftover, 0.0)
        dc_ev = np.maximum(dc_ev + 0.3 * leftover, 0.0)

        # conceded-penalty budget (FIX for mean bias): add E[floor(C/2)] back
        # into defensive players' residual and re-split their components
        lam_c = -np.log(np.clip(out[f"{g}_team_cs"].to_numpy(float), 0.02, 0.98))
        ks = np.arange(0, 12)
        # E[floor(C/2)] per player from their team's lambda
        pk = np.exp(-lam_c[:, None]) * np.power(lam_c[:, None], ks) / np.array([math.factorial(k) for k in ks])
        e_pen = (pk * (ks // 2)).sum(axis=1)
        is_def = cs_pts >= CS_FULL
        extra = np.where(is_def, e_pen * p_60, 0.0)
        scale = np.divide(resid + extra, np.maximum(resid, EPS), out=np.ones_like(resid), where=resid > EPS)
        adj = np.where(is_def, scale, 1.0)
        att_ev, cs_ev = att_ev * adj, cs_ev * adj
        dc_ev, sv_ev, bn_ev = dc_ev * adj, sv_ev * adj, bn_ev * adj
        # recompute stored rates with the corrected components
        lam_goal = att_ev * (1 - ASSIST_FRACTION) / goal_pts
        lam_assist = att_ev * ASSIST_FRACTION / ASSIST_PTS
        out[f"{g}_lam_goal"] = lam_goal / safe_p
        out[f"{g}_lam_assist"] = lam_assist / safe_p
        out[f"{g}_p_defcon"] = np.clip(dc_ev / DEFCON_PTS / np.maximum(p_60, 1e-6), 0, 0.85)
        out[f"{g}_saves_ev"] = sv_ev / safe_p
        out[f"{g}_bonus_ev"] = bn_ev / safe_p
        # cs rate needs re-pooling with corrected cs_ev
        with np.errstate(invalid="ignore", divide="ignore"):
            cs_prob_player = np.where(cs_pts > 0, cs_ev / cs_pts / np.maximum(p_60, 1e-6), np.nan)
        tmp = pd.DataFrame({"Team": out["Team"], "csp": cs_prob_player, "w": (cs_pts > 0) & (xm >= SIXTY)})
        team_cs = (tmp[tmp.w].groupby("Team").csp.median()).clip(0.02, 0.75)
        out[f"{g}_team_cs"] = out["Team"].map(team_cs).fillna(0.25)

        # team goals lambda from FINAL goal rates (post-redistribution)
        final_lam_goal = out[f"{g}_lam_goal"].to_numpy(float) * safe_p
        tg = pd.DataFrame({"Team": out["Team"], "lg": final_lam_goal}).groupby("Team").lg.sum() / (1 - UNLISTED_GOAL_SHARE)
        out[f"{g}_team_goals"] = out["Team"].map(tg.clip(lower=TEAM_GOAL_FLOOR))

    return out


# ------------------------------------------------------------------ sampling


def sample_scenarios(dec: pd.DataFrame, n_scen: int, seed: int):
    """
    Yields (scenario_index, DataFrame in solver schema with sampled outcomes).
    """
    rng = np.random.default_rng(seed)
    gws = dec.attrs["gameweeks"]
    n = len(dec)
    teams = dec["Team"].to_numpy()
    team_list = sorted(set(teams))
    t_idx = {t: i for i, t in enumerate(team_list)}
    player_team = np.array([t_idx[t] for t in teams])
    pos = dec["Pos"].to_numpy()
    goal_pts = np.vectorize(GOAL_PTS.get)(pos).astype(float)
    cs_pts = np.vectorize(CS_PTS.get)(pos).astype(float)
    is_gk = pos == "GKP"

    # epistemic tilt: one draw per scenario per player, persists across GWs
    mean_dis = np.mean([dec[f"{g}_disagree"].to_numpy(float) for g in gws], axis=0)
    tilt_sd = np.clip(mean_dis * DISAGREE_TILT_SCALE, 0.0, 0.6)

    for s in range(n_scen):
        tilt = np.exp(rng.normal(0, tilt_sd) - 0.5 * tilt_sd**2)  # E[tilt]=1
        cols = {"ID": dec["ID"], "Name": dec["Name"], "Pos": dec["Pos"], "Team": dec["Team"], "BV": dec["BV"]}
        for g in gws:
            lam_tg = dec[f"{g}_team_goals"].groupby(dec["Team"]).first()
            team_goals = rng.poisson(lam_tg.reindex(team_list).to_numpy())
            p_cs_team = dec[f"{g}_team_cs"].groupby(dec["Team"]).first().reindex(team_list).to_numpy()
            lam_conceded = -np.log(np.clip(p_cs_team, 0.02, 0.98))
            goals_against = rng.poisson(lam_conceded)
            team_cs = goals_against == 0

            p_play = dec[f"{g}_p_play"].to_numpy(float)
            p60 = dec[f"{g}_p_60"].to_numpy(float)
            plays = rng.random(n) < p_play
            plays60 = plays & (rng.random(n) < np.divide(p60, np.maximum(p_play, EPS)))

            # allocate team goals to players by tilted conditional rates
            lam_g = dec[f"{g}_lam_goal"].to_numpy(float) * tilt * plays
            goals = np.zeros(n, dtype=int)
            for ti, tg_total in enumerate(team_goals):
                if tg_total == 0:
                    continue
                k = player_team == ti
                w = lam_g[k]
                wsum = w.sum()
                if wsum <= EPS:
                    continue
                pvec = np.append(w / wsum * (1 - UNLISTED_GOAL_SHARE), UNLISTED_GOAL_SHARE)
                alloc = rng.multinomial(tg_total, pvec)
                goals[k] = alloc[:-1]

            # assists: direct Poisson per player (mean-exact). Team-goal
            # normalisation was tried and structurally under-delivers when a
            # team's summed assist rates exceed sampled-goals x assist-share.
            lam_a = dec[f"{g}_lam_assist"].to_numpy(float) * tilt * plays
            assists = rng.poisson(np.clip(lam_a, 0, None))

            defcon = (plays60 & (rng.random(n) < dec[f"{g}_p_defcon"].to_numpy(float) * tilt)).astype(int)
            cs_flag = team_cs[player_team] & plays60
            conceded = goals_against[player_team]
            # save POINTS sampled directly: mean-exact, coarser granularity
            gk_savepts = np.where(is_gk & plays, rng.poisson(np.clip(dec[f"{g}_saves_ev"].to_numpy(float), 0, None)), 0)
            # bonus: only with a notable event, scaled to preserve mean
            bn_ev = dec[f"{g}_bonus_ev"].to_numpy(float)
            lam_b = np.clip(bn_ev * 1.06, 0, None)  # 1.06 offsets the cap at 3
            bonus = np.where(plays, np.minimum(rng.poisson(lam_b), 3), 0)

            pts = np.zeros(n)
            pts += np.where(plays60, 2, np.where(plays, 1, 0))
            pts += goals * goal_pts
            pts += assists * ASSIST_PTS
            pts += np.where(cs_flag, cs_pts, 0)
            pts += defcon * DEFCON_PTS
            pts += np.where(is_gk, gk_savepts, 0)
            pts -= np.where((cs_pts >= CS_FULL) & plays60, (conceded // 2), 0)
            pts += bonus

            mins = np.where(plays60, np.minimum(dec[f"{g}_xMins"], 90), np.where(plays, 30, 0))
            cols[f"{g}_Pts"] = pts
            cols[f"{g}_xMins"] = mins
        yield s, pd.DataFrame(cols)


# ---------------------------------------------------------------- calibration


def calibrate_and_report(dec: pd.DataFrame, n_check: int, seed: int) -> dict:
    """Empirical mean across n_check scenarios vs blended projection."""
    gws = dec.attrs["gameweeks"]
    acc = None
    for _, df in sample_scenarios(dec, n_check, seed):
        v = df[[f"{g}_Pts" for g in gws]].to_numpy()
        acc = v if acc is None else acc + v
    emp = acc / n_check
    proj = dec[[f"{g}_Pts" for g in gws]].to_numpy()
    err = emp - proj
    mask = proj > 1.0
    return {
        "players": len(dec),
        "gameweeks": len(gws),
        "check_scenarios": n_check,
        "mean_bias_pts": float(err[mask].mean()),
        "mean_abs_err_pts": float(np.abs(err[mask]).mean()),
        "corr_emp_vs_proj": float(np.corrcoef(emp[mask], proj[mask])[0, 1]),
    }


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate outcome scenarios from projections.")
    ap.add_argument("--sources", nargs="+", default=["review", "solio"])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("scenarios"))
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate-only", action="store_true")
    args = ap.parse_args()

    blend = load_blend(args.data_dir, args.sources, args.horizon)
    dec = decompose(blend)
    dec.attrs["gameweeks"] = blend.attrs["gameweeks"]

    print(f"[scen] {len(dec)} players, GWs {dec.attrs['gameweeks'][0]}-{dec.attrs['gameweeks'][-1]}, sources {args.sources}")

    diag = calibrate_and_report(dec, min(150, max(args.scenarios, 50)), args.seed + 1)
    print(f"[scen] calibration: bias {diag['mean_bias_pts']:+.3f} pts, MAE {diag['mean_abs_err_pts']:.3f}, corr {diag['corr_emp_vs_proj']:.4f}")
    if args.calibrate_only:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    gws = dec.attrs["gameweeks"]
    tot = np.zeros((args.scenarios, len(dec)))
    for s, df in sample_scenarios(dec, args.scenarios, args.seed):
        df.to_csv(args.out / f"scenario_{s:03d}.csv", index=False, float_format="%.2f")
        tot[s] = df[[f"{g}_Pts" for g in gws]].sum(axis=1).to_numpy()

    summary = pd.DataFrame(
        {
            "ID": dec["ID"],
            "Name": dec["Name"],
            "Pos": dec["Pos"],
            "Team": dec["Team"],
            "proj_total": dec[[f"{g}_Pts" for g in gws]].sum(axis=1),
            "scen_mean": tot.mean(axis=0),
            "scen_sd": tot.std(axis=0),
            "p10": np.percentile(tot, 10, axis=0),
            "p90": np.percentile(tot, 90, axis=0),
        }
    )
    summary.to_csv(args.out / "summary.csv", index=False, float_format="%.2f")

    manifest = {
        "sources": args.sources,
        "scenarios": args.scenarios,
        "seed": args.seed,
        "gameweeks": gws,
        "diagnostics": diag,
        "priors": {
            "share": SHARE,
            "assist_fraction": ASSIST_FRACTION,
            "goals_with_assist": GOALS_WITH_ASSIST,
            "disagree_tilt_scale": DISAGREE_TILT_SCALE,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[scen] wrote {args.scenarios} scenarios + summary to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
