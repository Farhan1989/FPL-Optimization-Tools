#!/usr/bin/env python3
"""
cvar_solver.py — step 3 of the roadmap.

Optimises a squad against the FIELD, not just against expected points.

Where the stock solver maximises E[points], this maximises

    (1 - lambda) * E[Delta]  +  lambda * CVaR_alpha[Delta]

with Delta_s = (your decayed score in scenario s) - (the field's decayed score
in scenario s). The field is an effective-ownership portfolio built from the
archived bootstrap snapshot, so scenarios where the template hauls are
scenarios where you are punished for not owning it — which is precisely the
mechanism by which effective ownership matters, and which a linear EV
objective cannot see (subtracting the field from an expectation is a
constant; subtracting it inside a tail statistic is not).

CVaR enters via the Rockafellar-Uryasev linearisation: one free variable eta,
one slack per scenario, everything stays a MILP for HiGHS.

    max  (1-l)/S * sum_s Delta_s + l * (eta - 1/(alpha*S) * sum_s z_s)
    s.t. z_s >= eta - Delta_s,   z_s >= 0

lambda = 0    -> pure EV maximiser (sanity anchor; should broadly agree with
                 the stock solver on the same data)
lambda -> 1   -> pure tail protection: the squad whose worst alpha-fraction of
                 scenarios against the field is least bad. Expect it to hug
                 effective ownership.

Decision variables: squad (15), lineup and captain per gameweek. One plan
across all scenarios — non-anticipative everywhere. Transfers are step 4.

Usage
-----
    uv run python cvar_solver.py --scenario-dir scenarios/ \
        --bootstrap ../fpl-archive/.../bootstrap_slim.json \
        --weeks 4 --lam 0.5 --alpha 0.2

    # evaluate a fixed squad instead of optimising:
    uv run python cvar_solver.py ... --evaluate 411,426,368,...

    # score several named squads on the SAME scenarios and compare them
    # pairwise ('name=ids', squads separated by ';'):
    uv run python cvar_solver.py ... \
        --evaluate-many "held=109,8,469,...;ev=109,8,31,...;stoch=109,8,31,..."

Outputs the chosen squad, its E[Delta], CVaR, P(beat field), and the same
metrics for lambda=0 for comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POS_QUOTA = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
LINEUP_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
LINEUP_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15
LINEUP_SIZE = 11
CLUB_LIMIT = 3
BUDGET = 100.0
CAPTAIN_POOL = 12  # top players by own*proj considered field captains
POOL_TOP_EV = 130  # pool: top by projected total
POOL_TOP_VALUE = 60  # plus top by projection per price
POOL_MIN_OWN = 8.0  # plus everyone above this ownership (field cover)
UNPAIRED_NOISE_BAND = 2.5  # documented habit: unpaired E[D] gaps under this are called a tie


# ------------------------------------------------------------------ loading


def load_scenarios(scen_dir: Path, weeks: int, max_scen: int | None):
    files = sorted(glob.glob(str(scen_dir / "scenario_*.csv")))
    if not files:
        raise SystemExit(f"no scenario files in {scen_dir}")
    if max_scen:
        files = files[:max_scen]
    first = pd.read_csv(files[0])
    gws = sorted(int(c.split("_")[0]) for c in first.columns if c.endswith("_Pts"))[:weeks]
    meta = first[["ID", "Name", "Pos", "Team", "BV"]].copy()
    n, S = len(meta), len(files)
    pts = np.zeros((S, n, len(gws)))
    for s, f in enumerate(files):
        df = pd.read_csv(f)
        if not (df.ID.values == meta.ID.values).all():
            df = df.set_index("ID").loc[meta.ID].reset_index()
        pts[s] = df[[f"{g}_Pts" for g in gws]].to_numpy()
    return meta, gws, pts


def load_field(bootstrap: Path, meta: pd.DataFrame, proj_total: np.ndarray):
    """Effective-ownership weights per player: ownership + captaincy share."""
    boot = json.loads(bootstrap.read_text())
    own = {e["id"]: float(e["selected_by_percent"]) / 100 for e in boot["elements"]}
    w_own = np.array([own.get(int(i), 0.0) for i in meta.ID])
    # field captaincy: share proportional to own * proj among the top pool
    score = w_own * np.maximum(proj_total, 0)
    top = np.argsort(-score)[:CAPTAIN_POOL]
    cap = np.zeros_like(w_own)
    if score[top].sum() > 0:
        cap[top] = score[top] / score[top].sum()
    return w_own + cap, w_own, cap


def legal_formations() -> list[tuple[int, int, int]]:
    """Every valid outfield split: 1 GK plus 10, within FPL position limits."""
    out = []
    for d in range(LINEUP_MIN["DEF"], LINEUP_MAX["DEF"] + 1):
        for m in range(LINEUP_MIN["MID"], LINEUP_MAX["MID"] + 1):
            f = LINEUP_SIZE - 1 - d - m
            if LINEUP_MIN["FWD"] <= f <= LINEUP_MAX["FWD"]:
                out.append((d, m, f))
    return out


def best_legal_xi(rows: list[int], meta: pd.DataFrame, week_pts: np.ndarray) -> list[int]:
    """
    Highest-scoring XI that is actually a legal FPL formation.

    The previous greedy capped positions at their MAXIMUM but never enforced
    the MINIMUM, so it could field two defenders and five midfielders — a team
    you cannot submit. It flattered every evaluated squad, because it played a
    best-eleven that the rules forbid. With 15 players and eight formations,
    exact enumeration is cheap, so there is no reason to approximate.
    """
    by_pos = {q: sorted((p for p in rows if meta.Pos.iloc[p] == q), key=lambda p: -week_pts[p]) for q in POS_QUOTA}
    if not by_pos["GKP"]:
        return sorted(rows, key=lambda p: -week_pts[p])[:LINEUP_SIZE]
    keeper = by_pos["GKP"][0]
    best, best_total = None, -np.inf
    for d, m, f in legal_formations():
        if len(by_pos["DEF"]) < d or len(by_pos["MID"]) < m or len(by_pos["FWD"]) < f:
            continue
        xi = [keeper] + by_pos["DEF"][:d] + by_pos["MID"][:m] + by_pos["FWD"][:f]
        total = float(week_pts[xi].sum())
        if total > best_total:
            best, best_total = xi, total
    return best or sorted(rows, key=lambda p: -week_pts[p])[:LINEUP_SIZE]


# --------------------------------------------------------------------- pool


def build_pool(meta, pts, w_own):
    proj = pts.mean(axis=0).sum(axis=1)
    value = proj / np.maximum(meta.BV.to_numpy(float), 3.5)
    keep = set(np.argsort(-proj)[:POOL_TOP_EV])
    keep |= set(np.argsort(-value)[:POOL_TOP_VALUE])
    keep |= set(np.where(w_own >= POOL_MIN_OWN / 100)[0])
    # ensure feasibility per position
    for p, q in POS_QUOTA.items():
        idx = np.where(meta.Pos.to_numpy() == p)[0]
        have = [i for i in idx if i in keep]
        if len(have) < q + 2:
            extra = idx[np.argsort(-proj[idx])][: q + 2]
            keep |= set(extra)
    return sorted(keep)


# ------------------------------------------------------------------- solving


def solve_cvar(meta, gws, pts, F, lam, alpha, decay, secs, forced=None):
    """`forced` is a list of ROW indices into `meta` (i.e. pool positions), already validated.

    It used to be a list of raw FPL IDs looked up here with `np.where(...)` behind an
    `if len(hits)` guard, so an ID that resolved to nothing simply lost its constraint and
    the solve carried on as if it had never been asked for. Rows cannot miss.
    """
    import sasoptpy as so

    S, n, W = pts.shape
    d = np.array([decay**i for i in range(W)])

    m = so.Model(name=f"cvar_l{int(lam * 100)}_a{int(alpha * 100)}")
    players = list(range(n))
    weeks = list(range(W))
    x = m.add_variables(players, name="squad", vartype=so.BIN)
    y = m.add_variables(players, weeks, name="lineup", vartype=so.BIN)
    c = m.add_variables(players, weeks, name="captain", vartype=so.BIN)
    # eta is the VaR level, so it lies inside the achievable score range.
    # A -1e6 placeholder let the feasibility-jump heuristic open near 5e5 and
    # climb back, which is wasted search.
    d_arr = np.array([decay**i for i in range(W)])
    best_case = np.zeros(S)
    for w_ in range(W):
        top = np.sort(pts[:, :, w_], axis=1)[:, -LINEUP_SIZE:]
        best_case += d_arr[w_] * (top.sum(axis=1) + top[:, -1])
    eta_hi = float((best_case - F).max())
    eta_lo = float((-F).min())
    pad = 0.05 * max(abs(eta_hi), abs(eta_lo), 1.0)
    eta = m.add_variable(name="eta", lb=eta_lo - pad, ub=eta_hi + pad)
    z = m.add_variables(range(S), name="cvar_slack", lb=0)

    pos = meta.Pos.to_numpy()
    team = meta.Team.to_numpy()
    price = meta.BV.to_numpy(float)

    m.add_constraint(so.expr_sum(x[p] for p in players) == SQUAD_SIZE, name="sq15")
    m.add_constraint(so.expr_sum(price[p] * x[p] for p in players) <= BUDGET, name="budget")
    for q, k in POS_QUOTA.items():
        m.add_constraint(so.expr_sum(x[p] for p in players if pos[p] == q) == k, name=f"pos_{q}")
    for t in sorted(set(team)):
        m.add_constraint(so.expr_sum(x[p] for p in players if team[p] == t) <= CLUB_LIMIT, name=f"club_{t}")
    for w in weeks:
        m.add_constraint(so.expr_sum(y[p, w] for p in players) == LINEUP_SIZE, name=f"xi_{w}")
        m.add_constraint(so.expr_sum(c[p, w] for p in players) == 1, name=f"cap_{w}")
        for q in POS_QUOTA:
            e = so.expr_sum(y[p, w] for p in players if pos[p] == q)
            m.add_constraint(e >= LINEUP_MIN[q], name=f"fmin_{q}_{w}")
            m.add_constraint(e <= LINEUP_MAX[q], name=f"fmax_{q}_{w}")
        for p in players:
            m.add_constraint(y[p, w] <= x[p], name=f"y_in_x_{p}_{w}")
            m.add_constraint(c[p, w] <= y[p, w], name=f"c_in_y_{p}_{w}")
    if forced:
        for p in forced:
            m.add_constraint(x[int(p)] == 1, name=f"force_{int(meta.ID.iloc[p])}")

    # Delta_s as linear expression
    delta = {}
    for s in range(S):
        coef = np.einsum("nw,w->nw", pts[s], d)
        delta[s] = so.expr_sum(coef[p, w] * (y[p, w] + c[p, w]) for p in players for w in weeks) - float(F[s])
        m.add_constraint(z[s] >= eta - delta[s], name=f"cvar_{s}")

    ev_term = so.expr_sum(delta[s] for s in range(S)) / S
    cvar_term = eta - so.expr_sum(z[s] for s in range(S)) / (alpha * S)
    m.set_objective(-((1 - lam) * ev_term + lam * cvar_term), name="obj", sense="N")

    m.export_mps("cvar_model.mps")
    import highspy

    h = highspy.Highs()
    h.setOptionValue("time_limit", secs)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("output_flag", True)
    h.readModel("cvar_model.mps")
    h.run()
    Path("cvar_model.mps").unlink(missing_ok=True)
    sol = h.getSolution()
    info = h.getInfo()
    vals = np.array(sol.col_value)
    names = [h.getColName(i)[1] for i in range(h.getNumCol())]
    idx = {nm: i for i, nm in enumerate(names)}
    squad = [p for p in players if vals[idx[f"squad[{p}]"]] > 0.5]
    caps = {w: next(p for p in players if vals[idx[f"captain[{p},{w}]"]] > 0.5) for w in weeks}
    lineup = {w: [p for p in players if vals[idx[f"lineup[{p},{w}]"]] > 0.5] for w in weeks}
    return squad, lineup, caps, float(info.mip_gap)


# ----------------------------------------------------------------- reporting


def portfolio_delta(pts, d, F, lineup, caps):
    S = pts.shape[0]
    out = np.zeros(S)
    for w, members in lineup.items():
        out += d[w] * pts[:, members, w].sum(axis=1)
        out += d[w] * pts[:, caps[w], w]
    return out - F


def describe(tag, delta, alpha):
    k = max(1, int(np.floor(alpha * len(delta))))
    tail = np.sort(delta)[:k]
    print(
        f"  {tag:14s} E[D]={delta.mean():+7.2f}  sd={delta.std():6.2f}  CVaR{int(alpha * 100)}%={tail.mean():+7.2f}  P(D>0)={np.mean(delta > 0):.2f}"
    )
    return {"mean": float(delta.mean()), "sd": float(delta.std()), "cvar": float(tail.mean()), "p_beat": float(np.mean(delta > 0))}


class SquadError(ValueError):
    """A supplied ID list is not usable — refuse to score or solve on it."""


def resolve_ids(meta: pd.DataFrame, spec: str, label: str, dupe_note: str = "") -> list[int]:
    """
    Rows in `meta` for a comma-separated FPL ID list, or SquadError.

    The checks that apply to any ID list, whatever it is for: integer tokens,
    no repeats, and every ID naming a player the scenario set actually holds.
    Returned in meta order, not input order: a squad's score must not depend on
    the order the IDs were typed (best_legal_xi breaks equal-points ties by list
    position), and this is what the old isin() path returned.
    """
    raw = [t.strip() for t in spec.split(",") if t.strip()]
    if not raw:
        raise SquadError(f"{label}: no IDs supplied")
    try:
        ids = [int(t) for t in raw]
    except ValueError:
        bad = [t for t in raw if not t.lstrip("+-").isdigit()]
        raise SquadError(f"{label}: non-integer ID(s) {bad}") from None
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SquadError(f"{label}: duplicate ID(s) {dupes}{dupe_note}")
    row_of = {int(pid): r for r, pid in enumerate(meta.ID.to_numpy())}
    missing = [i for i in ids if i not in row_of]
    if missing:
        raise SquadError(f"{label}: {len(missing)} ID(s) absent from the scenario set: {missing}")
    return sorted(row_of[i] for i in ids)


def resolve_forced(meta: pd.DataFrame, spec: str, label: str = "--force") -> list[int]:
    """
    Rows in `meta` for a --force ID list, or SquadError.

    --force is not a squad: any number of players is legal and no positional
    quota applies, so the size and split checks below would be inventing a rule.
    What does apply is that every ID must resolve — the old `meta.ID.isin(...)`
    path dropped one that did not, and `solve_cvar` then skipped the constraint
    behind an `if len(hits)` guard, so the user got a solve that had quietly
    ignored the player they asked to force in.
    """
    return resolve_ids(meta, spec, label)


def resolve_squad(meta: pd.DataFrame, spec: str, label: str) -> list[int]:
    """
    Rows in `meta` for a comma-separated 15-man squad, or SquadError.

    --evaluate used to take whatever `meta.ID.isin(ids)` happened to match, so
    one stale or mistyped ID quietly scored a 13-man squad and printed a
    plausible number. Refuse instead, the way solio_enrich refuses to write on
    a thin parse: a confident wrong answer is the worst failure mode here.
    """
    rows = resolve_ids(meta, spec, label, dupe_note=" — a squad cannot own a player twice")
    if len(rows) != SQUAD_SIZE:
        raise SquadError(f"{label}: {len(rows)} players supplied, need exactly {SQUAD_SIZE}")
    pos = meta.Pos.to_numpy()[rows]
    counts = {q: int((pos == q).sum()) for q in POS_QUOTA}
    if counts != POS_QUOTA:
        raise SquadError(f"{label}: illegal split {counts} — FPL requires {POS_QUOTA}")
    check_club_limit(meta, rows, label)
    return rows


def check_club_limit(meta: pd.DataFrame, rows: list[int], label: str) -> None:
    """At most CLUB_LIMIT players per club, or SquadError.

    Unlike the budget, this one is unambiguous at every call site: no price move,
    no transfer history and no chip makes a 4-from-one-club squad legal, so a list
    that trips it is a typo or a hypothetical FPL would refuse to accept.
    """
    team = meta.Team.to_numpy()[rows]
    over = {str(t): int((team == t).sum()) for t in sorted(set(team)) if (team == t).sum() > CLUB_LIMIT}
    if over:
        raise SquadError(f"{label}: {over} — FPL allows at most {CLUB_LIMIT} players per club")


def squad_value(meta: pd.DataFrame, rows: list[int]) -> float:
    """Total BV of a resolved squad, at the prices the scenario set carries."""
    return round(float(meta.BV.to_numpy(float)[rows].sum()), 1)


def parse_named_squads(spec: str) -> dict[str, str]:
    """`name=id,id,...;name=...` -> {name: id-list}. ';' separates squads, '=' names one."""
    out: dict[str, str] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SquadError(f"squad {chunk!r} is not name=ids (use 'held=1,2,...;ev=3,4,...')")
        name, ids = chunk.split("=", 1)
        name = name.strip()
        if not name:
            raise SquadError(f"squad {chunk!r} has an empty name")
        if name in out:
            raise SquadError(f"duplicate squad name {name!r}")
        out[name] = ids
    if len(out) < 2:
        raise SquadError(f"--evaluate-many wants at least two squads, got {len(out)}")
    return out


def evaluate_squad(rows: list[int], meta: pd.DataFrame, pts: np.ndarray, d: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Delta per scenario for a fixed 15: best legal XI and captain chosen on MEAN points."""
    mean_pts = pts.mean(axis=0)
    lineup, caps = {}, {}
    for w in range(pts.shape[2]):
        lineup[w] = best_legal_xi(rows, meta, mean_pts[:, w])
        caps[w] = max(lineup[w], key=lambda p: mean_pts[p, w])
    return portfolio_delta(pts, d, F, lineup, caps)


def compare_paired(name_a: str, delta_a: np.ndarray, name_b: str, delta_b: np.ndarray, shared: int) -> dict:
    """
    Per-scenario difference of two squads scored on the SAME scenarios.

    The field term F cancels exactly in delta_a - delta_b, and so does every
    shared player's contribution week by week, so the only variance left is
    the slots that actually disagree. That is why the paired SE comes out a
    fraction of the ~1.3 each mean carries on its own, and why two squads that
    are individually inside the noise band can still be told apart.
    """
    diff = delta_a - delta_b
    S = len(diff)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(S))
    unpaired = float(np.sqrt(delta_a.var(ddof=1) + delta_b.var(ddof=1)) / np.sqrt(S))
    p_ab = float(np.mean(diff > 0))
    print(
        f"  {name_a} - {name_b}: mean={mean:+7.2f}  pairedSE={se:5.2f}  (unpaired SE {unpaired:5.2f})  "
        f"P({name_a}>{name_b})={p_ab:.2f}  shared {shared}/{SQUAD_SIZE}"
    )
    lead, trail = (name_a, name_b) if mean >= 0 else (name_b, name_a)
    t = abs(mean) / se if se > 0 else np.inf
    if t < 2:
        print(
            f"    VERDICT: inside noise — {abs(mean):.2f} is under 2 paired SE ({2 * se:.2f}); "
            f"'{name_a}' and '{name_b}' are not separable even paired. Decide on team news."
        )
    elif abs(mean) < UNPAIRED_NOISE_BAND:
        print(
            f"    VERDICT: '{lead}' ahead by {abs(mean):.2f} +/- {2 * se:.2f} (2 paired SE, {t:.1f} SE). "
            f"Unpaired this gap sits inside the {UNPAIRED_NOISE_BAND:.1f}pt band and would be called a tie; "
            f"the pairing resolves it — on these scenarios, real but small."
        )
    else:
        print(f"    VERDICT: '{lead}' separates from '{trail}' by {abs(mean):.2f} ({t:.1f} paired SE).")
    return {"mean": mean, "paired_se": se, "unpaired_se": unpaired, "p_a_beats_b": p_ab, "shared": shared}


def main() -> int:
    ap = argparse.ArgumentParser(description="CVaR squad optimiser over scenarios.")
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=Path, required=True)
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--use-scenarios", type=int, default=None)
    ap.add_argument("--lam", type=float, default=0.5, help="0=pure EV, 1=pure tail")
    ap.add_argument("--alpha", type=float, default=0.2, help="tail fraction for CVaR")
    ap.add_argument("--decay", type=float, default=0.87)
    ap.add_argument("--secs", type=float, default=600)
    ap.add_argument("--force", type=str, default=None, help="comma-sep FPL IDs to force in; every ID must resolve")
    ap.add_argument("--evaluate", type=str, default=None, help="comma-sep 15 FPL IDs: evaluate instead of optimise")
    ap.add_argument(
        "--evaluate-many",
        type=str,
        default=None,
        help="'name=ids;name=ids;...': score several squads on the same scenarios and compare them pairwise",
    )
    ap.add_argument("--compare-ev", action="store_true", help="also solve lam=0 and report both")
    args = ap.parse_args()

    meta, gws, pts = load_scenarios(args.scenario_dir, args.weeks, args.use_scenarios)
    proj_total = pts.mean(axis=0).sum(axis=1)
    field_w, w_own, _w_cap = load_field(args.bootstrap, meta, proj_total)
    S = pts.shape[0]
    d = np.array([args.decay**i for i in range(len(gws))])
    F = np.einsum("snw,n,w->s", pts, field_w, d)
    print(f"[cvar] {S} scenarios, {len(meta)} players, GWs {gws[0]}-{gws[-1]}, field EO sum={field_w.sum() * 100:.0f}%")
    print(f"[cvar] field decayed score: mean {F.mean():.1f}, sd {F.std():.1f}")

    if args.evaluate and args.evaluate_many:
        print("[cvar] pass one of --evaluate / --evaluate-many, not both — NOT evaluating", file=sys.stderr)
        return 1

    if args.evaluate or args.evaluate_many:
        try:
            specs = {"supplied": args.evaluate} if args.evaluate else parse_named_squads(args.evaluate_many)
            squads = {nm: resolve_squad(meta, s, nm) for nm, s in specs.items()}
        except SquadError as exc:
            print(f"[cvar] {exc} — NOT evaluating", file=sys.stderr)
            return 1
        # Budget is a WARNING here where the club limit is a failure, and the
        # difference is measured, not assumed. `BV` in a scenario set is each
        # player's price TODAY, while the £100.0m cap binds on what a squad cost
        # when it was BOUGHT, so a squad carried through price rises is
        # legitimately over it — on the GW5 set this pipeline's own EV and
        # stochastic squads value at £100.1m and £101.3m. Nothing in an ID list
        # distinguishes a mid-season holding from a preseason hypothetical, so
        # refusing here would reject real squads; report the number instead and
        # let the user judge. stderr, so the reported metrics stay clean.
        for nm, rows in squads.items():
            value = squad_value(meta, rows)
            if value > BUDGET:
                print(
                    f"[cvar] WARN {nm}: squad value £{value:.1f}m exceeds the £{BUDGET:.1f}m budget — "
                    "expected for a squad held through price rises (BV is today's price), "
                    "but unbuildable as a preseason or wildcard squad.",
                    file=sys.stderr,
                )
        # every squad is scored on the SAME pts array: common random numbers,
        # which is what makes the pairwise differences below meaningful.
        deltas = {nm: evaluate_squad(rows, meta, pts, d, F) for nm, rows in squads.items()}
        if args.evaluate:
            print("\n[cvar] evaluation of supplied squad:")
        else:
            print(f"\n[cvar] evaluation of {len(squads)} squads on the same {S} scenarios:")
        for nm, delta in deltas.items():
            describe(nm, delta, args.alpha)
        if len(deltas) > 1:
            print("\n[cvar] paired comparison (common random numbers; field term cancels in the difference):")
            names = list(deltas)
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    compare_paired(a, deltas[a], b, deltas[b], len(set(squads[a]) & set(squads[b])))
        return 0

    try:
        forced_rows = resolve_forced(meta, args.force) if args.force else []
    except SquadError as exc:
        print(f"[cvar] {exc} — NOT solving", file=sys.stderr)
        return 1

    pool = build_pool(meta, pts, w_own)
    if forced_rows:
        pool = sorted(set(pool) | set(forced_rows))  # forced players must be in the pool
        print(f"[cvar] forcing in: {', '.join(meta.Name.iloc[r] for r in forced_rows)}")
    meta_p = meta.iloc[pool].reset_index(drop=True)
    pts_p = pts[:, pool, :]
    fw_p = field_w[pool]
    forced = [pool.index(r) for r in forced_rows]
    print(f"[cvar] pool {len(pool)} players ({meta_p.Pos.value_counts().to_dict()})")

    results = {}
    lams = [args.lam] + ([0.0] if args.compare_ev and args.lam != 0 else [])
    squads = {}
    for lam in lams:
        print(f"\n[cvar] solving lam={lam} alpha={args.alpha} ...")
        squad, lineup, caps, gap = solve_cvar(meta_p, gws, pts_p, F, lam, args.alpha, args.decay, args.secs, forced)
        delta = portfolio_delta(pts_p, d, F, lineup, caps)
        tag = f"lam={lam}"
        results[tag] = describe(tag, delta, args.alpha)
        results[tag]["gap"] = gap
        squads[tag] = (squad, lineup, caps)
        eo = sum(fw_p[p] for p in squad)
        print(f"  squad EO-weight sum: {eo * 100:.0f}%  (field average ~{field_w.sum() * 100:.0f}%)")
        rows = meta_p.iloc[squad][["ID", "Name", "Pos", "Team", "BV"]].copy()
        rows["own%"] = (w_own[[pool[p] for p in squad]] * 100).round(1)
        rows["GW1cap"] = ["C" if p == caps[0] else "" for p in squad]
        print(rows.sort_values(["Pos", "BV"], ascending=[True, False]).to_string(index=False))

    if len(lams) > 1:
        a, b = squads[f"lam={lams[0]}"][0], squads["lam=0.0"][0]
        diff = set(meta_p.iloc[a].ID) ^ set(meta_p.iloc[b].ID)
        names = meta_p[meta_p.ID.isin(diff)].Name.tolist()
        print(f"\n[cvar] squad difference vs pure-EV: {len(diff) // 2} swaps ({', '.join(names) if names else 'identical'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
