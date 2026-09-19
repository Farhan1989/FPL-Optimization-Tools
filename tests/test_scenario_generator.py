"""Contract tests for scenario_generator.py.

`scenario_generator.py` is the load-bearing file: every CVaR number, every
stochastic stage-1 recommendation and every chip score is computed on the
scenarios it emits. Three claims carry the project and are pinned here:

1. **Mean-preserving spread.** Sampled per-player means must reproduce the
   blended projection. The runbook (§1.3) asks a human to eyeball
   `bias within ±0.05`; these tests assert it, both analytically (exact, no
   Monte-Carlo noise) and empirically through `calibrate_and_report`.
2. **EV conservation under enrichment.** `apply_enrichment` replaces the first
   horizon gameweek's clean-sheet and DefCon probabilities with Solio's
   published values and pushes the resulting delta into the assist rate so
   each player's TOTAL expectation is unchanged. PROJECT.md §4.4: market data
   entering the EV objective would be double-counting. **Two ways in which
   this currently FAILS are documented in the `test_bug_*` cases below.**
3. **Determinism by seed.** runbook §3's out-of-sample methodology is
   `--seed 42` vs `--seed 99`; both must be reproducible.

Everything is synthetic and lives in tmp_path. Nothing reads `data/` or
`scenarios/`, nothing solves, and `no_network` below makes any attempt to
reach the FPL API a test failure — the deadline check's graceful-degradation
path is what the tests exercise.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

import scenario_generator as sg
import solio_enrich
import utils
from scenario_generator import (
    ASSIST_PTS,
    CS_FULL,
    CS_PTS,
    DEFCON_PTS,
    GOAL_PTS,
    apply_enrichment,
    calibrate_and_report,
    check_first_gw_open,
    decompose,
    load_blend,
    sample_scenarios,
)

# The runbook's acceptance test, quoted: "bias should be within ±0.05".
RUNBOOK_BIAS_TOLERANCE = 0.05

# EV conservation is pure float arithmetic — one multiply and one divide per
# player — so the only error it may carry is double rounding, ~1e-16 relative
# on totals of order 10. 1e-12 leaves three orders of headroom and is still
# ten orders tighter than the 0.005 the "%.2f" CSV format could hide.
EV_TOLERANCE = 1e-12

# Scenario count for the Monte-Carlo checks. Enough that sampling noise on the
# calibration statistic is well inside the tolerance it is tested against,
# small enough that the whole file stays in the existing suite's time budget.
CHECK_SCENARIOS = 120

TEAMS = ["ARS", "AVL", "BOU", "BRE", "BHA", "CHE", "CRY", "EVE", "FUL", "LEE", "LIV", "MCI", "MUN", "NEW", "NFO", "SUN", "TOT", "WHU", "WOL", "BUR"]
# Clean sheets are perfectly correlated inside a team, so Monte-Carlo noise on
# the calibration statistic falls with the number of TEAMS, not the number of
# players. Three leagues' worth keeps the bias tests well clear of the ±0.05
# tolerance at a scenario count the suite can afford.
WIDE_TEAMS = TEAMS + [f"{t}2" for t in TEAMS] + [f"{t}3" for t in TEAMS]
SQUAD = [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]
GWS = (5, 6)


def refuse_network(*_args, **_kwargs):
    raise AssertionError("scenario_generator tests must never hit the network")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to reach the network from a test is a test bug.

    `check_first_gw_open` resolves `utils.cached_request` at call time, and
    that helper has its own on-disk cache — so patching `requests` alone would
    let a warm cache answer and make the test depend on machine state. Both
    doors are shut; what the tests then exercise is the documented
    graceful-degradation path (any exception => UNKNOWN, never a failure).
    """
    monkeypatch.setattr(utils, "cached_request", refuse_network)
    monkeypatch.setattr(requests, "get", refuse_network)
    stub = types.ModuleType("requests")
    stub.get = refuse_network
    stub.RequestException = type("RequestException", (Exception,), {})
    monkeypatch.setitem(sys.modules, "requests", stub)


# ------------------------------------------------------------------ fixtures


def write_sources(data_dir: Path, gws=GWS, teams=TEAMS, seed=0) -> Path:
    """A synthetic review.csv / solio.csv pair in the schema load_blend reads.

    Deliberately not a copy of the real data: `data/*.csv` is replaced weekly,
    so nothing here may depend on its values. Teams carry real three-letter
    codes and players a `TEAM_POSn` name because the enrichment contract is
    keyed on both.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, pid = [], 1
    for ti, team in enumerate(teams):
        strength = 0.85 + 0.5 * (ti % 5) / 4
        for pos, n in SQUAD:
            for i in range(n):
                rows.append({"ID": pid, "Name": f"{team}_{pos}{i}", "Pos": pos, "Team": team, "BV": 4.0 + 0.5 * i, "strength": strength})
                pid += 1
    meta = pd.DataFrame(rows)
    base = {"GKP": 4.2, "DEF": 4.6, "MID": 5.4, "FWD": 5.6}
    for src, jitter in (("review", 0.0), ("solio", 1.0)):
        df = meta.drop(columns="strength").copy()
        for g in gws:
            mins = np.clip(rng.normal(74, 18, len(meta)), 0, 90).round(0)
            pts = np.array([base[p] for p in meta.Pos]) * meta.strength.to_numpy() * (mins / 75) + jitter * rng.normal(0, 0.35, len(meta))
            df[f"{g}_Pts"] = np.clip(pts, 0, None).round(2)
            df[f"{g}_xMins"] = mins
        df.to_csv(data_dir / f"{src}.csv", index=False)
    return data_dir


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    return write_sources(tmp_path_factory.mktemp("sources"), teams=WIDE_TEAMS)


@pytest.fixture(scope="module")
def _decomposed(sources):
    blend = load_blend(sources, ["review", "solio"], None)
    return decompose(blend)


@pytest.fixture
def dec(_decomposed):
    """A fresh copy per test: `apply_enrichment` writes into the frame it is given."""
    out = _decomposed.copy()
    out.attrs["gameweeks"] = list(_decomposed.attrs["gameweeks"])
    return out


@pytest.fixture
def small_sources(tmp_path):
    """Four teams, sixty players — enough for a legal-looking run, fast enough
    for the end-to-end `main()` tests."""
    return write_sources(tmp_path / "data", teams=TEAMS[:4], seed=5)


def enrich_file(path: Path, **payload) -> Path:
    payload.setdefault("gameweek", GWS[0])
    payload.setdefault("teams", {})
    payload.setdefault("defcon", {})
    path.write_text(json.dumps(payload))
    return path


def run_main(monkeypatch, argv) -> int:
    monkeypatch.setattr(sys, "argv", ["scenario_generator.py", *argv])
    return sg.main()


def base_argv(data_dir: Path, out: Path, scenarios=4, seed=42) -> list[str]:
    return ["--data-dir", str(data_dir), "--out", str(out), "--scenarios", str(scenarios), "--seed", str(seed)]


# -------------------------------------------------------- the sampler's mean
#
# The EV-conservation and mean-preservation claims are claims about the
# EXPECTATION of `sample_scenarios`, so they need that expectation written
# down independently of the sampler. Every term below is read straight off the
# points accumulation in `sample_scenarios`; `test_analytic_ev_matches_the_sampler`
# checks the two agree to Monte-Carlo accuracy, so a drift in either is caught.


def _poisson_pmf(lam: np.ndarray, ks: np.ndarray) -> np.ndarray:
    return np.exp(-lam[:, None]) * np.power(lam[:, None], ks) / np.array([float(math.factorial(int(k))) for k in ks])


def _e_conceded_penalty(team_cs: np.ndarray) -> np.ndarray:
    """E[floor(C/2)] for C ~ Poisson(-log P(clean sheet)), the sampler's
    goals-conceded deduction for GK/DEF."""
    ks = np.arange(0, 30)
    return (_poisson_pmf(-np.log(np.clip(team_cs, 0.02, 0.98)), ks) * (ks // 2)).sum(axis=1)


def _e_capped_bonus(bonus_ev: np.ndarray) -> np.ndarray:
    """E[min(Poisson(1.06 * bonus_ev), 3)] — bonus is capped at 3 in the
    sampler and the 1.06 is the documented offset for that cap."""
    ks = np.arange(0, 30)
    return (_poisson_pmf(np.clip(bonus_ev * 1.06, 0, None), ks) * np.minimum(ks, 3)).sum(axis=1)


def ev_components(frame: pd.DataFrame, g: int) -> dict[str, np.ndarray]:
    """Per-player expected points of each sampled component, in gameweek g."""
    pos = frame["Pos"].to_numpy()
    cs_pts = np.vectorize(CS_PTS.get)(pos).astype(float)
    goal_pts = np.vectorize(GOAL_PTS.get)(pos).astype(float)
    p60 = frame[f"{g}_p_60"].to_numpy(float)
    p_play = frame[f"{g}_p_play"].to_numpy(float)
    team_cs = np.clip(frame[f"{g}_team_cs"].to_numpy(float), 0.02, 0.98)
    return {
        "appearance": 2 * p60 + (p_play - p60),
        "goals": frame[f"{g}_lam_goal"].to_numpy(float) * p_play * goal_pts,
        "assists": frame[f"{g}_lam_assist"].to_numpy(float) * p_play * ASSIST_PTS,
        "clean_sheet": team_cs * p60 * cs_pts,
        "defcon": frame[f"{g}_p_defcon"].to_numpy(float) * p60 * DEFCON_PTS,
        "saves": np.where(pos == "GKP", frame[f"{g}_saves_ev"].to_numpy(float) * p_play, 0.0),
        "bonus": _e_capped_bonus(frame[f"{g}_bonus_ev"].to_numpy(float)) * p_play,
        "conceded": -np.where(cs_pts >= CS_FULL, _e_conceded_penalty(team_cs) * p60, 0.0),
    }


def analytic_ev(frame: pd.DataFrame, g: int) -> np.ndarray:
    return sum(ev_components(frame, g).values())


def rebalanced_ev(frame: pd.DataFrame, g: int) -> np.ndarray:
    """The three components `apply_enrichment` explicitly balances against one
    another: it moves clean-sheet and DefCon expectation into assists."""
    parts = ev_components(frame, g)
    return parts["clean_sheet"] + parts["defcon"] + parts["assists"]


# ------------------------------------------------------------------ loading


def test_load_blend_averages_the_two_sources(sources):
    blend = load_blend(sources, ["review", "solio"], None)
    review = pd.read_csv(sources / "review.csv").set_index("ID")
    solio = pd.read_csv(sources / "solio.csv").set_index("ID")
    got = blend.set_index("ID")
    for g in GWS:
        expected = (review[f"{g}_Pts"] + solio[f"{g}_Pts"]) / 2
        assert np.allclose(got[f"{g}_Pts"], expected.loc[got.index])


def test_load_blend_records_the_shared_gameweeks(sources):
    assert load_blend(sources, ["review", "solio"], None).attrs["gameweeks"] == list(GWS)


def test_load_blend_truncates_to_the_horizon(sources):
    assert load_blend(sources, ["review", "solio"], 1).attrs["gameweeks"] == [GWS[0]]


def test_load_blend_disagreement_is_zero_when_the_sources_agree(tmp_path):
    """The epistemic tilt is driven by `{g}_disagree`; identical sources must
    produce no tilt at all, or scenarios would carry parameter uncertainty
    that the data does not contain."""
    data = write_sources(tmp_path / "d")
    (data / "solio.csv").write_text((data / "review.csv").read_text())
    blend = load_blend(data, ["review", "solio"], None)
    assert np.allclose(blend[f"{GWS[0]}_disagree"], 0.0)


def test_load_blend_refuses_sources_with_no_shared_gameweek(tmp_path):
    data = write_sources(tmp_path / "a")
    write_sources(tmp_path / "b", gws=(11, 12))
    (data / "solio.csv").write_text((tmp_path / "b" / "solio.csv").read_text())
    with pytest.raises(SystemExit):
        load_blend(data, ["review", "solio"], None)


# ------------------------------------------------------------ decomposition


SAMPLER_COLUMNS = ("lam_goal", "lam_assist", "p_defcon", "saves_ev", "bonus_ev", "p_play", "p_60", "team_cs", "team_goals", "Pts", "xMins")


@pytest.mark.parametrize("suffix", SAMPLER_COLUMNS)
def test_decompose_emits_every_column_the_sampler_reads(dec, suffix):
    for g in GWS:
        assert f"{g}_{suffix}" in dec.columns


def test_decompose_carries_the_gameweeks_attr_through(dec):
    """`decompose` builds its derived columns with a single `pd.concat`, which
    drops `attrs` across mismatched inputs; it restores them explicitly. Every
    downstream function reads `attrs["gameweeks"]`."""
    assert dec.attrs["gameweeks"] == list(GWS)


def test_decompose_keeps_probabilities_inside_their_bounds(dec):
    for g in GWS:
        assert dec[f"{g}_p_play"].between(0, 1).all()
        assert (dec[f"{g}_p_60"] <= dec[f"{g}_p_play"] + 1e-12).all()
        assert dec[f"{g}_p_defcon"].between(0, 0.85).all()
        assert dec[f"{g}_team_cs"].between(0.02, 0.75).all()
        assert (dec[f"{g}_team_goals"] >= sg.TEAM_GOAL_FLOOR - 1e-12).all()


def test_decompose_gives_every_player_on_a_team_the_same_clean_sheet_probability(dec):
    """Clean sheets are a TEAM event; the pooled probability is what makes
    teammates perfectly correlated in the sampler."""
    for g in GWS:
        assert (dec.groupby("Team")[f"{g}_team_cs"].nunique() == 1).all()


# ------------------------------------------------------- mean preservation


def test_analytic_ev_matches_the_sampler(dec):
    """Anchor for every EV assertion below: the expectation written down in
    `ev_components` is the expectation `sample_scenarios` actually realises.

    The threshold is Monte-Carlo noise, not model error — at 1500 scenarios the
    two agree to 0.004, but a suite-sized run of CHECK_SCENARIOS carries about
    ±0.02 on this statistic. Any component silently dropped or double-counted
    moves it by an order of magnitude more.
    """
    total = None
    for _, frame in sample_scenarios(dec, CHECK_SCENARIOS, 4242):
        v = frame[[f"{g}_Pts" for g in GWS]].to_numpy()
        total = v if total is None else total + v
    empirical = total / CHECK_SCENARIOS
    predicted = np.column_stack([analytic_ev(dec, g) for g in GWS])
    assert abs(float(np.mean(empirical - predicted))) < 0.03


def test_decomposition_is_mean_preserving_analytically(dec):
    """No Monte-Carlo noise: the sampler's own expectation must reproduce the
    blended projection it was decomposed from."""
    for g in GWS:
        proj = dec[f"{g}_Pts"].to_numpy(float)
        mask = proj > 1.0
        err = analytic_ev(dec, g)[mask] - proj[mask]
        assert abs(float(err.mean())) < RUNBOOK_BIAS_TOLERANCE


@pytest.mark.parametrize("seed", [42, 99])
def test_sampled_means_track_the_projection_within_the_runbook_tolerance(dec, seed):
    """The printed `[scen] calibration: bias ...` line, asserted. runbook §1.3
    tells the user to check this by eye every week; nothing enforced it.

    Both seeds, because runbook §3's out-of-sample method scores candidate
    squads on `--seed 99`: that is only a fair test if the fresh set is a
    mean-preserving spread of the same projections, not merely a different one.
    """
    diag = calibrate_and_report(dec, CHECK_SCENARIOS, seed)
    assert abs(diag["mean_bias_pts"]) < RUNBOOK_BIAS_TOLERANCE
    assert diag["corr_emp_vs_proj"] > 0.9
    assert diag["players"] == len(dec)
    assert diag["gameweeks"] == len(GWS)
    assert diag["check_scenarios"] == CHECK_SCENARIOS


def test_scenarios_are_a_spread_and_not_a_point_mass(dec):
    """A mean-preserving SPREAD has two halves. Without the second, a
    generator that emitted the projection 200 times would pass calibration and
    make every CVaR figure zero-variance nonsense."""
    g = GWS[0]
    draws = np.stack([frame[f"{g}_Pts"].to_numpy() for _, frame in sample_scenarios(dec, 40, 42)])
    sd = draws.std(axis=0)
    playing = dec[f"{g}_p_play"].to_numpy(float) > 0.5
    assert (sd[playing] > 0.5).all()
    assert draws.max() > 12  # somebody hauls


# ------------------------------------------------------------- determinism


def test_the_same_seed_reproduces_the_scenarios_exactly(dec):
    first = [frame for _, frame in sample_scenarios(dec, 3, 42)]
    second = [frame for _, frame in sample_scenarios(dec, 3, 42)]
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_gives_different_scenarios(dec):
    """runbook §3 scores candidate squads on `--seed 99` precisely because it
    must be a set they have never seen."""
    g = GWS[0]
    a = next(iter(sample_scenarios(dec, 1, 42)))[1][f"{g}_Pts"].to_numpy()
    b = next(iter(sample_scenarios(dec, 1, 99)))[1][f"{g}_Pts"].to_numpy()
    assert not np.array_equal(a, b)


def test_main_is_reproducible_end_to_end(monkeypatch, small_sources, tmp_path):
    runs = []
    for name in ("a", "b"):
        out = tmp_path / name
        assert run_main(monkeypatch, base_argv(small_sources, out)) == 0
        runs.append(sorted(p.name for p in out.glob("scenario_*.csv")))
        assert runs[0] == runs[-1]
    for name in runs[0] + ["summary.csv"]:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_main_with_a_different_seed_writes_different_scenarios(monkeypatch, small_sources, tmp_path):
    for name, seed in (("a", 42), ("b", 99)):
        assert run_main(monkeypatch, base_argv(small_sources, tmp_path / name, seed=seed)) == 0
    assert (tmp_path / "a" / "scenario_000.csv").read_bytes() != (tmp_path / "b" / "scenario_000.csv").read_bytes()
    assert json.loads((tmp_path / "b" / "manifest.json").read_text())["seed"] == 99


# -------------------------------------------------- enrichment: the GW guard


def test_enrichment_applies_when_the_gameweek_matches(dec, tmp_path):
    path = enrich_file(tmp_path / "e.json", gameweek=GWS[0], teams={"ARS": {"cs": 0.55}})
    _, status = apply_enrichment(dec, path)
    assert status == {"state": "APPLIED", "horizon_gw": GWS[0], "teams": 1, "players_cs": 15, "defcon": 0}


def test_enrichment_applies_when_the_file_names_no_gameweek(dec, tmp_path):
    """A hand-written or legacy file with a null gameweek is trusted, not skipped."""
    path = enrich_file(tmp_path / "e.json", gameweek=None, teams={"ARS": {"cs": 0.55}})
    assert apply_enrichment(dec, path)[1]["state"] == "APPLIED"


def test_enrichment_skips_when_the_feed_has_rolled_to_the_next_gameweek(dec, tmp_path, capsys):
    """The Solio feed rolls forward the moment a deadline passes. Applying
    GW6 clean sheets to a GW5 horizon would be worse than not enriching."""
    path = enrich_file(tmp_path / "e.json", gameweek=GWS[0] + 1, teams={"ARS": {"cs": 0.55}})
    _, status = apply_enrichment(dec, path)
    assert status["state"] == "SKIPPED"
    assert status["enrich_gw"] == GWS[0] + 1 and status["horizon_gw"] == GWS[0]
    out = capsys.readouterr().out
    assert "WARN" in out and "enrichment SKIPPED" in out
    assert "INFERRED clean-sheet and DefCon priors" in out


def test_a_skipped_enrichment_changes_not_one_number(dec, tmp_path):
    before = dec.copy()
    path = enrich_file(tmp_path / "e.json", gameweek=GWS[0] + 1, teams={"ARS": {"cs": 0.99}}, defcon={"ARS_DEF0|ARS": 0.9})
    after, _ = apply_enrichment(dec, path)
    pd.testing.assert_frame_equal(before, after)


def test_enrich_strict_names_the_mismatch_on_stderr(dec, tmp_path, capsys):
    path = enrich_file(tmp_path / "e.json", gameweek=GWS[0] + 1)
    _, status = apply_enrichment(dec, path, strict=True)
    assert status["state"] == "SKIPPED"
    err = capsys.readouterr().err
    assert "FAIL" in err and "enrichment REFUSED" in err
    assert "NOT writing scenarios" in err


def test_main_enrich_strict_exits_nonzero_and_writes_nothing(monkeypatch, small_sources, tmp_path):
    """A degraded set that silently reaches the solver is the failure mode
    §8 Phase 6 closed. Strict mode must leave no half-written directory
    for the next glob to pick up."""
    path = enrich_file(tmp_path / "e.json", gameweek=GWS[0] + 1)
    out = tmp_path / "scen"
    argv = [*base_argv(small_sources, out), "--enrich", str(path), "--enrich-strict"]
    assert run_main(monkeypatch, argv) == 1
    assert not out.exists()


def test_main_without_strict_degrades_but_still_writes(monkeypatch, small_sources, tmp_path, capsys):
    path = enrich_file(tmp_path / "e.json", gameweek=GWS[0] + 1)
    out = tmp_path / "scen"
    argv = [*base_argv(small_sources, out), "--enrich", str(path)]
    assert run_main(monkeypatch, argv) == 0
    assert (out / "scenario_000.csv").exists()
    assert json.loads((out / "manifest.json").read_text())["enrichment"]["state"] == "SKIPPED"
    assert "SKIPPED" in capsys.readouterr().out


def test_main_enrich_strict_writes_when_the_gameweek_matches(monkeypatch, small_sources, tmp_path):
    path = enrich_file(tmp_path / "e.json", gameweek=GWS[0], teams={"ARS": {"cs": 0.55}}, defcon={"ARS_DEF0|ARS": 0.6})
    out = tmp_path / "scen"
    argv = [*base_argv(small_sources, out), "--enrich", str(path), "--enrich-strict"]
    assert run_main(monkeypatch, argv) == 0
    assert json.loads((out / "manifest.json").read_text())["enrichment"]["state"] == "APPLIED"


def test_enrichment_actually_reaches_the_sampled_scenarios(monkeypatch, small_sources, tmp_path):
    """Applied to the frame but after the sampler had read it would be a silent
    no-op with an APPLIED banner over it."""
    for name, cs in (("bunker", 0.9), ("leaky", 0.03)):
        path = enrich_file(tmp_path / f"{name}.json", teams={t: {"cs": cs} for t in TEAMS[:4]})
        argv = [*base_argv(small_sources, tmp_path / name), "--enrich", str(path)]
        assert run_main(monkeypatch, argv) == 0
    bunker = pd.read_csv(tmp_path / "bunker" / "scenario_000.csv")
    leaky = pd.read_csv(tmp_path / "leaky" / "scenario_000.csv")
    assert not bunker[f"{GWS[0]}_Pts"].equals(leaky[f"{GWS[0]}_Pts"])


def test_main_honours_the_horizon(monkeypatch, small_sources, tmp_path):
    out = tmp_path / "scen"
    assert run_main(monkeypatch, [*base_argv(small_sources, out), "--horizon", "1"]) == 0
    assert json.loads((out / "manifest.json").read_text())["gameweeks"] == [GWS[0]]
    assert f"{GWS[1]}_Pts" not in pd.read_csv(out / "scenario_000.csv").columns


def test_manifest_records_no_enrichment_when_none_was_asked_for(monkeypatch, small_sources, tmp_path):
    out = tmp_path / "scen"
    assert run_main(monkeypatch, base_argv(small_sources, out)) == 0
    assert json.loads((out / "manifest.json").read_text())["enrichment"] == {"state": "NONE"}


# ----------------------------------------------------- enrichment: contents


def test_enrichment_overrides_the_published_team_clean_sheet(dec, tmp_path):
    path = enrich_file(tmp_path / "e.json", teams={"ARS": {"cs": 0.61}, "BUR": {"cs": 0.08}})
    out, _ = apply_enrichment(dec, path)
    got = out.groupby("Team")[f"{GWS[0]}_team_cs"].first()
    assert got["ARS"] == 0.61
    assert got["BUR"] == 0.08


def test_enrichment_ignores_a_team_entry_with_no_cs(dec, tmp_path):
    """`solio_enrich` emits goals-for/against for teams it could not price a
    clean sheet for; those must not land as a zero."""
    before = dec.groupby("Team")[f"{GWS[0]}_team_cs"].first()["ARS"]
    path = enrich_file(tmp_path / "e.json", teams={"ARS": {"gf": 2.1, "ga": 0.6}})
    out, status = apply_enrichment(dec, path)
    assert status["players_cs"] == 0
    assert out.groupby("Team")[f"{GWS[0]}_team_cs"].first()["ARS"] == before


def test_enrichment_overrides_the_published_defcon_probability(dec, tmp_path):
    path = enrich_file(tmp_path / "e.json", defcon={"ARS_DEF0|ARS": 0.71})
    out, status = apply_enrichment(dec, path)
    assert status["defcon"] == 1
    assert out.loc[out.Name == "ARS_DEF0", f"{GWS[0]}_p_defcon"].item() == 0.71


def test_enrichment_leaves_unlisted_players_and_teams_exactly_alone(dec, tmp_path):
    before = dec.copy()
    path = enrich_file(tmp_path / "e.json", teams={"ARS": {"cs": 0.61}}, defcon={"ARS_DEF0|ARS": 0.71})
    after, _ = apply_enrichment(dec, path)
    untouched = after.Team != "ARS"
    cols = [c for c in after.columns if c.startswith(f"{GWS[0]}_")]
    pd.testing.assert_frame_equal(before.loc[untouched, cols], after.loc[untouched, cols])


def test_enrichment_touches_only_the_first_horizon_gameweek(dec, tmp_path):
    """PROJECT.md §6: the feed carries the current gameweek only. Later weeks
    stay prior-based."""
    before = dec.copy()
    path = enrich_file(tmp_path / "e.json", teams={"ARS": {"cs": 0.61}}, defcon={"ARS_DEF0|ARS": 0.71})
    after, _ = apply_enrichment(dec, path)
    later = [c for c in after.columns if c.startswith(f"{GWS[1]}_")]
    pd.testing.assert_frame_equal(before[later], after[later])


# ------------------------------------------------------- EV CONSERVATION
#
# The claim (apply_enrichment's docstring, PROJECT.md §4.4): enrichment alters
# variance and correlation but NOT expectation. The clean-sheet and DefCon
# deltas are pushed into the assist rate so each player's total is unchanged.
# `rebalanced_ev` is the sum of exactly the three components that mechanism
# balances.


def published(value: float) -> float:
    """Keep a synthetic published probability inside the band the sampler uses
    verbatim: below 0.02 it clips, and `apply_enrichment` would then compensate
    for a clean-sheet change the sampler never makes."""
    return round(max(0.02, min(0.98, value)), 4)


def ev_owed(before: pd.DataFrame, after: pd.DataFrame, g: int) -> np.ndarray:
    """The expectation enrichment freed (+) or took on (-) at the clean-sheet
    and DefCon components — the quantity `apply_enrichment` repays through the
    assist rate."""
    cs_pts = np.vectorize(CS_PTS.get)(before["Pos"].to_numpy()).astype(float)
    p60 = before[f"{g}_p_60"].to_numpy(float)
    moved_cs = (before[f"{g}_team_cs"] - after[f"{g}_team_cs"]).to_numpy(float)
    moved_dc = (before[f"{g}_p_defcon"] - after[f"{g}_p_defcon"]).to_numpy(float)
    return moved_cs * p60 * cs_pts + moved_dc * p60 * DEFCON_PTS


def assist_floor_hit(before: pd.DataFrame, after: pd.DataFrame, g: int) -> np.ndarray:
    """Players for whom the repayment would need a NEGATIVE assist rate, so the
    `np.maximum(..., 0.0)` floor truncates it — the documented bug below."""
    p_play = np.maximum(before[f"{g}_p_play"].to_numpy(float), 1e-6)
    wanted = before[f"{g}_lam_assist"].to_numpy(float) + ev_owed(before, after, g) / (ASSIST_PTS * p_play)
    return wanted < 0.0


def test_ev_is_conserved_when_published_cs_is_lower_than_inferred(dec, tmp_path):
    """Delta flows OUT of clean sheets and INTO assists. Nothing can clip on
    this side, so the claim must hold for every player without exception."""
    g = GWS[0]
    before, inferred = dec.copy(), dec.groupby("Team")[f"{g}_team_cs"].first()["ARS"]
    path = enrich_file(tmp_path / "e.json", teams={"ARS": {"cs": published(inferred / 2)}})
    after, _ = apply_enrichment(dec, path)
    assert not assist_floor_hit(before, after, g).any()
    assert np.abs(rebalanced_ev(after, g) - rebalanced_ev(before, g)).max() < EV_TOLERANCE


def test_ev_is_conserved_when_published_cs_is_higher_than_inferred(dec, tmp_path):
    """Delta flows the other way: assists must give the EV back. It holds for
    every player whose assist rate is large enough to absorb the repayment —
    see `test_bug_assist_floor...` for the players where it is not."""
    g = GWS[0]
    before, inferred = dec.copy(), dec.groupby("Team")[f"{g}_team_cs"].first()["ARS"]
    path = enrich_file(tmp_path / "e.json", teams={"ARS": {"cs": published(inferred + 0.02)}})
    after, _ = apply_enrichment(dec, path)
    absorbed = ~assist_floor_hit(before, after, g)
    assert absorbed.mean() > 0.9
    assert np.abs(rebalanced_ev(after, g) - rebalanced_ev(before, g))[absorbed].max() < EV_TOLERANCE
    # named, so the test cannot pass by absorbing nobody
    assert absorbed[(after.Name == "ARS_MID0").to_numpy()].all()


def test_ev_is_conserved_when_defcon_is_revised_down(dec, tmp_path):
    g = GWS[0]
    before = dec.copy()
    path = enrich_file(tmp_path / "e.json", defcon={f"ARS_DEF{i}|ARS": 0.02 for i in range(5)})
    after, _ = apply_enrichment(dec, path)
    assert not assist_floor_hit(before, after, g).any()
    assert np.abs(rebalanced_ev(after, g) - rebalanced_ev(before, g)).max() < EV_TOLERANCE


def test_ev_conservation_holds_across_many_teams_at_once(dec, tmp_path):
    """The real feed carries all twenty teams and a dozen DefCon players; a
    per-player mechanism must not leak when applied wholesale."""
    g = GWS[0]
    before = dec.copy()
    inferred = dec.groupby("Team")[f"{g}_team_cs"].first()
    teams = {t: {"cs": published(inferred[t] * (0.5 + 0.02 * (i % 10)))} for i, t in enumerate(TEAMS)}
    path = enrich_file(tmp_path / "e.json", teams=teams, defcon={f"{t}_MID0|{t}": 0.05 for t in TEAMS[:6]})
    after, status = apply_enrichment(dec, path)
    assert status["teams"] == 20 and status["defcon"] == 6
    assert not assist_floor_hit(before, after, g).any()
    assert np.abs(rebalanced_ev(after, g) - rebalanced_ev(before, g)).max() < EV_TOLERANCE


# ---------------------------------------------------------- DOCUMENTED BUGS
#
# The two tests below assert CURRENT, INCORRECT behaviour so the suite stays
# green. Both break the EV-conservation claim above, both in the same
# direction — a published clean sheet HIGHER than the inferred one hands the
# affected defenders and keepers free expected points — and that is precisely
# the market-data-into-the-EV-objective double-count PROJECT.md §4.4 forbids.
# If either is fixed, the matching test will fail and should be rewritten to
# assert conservation.


def test_bug_assist_floor_creates_ev_when_the_owed_reduction_exceeds_the_assist_rate(dec, tmp_path):
    """BUG (documented, not fixed).

    `apply_enrichment` conserves EV by `lam_assist += delta / (3 * p_play)`
    under a `np.maximum(..., 0.0)` floor. When enrichment ADDS clean-sheet or
    DefCon expectation, delta is negative — and a goalkeeper's assist rate is
    a rounding error (SHARE["GKP"]["attack"] = 0.02). The floor clips the
    repayment and the player keeps EV he was never projected.

    Measured on the real GW5-GW16 blend against the real `data/enrich.json`:
    the floor binds for 62 of the 274 players projected above a point,
    manufacturing +15.6 points of EV, up to +0.99 on one goalkeeper. Add the
    unbalanced conceded term below and that keeper gains +1.27 on a 3.86
    projection — a 33% uplift handed out by a decomposition step.
    """
    g = GWS[0]
    before = dec.copy()
    inferred = dec.groupby("Team")[f"{g}_team_cs"].first()["ARS"]
    path = enrich_file(tmp_path / "e.json", teams={"ARS": {"cs": published(inferred + 0.45)}})
    after, _ = apply_enrichment(dec, path)
    gained = rebalanced_ev(after, g) - rebalanced_ev(before, g)

    keepers = ((after.Team == "ARS") & (after.Pos == "GKP") & (after[f"{g}_p_60"] > 0.5)).to_numpy()
    assert keepers.any()
    assert assist_floor_hit(before, after, g)[keepers].all(), "the floor should have clipped"
    assert gained[keepers].min() > 0.5, "EV is created, not conserved"
    assert np.abs(gained[(after.Team != "ARS").to_numpy()]).max() < EV_TOLERANCE  # only the enriched team


def test_bug_conceded_penalty_ev_is_not_rebalanced_when_team_cs_moves(dec, tmp_path):
    """BUG (documented, not fixed).

    Team clean-sheet probability enters the sampler TWICE: as the clean-sheet
    bonus, and as `lam_conceded = -log(P(CS))` behind the GK/DEF
    goals-conceded deduction. `decompose` budgets both (see the "conceded-
    penalty budget" block). `apply_enrichment` compensates only the first, so
    overriding a team's clean-sheet probability silently moves every one of
    its defenders' and keepers' totals by the change in E[floor(C/2)].

    Direction is the dangerous one: a published clean sheet HIGHER than the
    inferred one means fewer conceded goals and a smaller deduction, i.e. free
    EV on exactly the market signal that must not enter the EV objective.
    On the real blend against the real `data/enrich.json` the unbalanced term
    runs from -0.50 to +0.29 points per defensive player, 0.06 on average.

    The three balanced components are conserved here — the leak is outside them.
    """
    g = GWS[0]
    before = dec.copy()
    inferred = dec.groupby("Team")[f"{g}_team_cs"].first()["ARS"]
    # A drop, so the assist floor cannot be what moves the total: on this side
    # assists only ever go up.
    path = enrich_file(tmp_path / "e.json", teams={"ARS": {"cs": published(inferred / 3)}})
    after, _ = apply_enrichment(dec, path)
    assert not assist_floor_hit(before, after, g).any()
    assert np.abs(rebalanced_ev(after, g) - rebalanced_ev(before, g)).max() < EV_TOLERANCE

    moved = analytic_ev(after, g) - analytic_ev(before, g)
    defensive = ((after.Team == "ARS") & after.Pos.isin(["GKP", "DEF"]) & (after[f"{g}_p_60"] > 0.5)).to_numpy()
    assert defensive.any()
    assert moved[defensive].max() < -0.1, "conceding more must have cost EV that nothing gave back"

    outfield = ((after.Team == "ARS") & after.Pos.isin(["MID", "FWD"])).to_numpy()
    assert np.abs(moved[outfield]).max() < EV_TOLERANCE


# -------------------------------------------------- the enrichment contract
#
# solio_enrich.py writes the file scenario_generator.py reads. Nothing checks
# that the two agree, and the keys are easy to drift: `e["gameweek"]`,
# `e["teams"][CODE]["cs"]`, and `e["defcon"]` keyed exactly f"{Name}|{TEAM}".


def solio_feed(gameweek=GWS[0]):
    """The shape solio_enrich.parse_feed consumes, with full team names — the
    codes are its job to produce."""
    return {
        "gameweek": gameweek,
        "generatedAt": "2026-08-19T23:29:26.663Z",
        "bestCleanSheets": [
            {"team": "Arsenal", "prGoalsFor": 2.1, "prGoalsAgainst": 0.55, "csProb": 0.5982, "fixtures": [{"opponent": "BUR"}]},
            {"team": "Liverpool", "prGoalsFor": 2.4, "prGoalsAgainst": 0.7, "csProb": 0.4966, "fixtures": []},
        ],
        "topDefCon": [{"name": "ARS_DEF0", "team": "Arsenal", "prDefConProb": 0.6412}],
    }


def test_solio_enrich_output_is_consumable_by_apply_enrichment(dec, tmp_path):
    """The cross-module contract, end to end: parse a feed with solio_enrich
    and hand the result straight to the generator. This fails if EITHER side
    drifts — team-code mapping, the `Name|TEAM` DefCon key, or the gameweek
    field."""
    produced = solio_enrich.parse_feed(solio_feed())
    path = tmp_path / "enrich.json"
    path.write_text(json.dumps(produced))

    out, status = apply_enrichment(dec, path)
    assert status["state"] == "APPLIED"
    cs = out.groupby("Team")[f"{GWS[0]}_team_cs"].first()
    assert cs["ARS"] == 0.5982  # unrounded, as the JSON feed publishes it
    assert cs["LIV"] == 0.4966
    assert cs["BUR"] == round(math.exp(-2.1), 4)  # mirrored opponent, filled by fill_cs
    assert out.loc[out.Name == "ARS_DEF0", f"{GWS[0]}_p_defcon"].item() == 0.6412


def test_the_enrichment_file_carries_the_four_keys_the_generator_reads():
    produced = solio_enrich.parse_feed(solio_feed())
    assert set(produced) == {"gameweek", "generated", "teams", "defcon"}
    assert produced["gameweek"] == GWS[0]
    assert "cs" in produced["teams"]["ARS"]
    assert "ARS_DEF0|ARS" in produced["defcon"]


def test_the_defcon_key_is_name_pipe_team_code_not_the_full_team_name(dec, tmp_path):
    """A silent contract break: a defcon map keyed on "Name|Arsenal" matches
    no row and enriches nothing, exit code 0 and all."""
    before = dec.copy()
    path = enrich_file(tmp_path / "e.json", defcon={"ARS_DEF0|Arsenal": 0.71})
    out, status = apply_enrichment(dec, path)
    assert status["defcon"] == 0
    row = (out.Name == "ARS_DEF0").to_numpy()
    assert out.loc[row, f"{GWS[0]}_p_defcon"].item() == before.loc[row, f"{GWS[0]}_p_defcon"].item()


def test_the_team_map_is_keyed_on_the_three_letter_code(dec, tmp_path):
    path = enrich_file(tmp_path / "e.json", teams={"Arsenal": {"cs": 0.61}})
    _, status = apply_enrichment(dec, path)
    assert status["players_cs"] == 0


# ------------------------------------------------------ the deadline check


def bootstrap(deadlines: dict[int, str]) -> dict:
    return {"events": [{"id": gw, "deadline_time": stamp} for gw, stamp in deadlines.items()]}


def test_deadline_check_degrades_to_unknown_when_the_api_is_unreachable(capsys):
    """Building scenarios offline has to keep working; `no_network` makes the
    call raise, which is the degradation path verbatim."""
    status = check_first_gw_open(5)
    assert status["state"] == "UNKNOWN"
    assert "deadline check skipped" in capsys.readouterr().out


def test_deadline_check_reports_open_before_the_deadline(monkeypatch):
    monkeypatch.setattr(utils, "cached_request", lambda _url: bootstrap({5: "2099-01-01T11:30:00Z"}))
    status = check_first_gw_open(5)
    assert status["state"] == "OPEN" and "still open" in status["summary"]


def test_deadline_check_warns_loudly_when_the_first_gameweek_is_locked(monkeypatch, capsys):
    """PROJECT.md §8 Phase 6: warning-only, deliberately — the `--seed 99`
    out-of-sample workflow legitimately regenerates a locked gameweek."""
    monkeypatch.setattr(utils, "cached_request", lambda _url: bootstrap({5: "2020-01-01T11:30:00Z", 6: "2099-01-01T11:30:00Z"}))
    status = check_first_gw_open(5)
    assert status["state"] == "LOCKED"
    out = capsys.readouterr().out
    assert "WARN" in out and "can no longer be acted on" in out
    assert "next actionable gameweek is GW6" in out


def test_deadline_check_is_unknown_for_a_gameweek_the_calendar_does_not_list(monkeypatch):
    monkeypatch.setattr(utils, "cached_request", lambda _url: bootstrap({9: "2099-01-01T11:30:00Z"}))
    assert check_first_gw_open(5)["state"] == "UNKNOWN"


def test_a_locked_gameweek_never_stops_a_run(monkeypatch, small_sources, tmp_path):
    monkeypatch.setattr(utils, "cached_request", lambda _url: bootstrap({5: "2020-01-01T11:30:00Z", 6: "2099-01-01T11:30:00Z"}))
    out = tmp_path / "scen"
    assert run_main(monkeypatch, base_argv(small_sources, out)) == 0
    assert json.loads((out / "manifest.json").read_text())["first_gw"]["state"] == "LOCKED"


# ------------------------------------------------------------ output shape


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """One real run, shared by the shape tests. Module-scoped, so it patches
    the network door itself rather than relying on the function-scoped
    `no_network` autouse fixture."""
    root = tmp_path_factory.mktemp("written")
    out = root / "scen"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(utils, "cached_request", refuse_network)
        mp.setattr(requests, "get", refuse_network)
        data_dir = write_sources(root / "data", teams=TEAMS[:4], seed=5)
        mp.setattr(sys, "argv", ["scenario_generator.py", *base_argv(data_dir, out, scenarios=4)])
        assert sg.main() == 0
    return out


def test_scenario_files_are_zero_padded_and_sort_in_index_order(written):
    """cvar_solver, stochastic_solver and chip_planner all do
    `sorted(glob("scenario_*.csv"))` and index the result positionally."""
    assert sorted(p.name for p in written.glob("scenario_*.csv")) == [f"scenario_{s:03d}.csv" for s in range(4)]


def test_scenario_files_carry_the_solver_schema_and_nothing_else(written):
    got = pd.read_csv(written / "scenario_000.csv")
    assert list(got.columns) == ["ID", "Name", "Pos", "Team", "BV", *[f"{g}_{t}" for g in GWS for t in ("Pts", "xMins")]]


def test_every_scenario_file_has_the_same_players_in_the_same_order(written):
    """`load_scenarios` reindexes on a mismatch but the fast path assumes
    identical ID order; a silent reordering would be an expensive surprise."""
    ids = pd.read_csv(written / "scenario_000.csv").ID.tolist()
    for path in sorted(written.glob("scenario_*.csv")):
        assert pd.read_csv(path).ID.tolist() == ids


def test_sampled_outcomes_are_physically_possible(written):
    for path in sorted(written.glob("scenario_*.csv")):
        got = pd.read_csv(path)
        for g in GWS:
            assert got[f"{g}_xMins"].between(0, 90).all()
            assert np.isfinite(got[f"{g}_Pts"]).all()
            # a player who did not appear scores nothing
            assert (got.loc[got[f"{g}_xMins"] == 0, f"{g}_Pts"] == 0).all()


def test_summary_reports_the_projection_alongside_the_sampled_spread(written):
    got = pd.read_csv(written / "summary.csv")
    assert list(got.columns) == ["ID", "Name", "Pos", "Team", "proj_total", "scen_mean", "scen_sd", "p10", "p90"]
    assert (got.p10 <= got.p90).all()
    assert (got.scen_sd >= 0).all()


def test_manifest_records_everything_needed_to_reproduce_the_run(written):
    manifest = json.loads((written / "manifest.json").read_text())
    assert manifest["seed"] == 42
    assert manifest["scenarios"] == 4
    assert manifest["gameweeks"] == list(GWS)
    assert manifest["sources"] == ["review", "solio"]
    assert set(manifest) >= {"diagnostics", "enrichment", "first_gw", "priors"}
    assert manifest["priors"]["share"] == sg.SHARE


def test_calibrate_only_reports_without_writing_anything(monkeypatch, small_sources, tmp_path, capsys):
    out = tmp_path / "scen"
    assert run_main(monkeypatch, [*base_argv(small_sources, out), "--calibrate-only"]) == 0
    assert not out.exists()
    assert "calibration: bias" in capsys.readouterr().out


def test_the_run_summary_is_the_last_thing_printed(monkeypatch, small_sources, tmp_path, capsys):
    """A reader who sees only the tail of a 900-line log must still be able to
    tell an enriched set from one that fell back to priors."""
    assert run_main(monkeypatch, base_argv(small_sources, tmp_path / "scen")) == 0
    tail = capsys.readouterr().out.strip().splitlines()[-5:]
    assert any("run summary" in line for line in tail)
    assert any("NOT REQUESTED" in line for line in tail)
