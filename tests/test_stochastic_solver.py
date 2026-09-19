"""Contract tests for stochastic_solver.py's refusal layer.

A real two-stage solve on real data takes minutes, so mostly this stops at the
point where a bad input would otherwise reach `build_and_solve` and come back
as a confident wrong plan. Stage 1 is the move the user actually executes,
which is why these refusals exist at all.

The exception is the `solver infeasibility` section, which does solve. It has
to: the failure it pins is HiGHS's own reporting of an infeasible MILP. Those
models are 15 players, 2 scenarios and 2 gameweeks, and finish in about a tenth
of a second each.

The helpers duplicate tests/test_cvar_solver.py's on purpose: the two modules
are deliberately standalone and duplicate their own helpers, so their tests do
not share a fixture module either.
"""
# ruff: noqa: PLR0913, PLR0917  (write_scenarios, as in tests/test_validate_sources.py)

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import stochastic_solver as ss
from stochastic_solver import CLUB_LIMIT, FT_CAP, SquadError, resolve_squad

LEGAL_POS = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
IDS = list(range(1, 16))


def meta_frame(teams=None, bv=5.0):
    teams = list(teams) if teams is not None else [f"T{i:02d}" for i in range(15)]
    bv = list(bv) if isinstance(bv, (list, tuple)) else [bv] * 15
    return pd.DataFrame({"ID": IDS, "Name": [f"P{i}" for i in IDS], "Pos": LEGAL_POS, "Team": teams, "BV": bv})


def spec(ids=None):
    return ",".join(str(i) for i in (ids or IDS))


# ------------------------------------------------------------- club limit


def test_resolve_squad_accepts_a_legal_holding():
    assert resolve_squad(meta_frame(), spec()) == list(range(15))


def test_resolve_squad_accepts_exactly_three_from_one_club():
    teams = ["ARS", "ARS", "ARS"] + [f"T{i:02d}" for i in range(12)]
    assert resolve_squad(meta_frame(teams=teams), spec()) == list(range(15))


def test_resolve_squad_rejects_four_from_one_club():
    """`club1_<t> <= CLUB_LIMIT` is already a stage-1 constraint, so a holding
    that breaks it cannot be held into stage 1 at all."""
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    with pytest.raises(SquadError) as exc:
        resolve_squad(meta_frame(teams=teams), spec())
    assert "{'ARS': 4}" in str(exc.value)
    assert f"at most {CLUB_LIMIT} players per club" in str(exc.value)


def test_resolve_squad_does_not_check_the_budget():
    """Deliberate: --squad is what the user OWNS, and a held squad's value at
    today's prices legitimately passes £100.0m through price rises. The real GW5
    stochastic squad values at £101.3m."""
    assert resolve_squad(meta_frame(bv=9.0), spec()) == list(range(15))  # £135.0m


# ------------------------------------------------------------ --fts/--itb


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["stochastic_solver.py", *argv])
    return ss.main()


def base_args(tmp_path):
    """An EMPTY scenario dir: load_scenarios would SystemExit on it, so anything
    that returns 1 instead proves the check ran before the expensive load."""
    (tmp_path / "scen").mkdir(exist_ok=True)
    return ["--scenario-dir", str(tmp_path / "scen"), "--preseason"]


@pytest.mark.parametrize("fts", [-1, FT_CAP + 1, 9])
def test_main_refuses_a_free_transfer_count_outside_the_cap(tmp_path, monkeypatch, capsys, fts):
    """`pt1 >= tc1 - fts` hands out every unit of --fts as a hit-free transfer,
    so above the cap the plan's transfer count is not executable."""
    code = run_main(monkeypatch, [*base_args(tmp_path), "--fts", str(fts)])
    err = capsys.readouterr().err
    assert code == 1
    assert f"--fts {fts} is outside 0..{FT_CAP}" in err
    assert "NOT solving" in err


@pytest.mark.parametrize("fts", [0, 1, FT_CAP])
def test_main_accepts_every_reachable_free_transfer_count(tmp_path, monkeypatch, fts):
    """0..5 all pass validation and fall through to the scenario load, which is
    what fails here. 0 is allowed for the same reason dev/solver.py allows it
    (`initial_ft = max(0, ...)`): it only makes the plan more conservative."""
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, [*base_args(tmp_path), "--fts", str(fts)])
    assert "no scenario files" in str(exc.value)


def test_main_refuses_a_negative_bank(tmp_path, monkeypatch, capsys):
    assert run_main(monkeypatch, [*base_args(tmp_path), "--itb", "-0.5"]) == 1
    assert "--itb -0.5 is negative" in capsys.readouterr().err


def test_main_accepts_a_zero_bank(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, [*base_args(tmp_path), "--itb", "0"])
    assert "no scenario files" in str(exc.value)


def test_main_checks_fts_before_itb(tmp_path, monkeypatch, capsys):
    """One message per run, and the first thing wrong is the one reported."""
    assert run_main(monkeypatch, [*base_args(tmp_path), "--fts", "9", "--itb", "-1"]) == 1
    err = capsys.readouterr().err
    assert "--fts 9" in err and "--itb" not in err


def test_main_still_requires_preseason_or_squad(tmp_path, monkeypatch):
    """The pre-existing guard must keep firing ahead of the new ones."""
    (tmp_path / "scen").mkdir(exist_ok=True)
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, ["--scenario-dir", str(tmp_path / "scen"), "--fts", "9"])
    assert "either --preseason or --squad is required" in str(exc.value)


# ------------------------------------------------------------------ --force


def test_resolve_forced_accepts_a_partial_list():
    """--force is not a squad: any number of players, no positional quota."""
    assert ss.resolve_forced(meta_frame(), "1,2,3") == [0, 1, 2]


def test_resolve_forced_accepts_exactly_three_from_one_club():
    teams = ["ARS", "ARS", "ARS"] + [f"T{i:02d}" for i in range(12)]
    assert ss.resolve_forced(meta_frame(teams=teams), "1,2,3") == [0, 1, 2]


def test_resolve_forced_rejects_four_from_one_club():
    """`build_and_solve` posts `club1_<t> <= CLUB_LIMIT` alongside `sq1[p] == 1`
    per forced row, and replicates the club cap per week per scenario, so a
    fourth from one club is infeasible in stage 1 and in every branch."""
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    with pytest.raises(SquadError) as exc:
        ss.resolve_forced(meta_frame(teams=teams), "1,2,3,4")
    assert "--force: {'ARS': 4}" in str(exc.value)
    assert f"at most {CLUB_LIMIT} players per club" in str(exc.value)


def test_resolve_forced_counts_only_the_players_it_was_given():
    """The solver is free to avoid a club the --force list does not name."""
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    assert ss.resolve_forced(meta_frame(teams=teams), "1,2,3,5") == [0, 1, 2, 4]


# --------------------------------------------------- solver infeasibility
#
# The only tests in this file that SOLVE, and they have to: the failure is in
# how HiGHS reports an infeasible MILP. It does not raise, and does not return
# an empty solution either — `value_valid` goes False with `col_value`
# ZERO-FILLED to the right length — so `squad1` came back [] and
# `next(p for p in players if val(f"c1[{p}]") > 0.5)` raised a bare
# StopIteration with an empty message, out of a dict literal.
#
# Infeasible on BUDGET with every club count legal, which is the point: forcing
# four from one club is now refused up front, but a --force list that is legal
# on clubs can still be unbuildable, so (a) does not subsume (b).


def preseason_cfg(**over):
    cfg = {
        "lam": 0.0,
        "alpha": 0.2,
        "secs": 10,
        "gap": 0.005,
        "preseason": True,
        "squad0": [],
        "itb": 0.0,
        "fts": 1,
        "fixed_stage1": None,
        "forced_in": [],
        "max_rt": 1,
    }
    return cfg | over


def toy_pts(n_scen=2, n_gw=2, seed=0):
    return np.random.default_rng(seed).uniform(0, 8, (n_scen, 15, n_gw))


def decay(n_gw=2):
    return np.array([0.87**i for i in range(n_gw)])


def test_build_and_solve_reports_infeasibility_instead_of_raising_stopiteration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # build_and_solve writes two_stage.mps into CWD
    meta = meta_frame(bv=7.0)  # 15 x £7.0m = £105.0m against the £100.0m cap
    with pytest.raises(ss.SolveError) as exc:
        ss.build_and_solve(meta, toy_pts(), np.zeros(2), decay(), preseason_cfg())
    msg = str(exc.value)
    assert "no feasible stage-1 plan" in msg
    assert "Infeasible" in msg  # the status HiGHS actually gave
    assert f"0 of {ss.SQUAD_SIZE} players" in msg


def test_solve_error_is_not_a_squad_error():
    """SquadError means the ID list was never legal. SolveError is the model's
    verdict on a list that passed every check; stage 1 is the move actually
    executed, so the two must not be caught interchangeably."""
    assert not issubclass(ss.SolveError, SquadError)
    assert issubclass(ss.SolveError, RuntimeError)


def test_build_and_solve_points_at_force_when_a_force_list_was_supplied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ss.SolveError) as exc:
        ss.build_and_solve(meta_frame(bv=7.0), toy_pts(), np.zeros(2), decay(), preseason_cfg(forced_in=[0]))
    assert "--force list" in str(exc.value)


def test_build_and_solve_does_not_blame_force_when_none_was_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ss.SolveError) as exc:
        ss.build_and_solve(meta_frame(bv=7.0), toy_pts(), np.zeros(2), decay(), preseason_cfg())
    assert "--force" not in str(exc.value)


# ------------------------------------------------------------ main, solving


def write_scenarios(path, n_scen=4, gws=(5, 6), teams=None, bv=5.0, seed=0):
    path.mkdir(parents=True, exist_ok=True)
    meta = meta_frame(teams=teams, bv=bv)
    rng = np.random.default_rng(seed)
    for s in range(n_scen):
        df = meta.copy()
        for g in gws:
            df[f"{g}_Pts"] = rng.uniform(0, 8, len(df)).round(2)
        df.to_csv(path / f"scenario_{s:03d}.csv", index=False)
    return path


def solve_args(tmp_path, **kw):
    scen = write_scenarios(tmp_path / "scen", **kw)
    return ["--scenario-dir", str(scen), "--preseason", "--weeks", "2", "--use-scenarios", "2", "--secs", "10"]


def test_main_refuses_a_force_list_breaking_the_club_limit(tmp_path, monkeypatch, capsys):
    """Refused before the model is built — the point of checking it up front."""
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    code = run_main(monkeypatch, [*solve_args(tmp_path, teams=teams), "--force", "1,2,3,4"])
    err = capsys.readouterr().err
    assert code == 1
    assert "--force: {'ARS': 4}" in err
    assert "NOT solving" in err


def test_main_reports_an_infeasible_solve_and_exits_one(tmp_path, monkeypatch, capsys):
    """End to end: this used to end in a bare StopIteration traceback."""
    args = solve_args(tmp_path, bv=7.0)  # 15 x £7.0m = £105.0m
    monkeypatch.chdir(tmp_path)
    assert run_main(monkeypatch, args) == 1
    err = capsys.readouterr().err
    assert "no feasible stage-1 plan" in err
    assert "NO PLAN REPORTED" in err


def test_main_still_solves_a_feasible_model(tmp_path, monkeypatch, capsys):
    """The companion: a buildable pool must still come back with a stage-1
    squad, so the new guard cannot be passing by refusing everything."""
    args = solve_args(tmp_path, bv=5.0)  # 15 x £5.0m = £75.0m
    monkeypatch.chdir(tmp_path)
    assert run_main(monkeypatch, args) == 0
    assert "Stage-1 squad (the committed decision)" in capsys.readouterr().out
