"""Contract tests for cvar_solver.py's refusal layer.

Mostly the path between an ID list arriving on the command line and a number
being printed for it — what PROJECT.md §8 Phase 6 records as having once scored
a 13-man squad in silence. A real solve on real data takes minutes, so none of
that is exercised here.

The one exception is the `solver infeasibility` section at the end, which does
solve. It has to: the failure it pins is HiGHS's own reporting of an infeasible
MILP, which no input-validation stand-in can reproduce. Those models are 15
players over 2 gameweeks and finish in about a tenth of a second each.

Everything is synthetic and lives in tmp_path; nothing reads scenarios/.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import cvar_solver as cs
from cvar_solver import CLUB_LIMIT, SquadError, resolve_squad, squad_value

LEGAL_POS = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
IDS = list(range(1, 16))


def meta_frame(teams=None, bv=5.0):
    """15 players in a legal 2/5/5/3 split, one club each unless told otherwise."""
    teams = list(teams) if teams is not None else [f"T{i:02d}" for i in range(15)]
    bv = list(bv) if isinstance(bv, (list, tuple)) else [bv] * 15
    return pd.DataFrame(
        {
            "ID": IDS,
            "Name": [f"P{i}" for i in IDS],
            "Pos": LEGAL_POS,
            "Team": teams,
            "BV": bv,
        }
    )


def spec(ids=None):
    return ",".join(str(i) for i in (ids or IDS))


# ------------------------------------------------------------- club limit


def test_resolve_squad_accepts_a_legal_squad():
    assert resolve_squad(meta_frame(), spec(), "--evaluate") == list(range(15))


def test_resolve_squad_accepts_exactly_three_from_one_club():
    """The boundary: three is legal, and all three real GW5 squads sit on it."""
    teams = ["ARS", "ARS", "ARS"] + [f"T{i:02d}" for i in range(12)]
    assert resolve_squad(meta_frame(teams=teams), spec(), "--evaluate") == list(range(15))


def test_resolve_squad_rejects_four_from_one_club():
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    with pytest.raises(SquadError) as exc:
        resolve_squad(meta_frame(teams=teams), spec(), "--evaluate")
    assert "{'ARS': 4}" in str(exc.value)
    assert f"at most {CLUB_LIMIT} players per club" in str(exc.value)


def test_resolve_squad_names_every_club_over_the_limit():
    teams = ["ARS"] * 4 + ["BHA"] * 5 + [f"T{i:02d}" for i in range(6)]
    with pytest.raises(SquadError) as exc:
        resolve_squad(meta_frame(teams=teams), spec(), "held")
    assert "{'ARS': 4, 'BHA': 5}" in str(exc.value)


def test_club_limit_is_checked_after_the_cheaper_structural_rules():
    """A 14-man list gets the size message, not a club message about a squad
    that was never a squad."""
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    with pytest.raises(SquadError) as exc:
        resolve_squad(meta_frame(teams=teams), spec(IDS[:14]), "--evaluate")
    assert "14 players supplied" in str(exc.value)


# ----------------------------------------------------------- squad value


def test_squad_value_sums_buy_values():
    assert squad_value(meta_frame(bv=5.0), list(range(15))) == 75.0


def test_squad_value_rounds_away_binary_float_noise():
    """Prices come in 0.1 steps, which do not sum exactly in binary."""
    assert squad_value(meta_frame(bv=[4.1] * 15), list(range(15))) == 61.5


# ------------------------------------------------------------------- main


def write_scenarios(path, n_scen=6, gws=(5, 6), teams=None, bv=5.0, seed=0):
    path.mkdir(parents=True, exist_ok=True)
    meta = meta_frame(teams=teams, bv=bv)
    rng = np.random.default_rng(seed)
    for s in range(n_scen):
        df = meta.copy()
        for g in gws:
            df[f"{g}_Pts"] = rng.uniform(0, 8, len(df)).round(2)
        df.to_csv(path / f"scenario_{s:03d}.csv", index=False)
    return path


def write_bootstrap(path):
    path.write_text(json.dumps({"elements": [{"id": i, "selected_by_percent": "10.0"} for i in IDS]}))
    return path


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["cvar_solver.py", *argv])
    return cs.main()


def base_args(tmp_path, **kw):
    scen = write_scenarios(tmp_path / "scen", **kw)
    boot = write_bootstrap(tmp_path / "boot.json")
    return ["--scenario-dir", str(scen), "--bootstrap", str(boot), "--weeks", "2"]


def test_main_evaluates_a_legal_squad(tmp_path, monkeypatch, capsys):
    assert run_main(monkeypatch, [*base_args(tmp_path), "--evaluate", spec()]) == 0
    assert "evaluation of supplied squad" in capsys.readouterr().out


def test_main_refuses_a_squad_breaking_the_club_limit(tmp_path, monkeypatch, capsys):
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    code = run_main(monkeypatch, [*base_args(tmp_path, teams=teams), "--evaluate", spec()])
    err = capsys.readouterr().err
    assert code == 1
    assert "{'ARS': 4}" in err and "NOT evaluating" in err


def test_main_refuses_the_whole_comparison_if_one_squad_breaks_the_club_limit(tmp_path, monkeypatch, capsys):
    """--evaluate-many is a paired comparison: scoring the survivors would print
    a table missing the squad the user asked about."""
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    args = [*base_args(tmp_path, teams=teams), "--evaluate-many", f"a={spec()};b={spec()}"]
    assert run_main(monkeypatch, args) == 1
    assert "NOT evaluating" in capsys.readouterr().err


# The budget is a WARNING, not a refusal: BV is today's price, while the
# £100.0m cap binds on what the squad cost when it was bought. On the real GW5
# set the pipeline's own EV and stochastic squads value at £100.1m and £101.3m.


def test_main_warns_but_still_scores_a_squad_over_budget(tmp_path, monkeypatch, capsys):
    args = [*base_args(tmp_path, bv=7.0), "--evaluate", spec()]  # 15 x 7.0 = 105.0
    assert run_main(monkeypatch, args) == 0
    cap = capsys.readouterr()
    assert "WARN supplied: squad value £105.0m exceeds the £100.0m budget" in cap.err
    assert "held through price rises" in cap.err
    assert "E[D]=" in cap.out  # the metrics are still reported


def test_main_is_silent_about_a_squad_inside_the_budget(tmp_path, monkeypatch, capsys):
    assert run_main(monkeypatch, [*base_args(tmp_path, bv=6.0), "--evaluate", spec()]) == 0
    assert "budget" not in capsys.readouterr().err


def test_main_budget_warning_names_the_squad_it_applies_to(tmp_path, monkeypatch, capsys):
    args = [*base_args(tmp_path, bv=7.0), "--evaluate-many", f"held={spec()};ev={spec()}"]
    assert run_main(monkeypatch, args) == 0
    err = capsys.readouterr().err
    assert "WARN held: squad value £105.0m" in err
    assert "WARN ev: squad value £105.0m" in err


def test_main_does_not_warn_at_exactly_the_budget(tmp_path, monkeypatch, capsys):
    """£100.0m is legal; the check must not fire on float summation noise."""
    bv = [6.7] * 14 + [6.2]  # 100.0 exactly, in prices that do not sum cleanly
    assert run_main(monkeypatch, [*base_args(tmp_path, bv=bv), "--evaluate", spec()]) == 0
    assert "budget" not in capsys.readouterr().err


# ------------------------------------------------------------------ --force


def test_resolve_forced_accepts_a_partial_list():
    """--force is not a squad: any number of players, no positional quota."""
    assert cs.resolve_forced(meta_frame(), "1,2,3") == [0, 1, 2]


def test_resolve_forced_accepts_exactly_three_from_one_club():
    teams = ["ARS", "ARS", "ARS"] + [f"T{i:02d}" for i in range(12)]
    assert cs.resolve_forced(meta_frame(teams=teams), "1,2,3") == [0, 1, 2]


def test_resolve_forced_rejects_four_from_one_club():
    """`solve_cvar` posts `club_<t> <= CLUB_LIMIT` and `x[p] == 1` per forced
    row, so a fourth from one club is infeasible by construction — no choice of
    the other eleven rescues it. Naming the club beats spending --secs to be
    told nothing."""
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    with pytest.raises(SquadError) as exc:
        cs.resolve_forced(meta_frame(teams=teams), "1,2,3,4")
    assert "--force: {'ARS': 4}" in str(exc.value)
    assert f"at most {CLUB_LIMIT} players per club" in str(exc.value)


def test_resolve_forced_counts_only_the_players_it_was_given():
    """The solver is free to avoid a club the --force list does not name."""
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    assert cs.resolve_forced(meta_frame(teams=teams), "1,2,3,5") == [0, 1, 2, 4]


def test_main_refuses_a_force_list_breaking_the_club_limit(tmp_path, monkeypatch, capsys):
    teams = ["ARS"] * 4 + [f"T{i:02d}" for i in range(11)]
    code = run_main(monkeypatch, [*base_args(tmp_path, teams=teams), "--force", "1,2,3,4"])
    err = capsys.readouterr().err
    assert code == 1
    assert "--force: {'ARS': 4}" in err
    assert "NOT solving" in err


# --------------------------------------------------- solver infeasibility
#
# The only tests in this file that actually SOLVE, and they have to: the bug is
# in how HiGHS reports an infeasible MILP, which no amount of input validation
# stands in for. HiGHS does not raise, and does not hand back an empty solution
# either — `value_valid` goes False with `col_value` ZERO-FILLED to the right
# length — so `squad` came back [] and the captain dict comprehension's
# `next(p for p in players if ...)` raised a bare StopIteration with an empty
# message. These models are 15 players over 2 gameweeks and solve in ~0.1s.
#
# Infeasible on BUDGET, with every club count legal: forcing four from one club
# is now refused up front, but a legal --force list can still be unbuildable, so
# the club check does not subsume this.


def infeasible_meta():
    """15 players at £7.0m = £105.0m against the £100.0m cap."""
    return meta_frame(bv=7.0)


def toy_pts(n_scen=4, n_gw=2, seed=0):
    return np.random.default_rng(seed).uniform(0, 8, (n_scen, 15, n_gw))


def test_solve_cvar_reports_infeasibility_instead_of_raising_stopiteration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # solve_cvar writes cvar_model.mps into CWD
    with pytest.raises(cs.SolveError) as exc:
        cs.solve_cvar(infeasible_meta(), [5, 6], toy_pts(), np.zeros(4), 0.5, 0.2, 0.87, 10)
    msg = str(exc.value)
    assert "no feasible squad" in msg
    assert "Infeasible" in msg  # the status HiGHS actually gave
    assert f"0 of {cs.SQUAD_SIZE} players" in msg


def test_solve_error_is_not_a_squad_error():
    """SquadError means the ID list was never legal. SolveError is the model's
    own verdict on a list that passed every check, so the two must not be
    caught interchangeably."""
    assert not issubclass(cs.SolveError, SquadError)
    assert issubclass(cs.SolveError, RuntimeError)


def test_solve_cvar_points_at_force_when_a_force_list_was_supplied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cs.SolveError) as exc:
        cs.solve_cvar(infeasible_meta(), [5, 6], toy_pts(), np.zeros(4), 0.5, 0.2, 0.87, 10, forced=[0])
    assert "--force list" in str(exc.value)


def test_solve_cvar_does_not_blame_force_when_none_was_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cs.SolveError) as exc:
        cs.solve_cvar(infeasible_meta(), [5, 6], toy_pts(), np.zeros(4), 0.5, 0.2, 0.87, 10)
    assert "--force" not in str(exc.value)


def test_main_reports_an_infeasible_solve_and_exits_one(tmp_path, monkeypatch, capsys):
    """End to end: this used to end in a bare StopIteration traceback."""
    args = base_args(tmp_path, bv=7.0)  # 15 x £7.0m = £105.0m
    monkeypatch.chdir(tmp_path)
    assert run_main(monkeypatch, args) == 1
    err = capsys.readouterr().err
    assert "no feasible squad" in err
    assert "NO SQUAD REPORTED" in err


def test_main_still_solves_a_feasible_model(tmp_path, monkeypatch, capsys):
    """The companion to the above: a buildable pool must still come back with a
    squad, so the new guard cannot be passing by refusing everything."""
    args = base_args(tmp_path, bv=5.0)  # 15 x £5.0m = £75.0m
    monkeypatch.chdir(tmp_path)
    assert run_main(monkeypatch, args) == 0
    assert "squad EO-weight sum" in capsys.readouterr().out
