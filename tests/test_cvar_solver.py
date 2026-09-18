"""Contract tests for cvar_solver.py's input validation.

Scope is the refusal layer, not the MILP: `solve_cvar` imports sasoptpy and
HiGHS and takes minutes, so nothing here solves. What is covered is everything
between an ID list arriving on the command line and a number being printed for
it — the path PROJECT.md §8 Phase 6 records as having once scored a 13-man
squad in silence.

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
