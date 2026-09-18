"""Contract tests for stochastic_solver.py's input validation.

The two-stage model itself is out of scope — it is a scenario-replicated MILP
that takes minutes — so everything here stops at the point where a bad input
would otherwise reach `build_and_solve` and come back as a confident wrong
plan. Stage 1 is the move the user actually executes, which is why these
refusals exist at all.

The helpers duplicate tests/test_cvar_solver.py's on purpose: the two modules
are deliberately standalone and duplicate their own helpers, so their tests do
not share a fixture module either.
"""

from __future__ import annotations

import sys

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
