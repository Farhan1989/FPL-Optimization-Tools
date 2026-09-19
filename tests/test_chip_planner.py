"""Contract tests for chip_planner.py.

Covers candidate parsing, plan-log parsing, chip effects in scoring, and the
verdict / noise-band logic that runbook §4 tells the user to trust.

Nothing here runs a solver: `enumerate` is deliberately out of scope (it is a
ProcessPoolExecutor over full MILP solves). The log fixtures reproduce the
stock solver's real printed format — `dev/solver.py` builds the summary and
`run/solve.py` prints it under `textwrap.indent(..., "    ")`.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import pytest

import chip_planner as cp
from chip_planner import combo_name, load_field, load_scenarios, name_index, parse_candidate, parse_plan_log, score_plan

PLAYERS = ["Alpha", "Bravo", "Charlie", "Delta"]


# ------------------------------------------------------------------ helpers


def write_scenarios(path, pts, gws=(1,), names=PLAYERS):
    """pts: array shaped (scenarios, players, gameweeks)."""
    path.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(pts, dtype=float)
    for s in range(pts.shape[0]):
        df = pd.DataFrame(
            {
                "ID": list(range(1, len(names) + 1)),
                "Name": names,
                "Pos": ["M"] * len(names),
                "Team": ["AAA"] * len(names),
            }
        )
        for k, g in enumerate(gws):
            df[f"{g}_Pts"] = pts[s, :, k]
        df.to_csv(path / f"scenario_{s:03d}.csv", index=False)
    return path


def write_plan(path, gw_blocks):
    """gw_blocks: [(gw, chip_or_None, [lineup display strings], [bench display strings])]."""
    out = ["", "", "Solution 1"]
    for gw, chip, lineup, bench in gw_blocks:
        out.append(f"    ** GW {gw}:")
        if chip:
            out.append(f"    CHIP {chip}")
        out.append("    ITB=0.5->0.5, FT=1, PT=0, NT=0")
        out.append("")
        out.append("    Lineup: ")
        out.append("    \t" + ", ".join(lineup))
        out.append("    Bench: ")
        out.append("    \t" + ", ".join(bench))
        out.append("    Lineup xPts: 43.12")
        out.append("")
    path.write_text("\n".join(out))
    return path


def simple_plans(tmp_path, first="Alpha", second="Bravo"):
    """Two single-player plans whose winner is decided entirely by scenarios."""
    plans = tmp_path / "plans"
    plans.mkdir()
    write_plan(plans / "plan_a.log", [(1, None, [f"{first} (5.0, C)"], ["Charlie (1.0)"])])
    write_plan(plans / "plan_b.log", [(1, None, [f"{second} (5.0, C)"], ["Delta (1.0)"])])
    return plans


def split_pts(n_scen, alpha_wins):
    """Alpha scores 10 in the first `alpha_wins` scenarios, Bravo in the rest."""
    pts = np.zeros((n_scen, len(PLAYERS), 1))
    pts[:alpha_wins, 0, 0] = 10.0
    pts[alpha_wins:, 1, 0] = 10.0
    return pts


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["chip_planner.py", *argv])
    return cp.main()


# --------------------------------------------------------- parse_candidate


def test_parse_candidate_full_plan():
    assert parse_candidate("bb:1,fh:3,wc:7") == {"bb": 1, "fh": 3, "wc": 7}


def test_parse_candidate_single_chip():
    assert parse_candidate("tc:12") == {"tc": 12}


def test_parse_candidate_tolerates_surrounding_whitespace():
    assert parse_candidate(" bb:1, fh:3 ,wc:7 ") == {"bb": 1, "fh": 3, "wc": 7}


@pytest.mark.parametrize("chip", ["wc", "bb", "fh", "tc"])
def test_parse_candidate_accepts_every_first_set_chip(chip):
    assert parse_candidate(f"{chip}:5") == {chip: 5}


@pytest.mark.parametrize("text", ["xx:1", "BB:1", "amc:4"])
def test_parse_candidate_rejects_unknown_chips(text):
    with pytest.raises(SystemExit) as exc:
        parse_candidate(text)
    assert "unknown chip" in str(exc.value)


def test_parse_candidate_last_entry_wins_on_a_repeated_chip():
    """Documents current behaviour: a repeated chip is not an error."""
    assert parse_candidate("bb:1,bb:4") == {"bb": 4}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("bb1", "malformed candidate 'bb1'"),
        ("bb:1:2", "malformed candidate 'bb:1:2'"),
        ("bb:", "gameweek '' for chip 'bb'"),
        ("bb:one", "gameweek 'one' for chip 'bb'"),
        ("bb:-1", "gameweek '-1' for chip 'bb'"),
        ("", "empty entry in --candidates"),
        ("bb:1,", "empty entry in --candidates"),
    ],
)
def test_parse_candidate_rejects_malformed_input(text, expected):
    """Malformed syntax gets the same friendly SystemExit an unknown chip name
    gets; it used to surface as a raw ValueError traceback, so the two halves of
    one typo read as two different classes of problem."""
    with pytest.raises(SystemExit) as exc:
        parse_candidate(text)
    assert expected in str(exc.value)
    assert "bb:1,fh:3,wc:7" in str(exc.value)  # every message shows the syntax


# --------------------------------------------- parse_candidate: GW range
#
# The bound is the SEASON, 1..38, not the first-set deadline. 26/27 has TWO
# chip sets (PROJECT.md §4.7): the first expires at the GW19 deadline, the
# second runs after it, and scoring a second-set calendar is an explicit
# workflow (runbook §4). Anything narrower than 1..38 would reject half the
# season's legal plans. The floor is 1 rather than 0 because combo_name spells
# "chip not played" as a falsy gameweek.


@pytest.mark.parametrize("gw", [1, 2, 18, 19, 20, 37, 38])
def test_parse_candidate_accepts_every_gameweek_in_the_season(gw):
    assert parse_candidate(f"bb:{gw}") == {"bb": gw}


def test_parse_candidate_does_not_stop_at_the_first_set_deadline():
    """A 1..19 bound would be defensible only if this tool were first-set only.
    It is not — the second set has to be plannable through the same entry."""
    gw = cp.FIRST_SET_DEADLINE_GW + 1
    assert parse_candidate(f"wc:{gw}") == {"wc": gw}
    assert parse_candidate(f"bb:{cp.LAST_GW}") == {"bb": cp.LAST_GW}


def test_parse_candidate_rejects_gameweek_zero():
    """The dangerous one. `bb:0` parsed, and combo_name — which spells an
    unplayed chip as a falsy gameweek — then dropped it, so a one-key typo
    produced a plan silently missing a bench boost rather than an error."""
    with pytest.raises(SystemExit) as exc:
        parse_candidate("bb:0")
    msg = str(exc.value)
    assert "gameweek 0 for chip 'bb'" in msg
    assert f"outside {cp.FIRST_GW}..{cp.LAST_GW}" in msg
    assert "silently drop the chip" in msg


def test_a_zero_gameweek_really_is_indistinguishable_from_no_chip():
    """Why the floor is 1 and not 0: this is what used to get through."""
    assert combo_name(dict.fromkeys(cp.CHIPS) | {"bb": 0}) == combo_name(dict.fromkeys(cp.CHIPS))


@pytest.mark.parametrize("gw", [39, 44, 100])
def test_parse_candidate_rejects_gameweeks_past_the_end_of_the_season(gw):
    with pytest.raises(SystemExit) as exc:
        parse_candidate(f"bb:{gw}")
    msg = str(exc.value)
    assert f"gameweek {gw} for chip 'bb'" in msg
    assert f"the season has {cp.LAST_GW} gameweeks" in msg


def test_parse_candidate_rejects_a_bad_gameweek_beside_good_ones():
    """One bad entry condemns the whole candidate, not just its own chip."""
    with pytest.raises(SystemExit) as exc:
        parse_candidate("bb:1,fh:0,wc:7")
    assert "gameweek 0 for chip 'fh'" in str(exc.value)


def test_a_negative_gameweek_is_still_a_syntax_error_not_a_range_error():
    """Pins the order: `-1` never reaches the range check because `.isdigit()`
    rejects the token first, so the message stays the syntax one."""
    with pytest.raises(SystemExit) as exc:
        parse_candidate("bb:-1")
    assert "is not a gameweek number" in str(exc.value)


# -------------------------------------------------------------- combo_name


def test_combo_name_sorts_chips_alphabetically():
    assert combo_name({"wc": 7, "bb": 1, "fh": 3, "tc": None}) == "bb1_fh3_wc7"


def test_combo_name_drops_unplayed_chips():
    assert combo_name({"bb": 2, "fh": None, "wc": None, "tc": None}) == "bb2"


def test_combo_name_of_the_empty_combo():
    assert combo_name(dict.fromkeys(cp.CHIPS)) == "nochip"


def test_combo_name_round_trips_a_candidate_string():
    assert combo_name(dict.fromkeys(cp.CHIPS) | parse_candidate("bb:1,fh:3,wc:7")) == "bb1_fh3_wc7"


# ----------------------------------------------------------- parse_plan_log


def test_parse_plan_log_reads_lineup_bench_and_captain(tmp_path):
    log = write_plan(
        tmp_path / "p.log",
        [
            (
                1,
                None,
                ["Raya (4.2)", "Gabriel (5.1), Murillo (4.8)", "Salah (8.9, C), Palmer (6.1, V)", "Haaland (7.7)"],
                ["Sels (3.1), Cash (2.2), Rice (4.0), Wood (3.9)"],
            )
        ],
    )
    plan = parse_plan_log(log)
    assert set(plan) == {1}
    assert plan[1]["lineup"] == ["Raya", "Gabriel", "Murillo", "Salah", "Palmer", "Haaland"]
    assert plan[1]["bench"] == ["Sels", "Cash", "Rice", "Wood"]
    assert plan[1]["captain"] == "Salah"


def test_parse_plan_log_ignores_the_vice_captain_flag(tmp_path):
    log = write_plan(tmp_path / "p.log", [(1, None, ["Salah (8.9, C), Palmer (6.1, V)"], ["Sels (3.1)"])])
    assert parse_plan_log(log)[1]["captain"] == "Salah"


def test_parse_plan_log_reads_chip_lines(tmp_path):
    log = write_plan(tmp_path / "p.log", [(1, "BB", ["Salah (8.9, C)"], ["Sels (3.1)"]), (2, None, ["Salah (8.9, C)"], ["Sels (3.1)"])])
    plan = parse_plan_log(log)
    assert plan[1]["chip"] == "bb"
    assert plan[2]["chip"] is None


def test_parse_plan_log_handles_multiple_gameweeks(tmp_path):
    log = write_plan(
        tmp_path / "p.log",
        [
            (1, None, ["Salah (8.9, C)"], ["Sels (3.1)"]),
            (2, None, ["Haaland (9.1, C)"], ["Cash (2.0)"]),
            (3, None, ["Salah (7.0, C)"], ["Rice (3.0)"]),
        ],
    )
    plan = parse_plan_log(log)
    assert sorted(plan) == [1, 2, 3]
    assert plan[2]["captain"] == "Haaland"


def test_parse_plan_log_reads_only_the_first_solution_block(tmp_path):
    """`--num_iterations > 1` prints several plans; only the first is the plan."""
    p = tmp_path / "p.log"
    write_plan(p, [(1, None, ["Salah (8.9, C)"], ["Sels (3.1)"])])
    p.write_text(
        p.read_text() + "\n\nSolution 2\n    ** GW 1:\n    Lineup: \n    \tHaaland (9.0, C)\n    Bench: \n    \tCash (2.0)\n    Lineup xPts: 11.0\n"
    )
    plan = parse_plan_log(p)
    assert plan[1]["lineup"] == ["Salah"]


def test_parse_plan_log_skips_gameweeks_with_no_lineup(tmp_path):
    p = tmp_path / "p.log"
    p.write_text("Solution 1\n    ** GW 1:\n    ITB=0.5->0.5, FT=1, PT=0, NT=0\n")
    assert parse_plan_log(p) == {}


def test_parse_plan_log_of_an_empty_file(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("")
    assert parse_plan_log(p) == {}


def test_parse_plan_log_ignores_transfer_lines(tmp_path):
    p = tmp_path / "p.log"
    p.write_text(
        "Solution 1\n"
        "    ** GW 1:\n"
        "    ITB=0.5->0.5, FT=1, PT=0, NT=1\n"
        "    Buy 123 - Salah\n"
        "    Sell 456 - Palmer\n"
        "\n"
        "    Lineup: \n"
        "    \tSalah (8.9, C)\n"
        "    Bench: \n"
        "    \tSels (3.1)\n"
        "    Lineup xPts: 12.0\n"
    )
    plan = parse_plan_log(p)
    assert plan[1]["lineup"] == ["Salah"]
    assert plan[1]["bench"] == ["Sels"]


# -------------------------------------------------------------- name_index


def test_name_index_keeps_the_first_row_for_a_duplicated_name():
    meta = pd.DataFrame({"Name": ["Alpha", "Bravo", "Alpha"]})
    assert name_index(meta) == {"Alpha": 0, "Bravo": 1}


# ----------------------------------------------------------- load_scenarios


def test_load_scenarios_reads_every_file_and_derives_gameweeks(tmp_path):
    pts = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    d = write_scenarios(tmp_path / "scen", pts, gws=(5, 6, 7))
    meta, gws, out = load_scenarios(d)
    assert gws == [5, 6, 7]
    assert list(meta.columns) == ["ID", "Name", "Pos", "Team"]
    assert out.shape == (2, 4, 3)
    np.testing.assert_allclose(out, pts)


def test_load_scenarios_realigns_a_file_with_a_different_row_order(tmp_path):
    pts = np.arange(2 * 4 * 1, dtype=float).reshape(2, 4, 1)
    d = write_scenarios(tmp_path / "scen", pts)
    shuffled = pd.read_csv(d / "scenario_001.csv").iloc[::-1]
    shuffled.to_csv(d / "scenario_001.csv", index=False)
    _, _, out = load_scenarios(d)
    np.testing.assert_allclose(out, pts)  # realigned back onto scenario_000's IDs


def test_load_scenarios_refuses_an_empty_directory(tmp_path):
    empty = tmp_path / "scen"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        load_scenarios(empty)
    assert "no scenarios" in str(exc.value)


# -------------------------------------------------------------- score_plan

META = pd.DataFrame({"ID": [1, 2, 3, 4], "Name": PLAYERS, "Pos": ["M"] * 4, "Team": ["AAA"] * 4})


def one_gw_pts(values):
    return np.array(values, dtype=float).reshape(1, len(PLAYERS), 1)


def test_score_plan_counts_the_captain_twice():
    plan = {1: {"lineup": ["Alpha", "Bravo"], "bench": [], "captain": "Alpha", "chip": None}}
    total, missing = score_plan(plan, {}, META, [1], one_gw_pts([10, 3, 0, 0]), 0.87)
    assert total[0] == pytest.approx(23.0)  # 10 + 3 + 10
    assert missing == set()


def test_score_plan_bench_is_excluded_without_bench_boost():
    plan = {1: {"lineup": ["Alpha"], "bench": ["Bravo", "Charlie"], "captain": None, "chip": None}}
    total, _ = score_plan(plan, {}, META, [1], one_gw_pts([10, 3, 4, 0]), 0.87)
    assert total[0] == pytest.approx(10.0)


def test_score_plan_bench_boost_adds_the_bench_that_week():
    plan = {1: {"lineup": ["Alpha"], "bench": ["Bravo", "Charlie"], "captain": None, "chip": None}}
    total, _ = score_plan(plan, {"bb": 1}, META, [1], one_gw_pts([10, 3, 4, 0]), 0.87)
    assert total[0] == pytest.approx(17.0)


def test_score_plan_triple_captain_triples_the_captain():
    plan = {1: {"lineup": ["Alpha", "Bravo"], "bench": [], "captain": "Alpha", "chip": None}}
    total, _ = score_plan(plan, {"tc": 1}, META, [1], one_gw_pts([10, 3, 0, 0]), 0.87)
    assert total[0] == pytest.approx(33.0)  # 10 + 3 + 2x10


def test_score_plan_chip_applies_only_to_its_own_gameweek():
    pts = np.zeros((1, 4, 2))
    pts[0, :, 0] = [10, 3, 4, 0]
    pts[0, :, 1] = [10, 3, 4, 0]
    plan = {
        1: {"lineup": ["Alpha"], "bench": ["Bravo"], "captain": None, "chip": None},
        2: {"lineup": ["Alpha"], "bench": ["Bravo"], "captain": None, "chip": None},
    }
    total, _ = score_plan(plan, {"bb": 2}, META, [1, 2], pts, 1.0)
    assert total[0] == pytest.approx(10.0 + 13.0)


def test_score_plan_manifest_overrides_the_plans_own_chip_line():
    """The manifest is authoritative; the log's CHIP text is only a fallback."""
    plan = {
        1: {"lineup": ["Alpha"], "bench": ["Bravo"], "captain": None, "chip": "bb"},
        2: {"lineup": ["Alpha"], "bench": ["Charlie"], "captain": None, "chip": None},
    }
    pts = np.zeros((1, 4, 2))
    pts[0, :, 0] = [10, 3, 4, 0]
    pts[0, :, 1] = [10, 3, 4, 0]
    total, _ = score_plan(plan, {"bb": 2}, META, [1, 2], pts, 1.0)
    assert total[0] == pytest.approx(10.0 + 14.0)  # bench boost landed in GW2, not GW1


def test_score_plan_falls_back_to_the_logged_chip_when_the_manifest_is_empty():
    plan = {1: {"lineup": ["Alpha"], "bench": ["Bravo"], "captain": None, "chip": "bb"}}
    total, _ = score_plan(plan, {}, META, [1], one_gw_pts([10, 3, 4, 0]), 1.0)
    assert total[0] == pytest.approx(13.0)


def test_score_plan_applies_decay_by_horizon_position_not_gameweek_number():
    pts = np.ones((1, 4, 2))
    plan = {
        7: {"lineup": ["Alpha"], "bench": [], "captain": None, "chip": None},
        8: {"lineup": ["Alpha"], "bench": [], "captain": None, "chip": None},
    }
    total, _ = score_plan(plan, {}, META, [7, 8], pts, 0.5)
    assert total[0] == pytest.approx(1.0 + 0.5)


def test_score_plan_ignores_gameweeks_outside_the_scenario_horizon():
    plan = {
        1: {"lineup": ["Alpha"], "bench": [], "captain": None, "chip": None},
        9: {"lineup": ["Alpha"], "bench": [], "captain": None, "chip": None},
    }
    total, _ = score_plan(plan, {}, META, [1], one_gw_pts([10, 0, 0, 0]), 1.0)
    assert total[0] == pytest.approx(10.0)


def test_score_plan_reports_names_absent_from_the_scenarios():
    plan = {1: {"lineup": ["Alpha", "Nobody"], "bench": [], "captain": "Alpha", "chip": None}}
    total, missing = score_plan(plan, {}, META, [1], one_gw_pts([10, 3, 0, 0]), 1.0)
    assert missing == {"Nobody"}
    assert total[0] == pytest.approx(20.0)  # the unknown player contributes nothing


def test_score_plan_returns_one_value_per_scenario():
    pts = np.zeros((5, 4, 1))
    pts[:, 0, 0] = [1, 2, 3, 4, 5]
    plan = {1: {"lineup": ["Alpha"], "bench": [], "captain": None, "chip": None}}
    total, _ = score_plan(plan, {}, META, [1], pts, 1.0)
    np.testing.assert_allclose(total, [1, 2, 3, 4, 5])


# -------------------------------------------------------------- load_field


def test_load_field_is_zero_without_a_bootstrap():
    np.testing.assert_allclose(load_field(None, META, np.ones((7, 4, 1)), [1], 0.87), np.zeros(7))


def test_load_field_scales_with_ownership(tmp_path):
    boot = tmp_path / "boot.json"
    boot.write_text(json.dumps({"elements": [{"id": i, "selected_by_percent": "50.0"} for i in range(1, 5)]}))
    field = load_field(str(boot), META, np.ones((3, 4, 1)), [1], 1.0)
    assert field.shape == (3,)
    assert (field > 0).all()


# ------------------------------------------------ cmd_score: verdict / noise


def test_score_reports_a_separation_outside_the_noise_band(tmp_path, monkeypatch, capsys):
    plans = simple_plans(tmp_path)
    scen = write_scenarios(tmp_path / "scen", split_pts(10, 6))

    assert run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen)]) == 0
    out = capsys.readouterr().out
    assert "VERDICT: 'plan_a' separates from the field." in out
    assert "60.00" in out and "40.00" in out  # win shares


def test_score_calls_a_tie_inside_the_noise_band(tmp_path, monkeypatch, capsys):
    plans = simple_plans(tmp_path)
    scen = write_scenarios(tmp_path / "scen", split_pts(10, 5))

    assert run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen)]) == 0
    out = capsys.readouterr().out
    assert "inside the noise band" in out
    assert "coin flip — decide on team news" in out


def test_score_noise_band_is_configurable(tmp_path, monkeypatch, capsys):
    """A 20pp separation is decisive at the default 8pp band and a coin flip at 25pp."""
    plans = simple_plans(tmp_path)
    scen = write_scenarios(tmp_path / "scen", split_pts(10, 6))

    run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen), "--noise-band", "25"])
    assert "inside the noise band" in capsys.readouterr().out


def test_score_always_prints_the_deadline_guard(tmp_path, monkeypatch, capsys):
    plans = simple_plans(tmp_path)
    scen = write_scenarios(tmp_path / "scen", split_pts(10, 6))
    run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen)])
    assert f"expire at the GW{cp.FIRST_SET_DEADLINE_GW} deadline" in capsys.readouterr().out


def test_score_win_share_is_unchanged_by_the_field_term(tmp_path, monkeypatch, capsys):
    """The field is common to every plan, so it cannot reorder them."""
    plans = simple_plans(tmp_path)
    scen = write_scenarios(tmp_path / "scen", split_pts(10, 6))
    boot = tmp_path / "boot.json"
    boot.write_text(json.dumps({"elements": [{"id": i, "selected_by_percent": "25.0"} for i in range(1, 5)]}))

    run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen), "--bootstrap", str(boot)])
    out = capsys.readouterr().out
    assert "vs EO field" in out
    assert "VERDICT: 'plan_a' separates from the field." in out


def test_score_applies_chips_from_the_manifest(tmp_path, monkeypatch, capsys):
    """Bench Boost is read from the manifest, not from the log text."""
    plans = tmp_path / "plans"
    plans.mkdir()
    write_plan(plans / "bb1.log", [(1, None, ["Alpha (5.0, C)"], ["Charlie (1.0), Delta (1.0)"])])
    write_plan(plans / "nochip.log", [(1, None, ["Alpha (5.0, C)"], ["Charlie (1.0), Delta (1.0)"])])
    (plans / "manifest.json").write_text(json.dumps({"bb1": {"bb": 1}}))

    pts = np.zeros((4, 4, 1))
    pts[:, 0, 0] = 5.0  # Alpha
    pts[:, 2, 0] = 3.0  # Charlie, benched
    scen = write_scenarios(tmp_path / "scen", pts)

    run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen)])
    out = capsys.readouterr().out
    assert "VERDICT: 'bb1' separates from the field." in out


def test_score_skips_an_unparseable_log(tmp_path, monkeypatch, capsys):
    plans = simple_plans(tmp_path)
    (plans / "garbage.log").write_text("this is not a solver plan\n")
    scen = write_scenarios(tmp_path / "scen", split_pts(10, 6))

    assert run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen)]) == 0
    assert "[skip] garbage.log: no parseable plan" in capsys.readouterr().out


def test_score_warns_about_names_missing_from_the_scenarios(tmp_path, monkeypatch, capsys):
    plans = simple_plans(tmp_path, first="Ghost")
    scen = write_scenarios(tmp_path / "scen", split_pts(10, 6))

    run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen)])
    assert "[warn] plan_a.log: 1 names not in scenarios" in capsys.readouterr().out


def test_score_refuses_when_every_log_fails_to_parse(tmp_path, monkeypatch, capsys):
    """It used to print [skip] per log and then die inside np.vstack([]) with
    'need at least one array to concatenate' — a numpy message about the scorer,
    not about the plans directory that is actually the problem."""
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "a.log").write_text("this is not a solver plan\n")
    (plans / "b.log").write_text("nor is this\n")
    scen = write_scenarios(tmp_path / "scen", split_pts(4, 2))

    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen)])
    assert "none of the 2 .log file(s)" in str(exc.value)
    assert "chip_planner.py enumerate" in str(exc.value)
    assert exc.value.code != 0
    out = capsys.readouterr().out
    assert "[skip] a.log" in out and "[skip] b.log" in out


def test_score_refuses_a_plans_directory_with_no_logs(tmp_path, monkeypatch):
    plans = tmp_path / "plans"
    plans.mkdir()
    scen = write_scenarios(tmp_path / "scen", split_pts(4, 2))
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, ["score", "--plans", str(plans), "--scenario-dir", str(scen)])
    assert "no .log plans" in str(exc.value)


# ---------------------------------------------------------------- tc-table


def test_tc_table_ranks_by_tail_mass_not_mean(tmp_path, monkeypatch, capsys):
    """Charlie has the higher mean; Bravo carries the haul probability. TC is
    an option on the right tail, so Bravo must top the P(>=15) list."""
    n = 20
    pts = np.zeros((n, 4, 1))
    pts[:, 1, 0] = [0.0] * 14 + [20.0] * 6  # Bravo: mean 6, hauls often
    pts[:, 2, 0] = 7.0  # Charlie: mean 7, never hauls
    scen = write_scenarios(tmp_path / "scen", pts)

    assert run_main(monkeypatch, ["tc-table", "--scenario-dir", str(scen), "--top", "4"]) == 0
    out = capsys.readouterr().out
    assert "Best TC weeks by P(haul>=15)" in out
    body = out.split("Best TC weeks by P(haul>=15):")[1]
    assert body.strip().splitlines()[1].split()[1] == "Bravo"


# ------------------------------------------------ grid-flag gameweek bounds
# `parse_candidate` was range-checked, but the --bb/--fh/--wc/--tc grid was
# not, and that gap was not cosmetic: `combo_name` spells "chip not played"
# as a falsy gameweek, so `--bb 0` named its plan `nochip` and collided with
# the genuine no-chip combination — same plans/ filename, same manifest key —
# so one full MILP solve silently overwrote another.


def test_check_gw_accepts_the_whole_season():
    for gw in (cp.FIRST_GW, 19, cp.LAST_GW):
        assert cp.check_gw(gw, "bb", "--bb") == gw


def test_check_gw_rejects_zero_and_names_the_collision():
    with pytest.raises(SystemExit) as exc:
        cp.check_gw(0, "bb", "--bb")
    assert "unplayed chip is spelled" in str(exc.value)
    assert "--bb" in str(exc.value)


@pytest.mark.parametrize("gw", [-1, 39, 44])
def test_check_gw_rejects_out_of_season(gw):
    with pytest.raises(SystemExit) as exc:
        cp.check_gw(gw, "fh", "--fh")
    assert "38 gameweeks" in str(exc.value)


def test_a_zero_grid_value_would_have_collided_with_the_real_nochip_combo():
    """Why the floor is 1 rather than just 'because'."""
    nochip = dict.fromkeys(cp.CHIPS)
    assert cp.combo_name(nochip | {"bb": 0}) == cp.combo_name(nochip)


@pytest.mark.parametrize("chip", ["bb", "fh", "wc", "tc"])
def test_enumerate_validates_every_grid_flag_before_solving(chip, tmp_path):
    """Must fail before any MILP starts — each combination is minutes of work."""
    args = argparse.Namespace(bb=[], fh=[], wc=[], tc=[], candidates=[], plans=str(tmp_path / "plans"), workers=1)
    setattr(args, chip, [0])
    with pytest.raises(SystemExit) as exc:
        cp.cmd_enumerate(args)
    assert f"--{chip}" in str(exc.value)
    assert not (tmp_path / "plans").exists()
