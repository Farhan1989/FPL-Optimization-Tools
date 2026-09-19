"""Contract tests for validate_sources.py — the gate that stops bad data
reaching a solve.

All frames here are small and synthetic. Nothing reads data/, nothing writes
outside tmp_path, nothing touches the network.
"""
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

import sys

import pandas as pd
import pytest

import validate_sources as vs
from validate_sources import Report, check_blend, check_contiguity, check_decay, check_fixtures, check_xmins_multiplier, gameweeks

DEFAULT_GWS = (1, 2, 3, 4, 5, 6)


def _val(spec, i, k, gw):
    return float(spec(i, k, gw)) if callable(spec) else float(spec)


def source(n_players=40, gws=DEFAULT_GWS, pts=5.0, xmins=90.0, team_of=None, id_offset=1):
    """Build a projection frame in the solver's column schema.

    `pts` / `xmins` accept a constant or a callable (player_index, gw_index, gw).
    """
    ids = list(range(id_offset, id_offset + n_players))
    cols = {
        "Pos": ["M"] * n_players,
        "ID": ids,
        "Name": [f"P{i}" for i in ids],
        "BV": [5.0] * n_players,
        "SV": [5.0] * n_players,
        "Team": [team_of(i) if team_of else f"T{i % 4}" for i in range(n_players)],
    }
    for k, gw in enumerate(gws):
        cols[f"{gw}_xMins"] = [_val(xmins, i, k, gw) for i in range(n_players)]
        cols[f"{gw}_Pts"] = [_val(pts, i, k, gw) for i in range(n_players)]
    return pd.DataFrame(cols)


def blended(df, gws, factor):
    """A mixed frame derived from `df` with each listed GW's Pts scaled."""
    m = df.copy()
    for gw in gws:
        m[f"{gw}_Pts"] = m[f"{gw}_Pts"] * factor
    return m


# ------------------------------------------------------------------- Report


def test_report_starts_clean():
    rep = Report()
    assert rep.failures == [] and rep.warnings == []


def test_report_ok_records_nothing(capsys):
    rep = Report()
    rep.ok("all good")
    assert rep.failures == [] and rep.warnings == []
    assert "  ok    all good" in capsys.readouterr().out


def test_report_warn_and_fail_accounting(capsys):
    rep = Report()
    rep.warn("w1")
    rep.fail("f1")
    rep.warn("w2")
    out = capsys.readouterr().out
    assert rep.warnings == ["w1", "w2"]
    assert rep.failures == ["f1"]
    assert "  WARN  w1" in out and "  FAIL  f1" in out


# ---------------------------------------------------------------- gameweeks


def test_gameweeks_sorts_numerically_not_lexicographically():
    df = pd.DataFrame({"10_Pts": [1], "2_Pts": [1], "1_Pts": [1]})
    assert gameweeks(df) == [1, 2, 10]


def test_gameweeks_ignores_non_pts_columns():
    df = pd.DataFrame({"ID": [1], "3_xMins": [1], "3_Pts": [1], "Name": ["x"]})
    assert gameweeks(df) == [3]


def test_gameweeks_empty_when_no_pts_columns():
    assert gameweeks(pd.DataFrame({"ID": [1], "Name": ["x"]})) == []


def test_gameweeks_reads_columns_not_rows():
    """Column-derived, so an empty frame still reports its horizon — this is
    what lets main() pass `dfs[0].iloc[0:0]` as a placeholder mixed frame."""
    df = source(gws=(5, 6, 7))
    assert gameweeks(df.iloc[0:0]) == [5, 6, 7]


# --------------------------------------------------------------- check_decay


def test_check_decay_passes_on_an_undecayed_source():
    rep = Report()
    check_decay("review", source(), rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_decay_passes_just_above_the_floor():
    """Review's measured 0.9945/GW is noise, not decay (PROJECT.md §4.2)."""
    rep = Report()
    check_decay("review", source(pts=lambda i, k, gw: 5.0 * 0.995**k), rep)
    assert rep.failures == []


def test_check_decay_fails_on_a_pre_decayed_source():
    """The decay_solio.py bug: a 0.97/GW multiplier baked into a source."""
    rep = Report()
    check_decay("solio", source(pts=lambda i, k, gw: 5.0 * 0.97**k), rep)
    assert len(rep.failures) == 1
    assert "pre-decayed" in rep.failures[0]
    assert "Remove source decay" in rep.failures[0]


def test_check_decay_skips_short_horizons():
    rep = Report()
    check_decay("solio", source(gws=(1, 2, 3)), rep)
    assert rep.failures == []
    assert rep.warnings == ["solio: only 3 gameweeks, decay test skipped"]


def test_check_decay_warns_when_too_few_players_per_gameweek():
    rep = Report()
    check_decay("solio", source(n_players=10), rep)
    assert rep.failures == []
    assert "too few usable gameweeks" in rep.warnings[0]


def test_check_decay_ignores_players_outside_the_single_fixture_xmins_window():
    """Doubles (>100) and blanks (<60) must not drag the pts/90 trend."""
    rep = Report()
    df = source(n_players=60, pts=lambda i, k, gw: 12.0 if i >= 40 else 5.0, xmins=lambda i, k, gw: 140.0 if i >= 40 else 90.0)
    check_decay("solio", df, rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_decay_is_blind_to_decay_applied_to_pts_and_xmins_together():
    """pts/90 stays flat, so check_decay cannot see it — that is exactly why
    check_xmins_multiplier exists (PROJECT.md §4.2)."""
    rep = Report()
    df = source(pts=lambda i, k, gw: 5.0 * 0.97**k, xmins=lambda i, k, gw: 90.0 * 0.97**k)
    check_decay("solio", df, rep)
    assert rep.failures == []

    rep2 = Report()
    check_xmins_multiplier("solio", df, rep2)
    assert len(rep2.failures) == 1
    assert "uniform multiplier on minutes" in rep2.failures[0]


# --------------------------------------------------- check_xmins_multiplier


def test_check_xmins_multiplier_accepts_dispersed_decline():
    """Review's real minutes modelling: median declines but the ratio is
    dispersed and some players rise."""
    rep = Report()
    df = source(n_players=40, xmins=lambda i, k, gw: 90.0 if k == 0 else [40.0, 60.0, 95.0, 88.0][i % 4])
    check_xmins_multiplier("review", df, rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_xmins_multiplier_accepts_flat_minutes():
    rep = Report()
    check_xmins_multiplier("review", source(), rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_xmins_multiplier_warns_on_a_thin_sample():
    rep = Report()
    check_xmins_multiplier("solio", source(n_players=5), rep)
    assert rep.failures == []
    assert "too few players for xMins-multiplier test" in rep.warnings[0]


# ------------------------------------------------------------ check_fixtures


def two_teams(i):
    return "AAA" if i < 20 else "BBB"


def test_check_fixtures_passes_when_structure_matches():
    rep = Report()
    a = source(gws=(1, 2), team_of=two_teams)
    b = source(gws=(1, 2), team_of=two_teams)
    check_fixtures(["solio", "review"], [a, b], rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_fixtures_fails_on_a_blank_disagreement():
    """One source thinks a team has no fixture; the other does not. Blending
    across that manufactures half-existing players."""
    rep = Report()
    a = source(gws=(1, 2), team_of=two_teams)
    b = source(gws=(1, 2), team_of=two_teams, xmins=lambda i, k, gw: 0.0 if (gw == 2 and i < 20) else 90.0)
    check_fixtures(["solio", "review"], [a, b], rep)
    assert len(rep.failures) == 1
    assert "fixture structure differs in 1/2 gameweeks" in rep.failures[0]


def test_check_fixtures_fails_on_a_double_disagreement():
    rep = Report()
    a = source(gws=(1, 2), team_of=two_teams)
    b = source(gws=(1, 2), team_of=two_teams, xmins=lambda i, k, gw: 160.0 if (gw == 2 and i >= 20) else 90.0)
    check_fixtures(["solio", "review"], [a, b], rep)
    assert len(rep.failures) == 1
    assert "fixture structure differs" in rep.failures[0]


def test_check_fixtures_tolerates_fringe_player_minute_differences():
    """Structure, not individual minutes: a benchwarmer projected 0 by one
    source and 3 by the other is squad-depth modelling, not a calendar clash."""
    rep = Report()
    a = source(gws=(1, 2), team_of=two_teams)
    b = source(gws=(1, 2), team_of=two_teams, xmins=lambda i, k, gw: 0.0 if i % 20 == 0 else 90.0)
    check_fixtures(["solio", "review"], [a, b], rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_fixtures_reports_near_zero_minute_noise_as_info(capsys):
    rep = Report()
    a = source(gws=(1, 2), team_of=two_teams)
    b = source(gws=(1, 2), team_of=two_teams, xmins=lambda i, k, gw: 0.0 if i % 20 == 0 else 90.0)
    check_fixtures(["solio", "review"], [a, b], rep)
    assert "players/GW differ on near-zero minutes" in capsys.readouterr().out


def test_check_fixtures_warns_when_horizons_do_not_overlap():
    rep = Report()
    check_fixtures(["solio", "review"], [source(gws=(1, 2)), source(gws=(8, 9))], rep)
    assert rep.failures == []
    assert rep.warnings == ["no overlapping gameweeks between sources"]


def test_check_fixtures_only_compares_shared_gameweeks():
    rep = Report()
    a = source(gws=(1, 2, 3), team_of=two_teams)
    b = source(gws=(1, 2), team_of=two_teams)
    check_fixtures(["solio", "review"], [a, b], rep)
    assert rep.failures == []


# --------------------------------------------------------------- check_blend


def test_check_blend_passes_when_a_solo_gameweek_is_not_halved():
    """PROJECT.md §4.6: read_mixed renormalises per gameweek by a summed
    weight, so a GW only Solio publishes must arrive at full value."""
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    check_blend(["solio", "review"], [a, b], blended(a, [3], 1.0), rep)
    assert rep.failures == []


def test_check_blend_fails_when_a_solo_gameweek_is_halved():
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    check_blend(["solio", "review"], [a, b], blended(a, [3], 0.5), rep)
    assert len(rep.failures) == 1
    assert "GW3 exists only in solio" in rep.failures[0]
    assert "HALVED, weights not normalised" in rep.failures[0]


def test_check_blend_fails_when_a_solo_gameweek_is_inflated():
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    check_blend(["solio", "review"], [a, b], blended(a, [3], 1.5), rep)
    assert len(rep.failures) == 1
    assert "inflated" in rep.failures[0]


@pytest.mark.parametrize("factor", [0.80, 1.20])
def test_check_blend_tolerates_small_deviations(factor):
    """The band is 0.75-1.25: real blends never land exactly on 1.000."""
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    check_blend(["solio", "review"], [a, b], blended(a, [3], factor), rep)
    assert rep.failures == []


def test_check_blend_skips_solo_gameweeks_with_too_small_a_sample():
    """Below MIN_SOLO_SAMPLE the ratio is not evidence either way."""
    rep = Report()
    a = source(n_players=8, gws=(1, 2, 3))
    b = source(n_players=8, gws=(1, 2))
    check_blend(["solio", "review"], [a, b], blended(a, [3], 0.5), rep)
    assert rep.failures == []


def test_check_blend_ignores_gameweeks_the_blend_does_not_carry():
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    mixed = blended(a, [3], 0.5).drop(columns=["3_Pts", "3_xMins"])
    check_blend(["solio", "review"], [a, b], mixed, rep)
    assert rep.failures == []


def test_check_blend_fails_when_the_mixed_frame_has_pts_but_no_xmins():
    """A blend carrying points with no minutes is malformed, and this gate must
    say so rather than raise. It used to KeyError on `{gw}_xMins_s` — a name that
    exists only because the column collides with one in `mixed` during the merge —
    which told the user nothing about whether the data or the validator was broken."""
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    mixed = blended(a, [3], 1.0).drop(columns=["3_xMins"])
    check_blend(["solio", "review"], [a, b], mixed, rep)
    assert len(rep.failures) == 1
    assert "1 gameweek(s) carry Pts but no xMins (3_xMins)" in rep.failures[0]
    assert "malformed blend, unusable downstream" in rep.failures[0]


def test_check_blend_names_every_gameweek_missing_xmins():
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    mixed = blended(a, [3], 1.0).drop(columns=["1_xMins", "3_xMins"])
    check_blend(["solio", "review"], [a, b], mixed, rep)
    assert "2 gameweek(s) carry Pts but no xMins (1_xMins, 3_xMins)" in rep.failures[0]


def test_check_blend_skips_the_halving_ratio_for_a_gameweek_with_no_xmins():
    """The missing column is reported once; the ratio it cannot compute is not
    also reported as a halving, and the single-source player checks still run."""
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    mixed = blended(a, [3], 0.5).drop(columns=["3_xMins"])
    check_blend(["solio", "review"], [a, b], mixed, rep)
    assert len(rep.failures) == 1
    assert "HALVED" not in rep.failures[0]


def test_check_blend_accepts_a_mixed_frame_carrying_both_columns():
    """The regression guard for the fail above: the normal shape must stay silent."""
    rep = Report()
    a = source(gws=(1, 2, 3))
    b = source(gws=(1, 2))
    check_blend(["solio", "review"], [a, b], blended(a, [3], 1.0), rep)
    assert rep.failures == []


def test_check_blend_silent_when_no_gameweek_is_solo():
    rep = Report()
    a = source(gws=(1, 2))
    b = source(gws=(1, 2))
    check_blend(["solio", "review"], [a, b], blended(a, [1, 2], 0.5), rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_blend_warns_about_single_source_players_with_real_projections(capsys):
    rep = Report()
    a = source(n_players=40, gws=(1, 2))
    b = source(n_players=40, gws=(1, 2), id_offset=11)  # ids 11..50 vs 1..40
    check_blend(["solio", "review"], [a, b], a.iloc[0:0], rep)
    assert rep.failures == []
    assert len(rep.warnings) == 2
    assert "players projected by solio ONLY" in rep.warnings[0]
    assert "no cross-check" in rep.warnings[0]
    assert "pts over the shared horizon" in capsys.readouterr().out


def test_check_blend_single_source_players_at_zero_points_are_harmless():
    rep = Report()
    a = source(n_players=40, gws=(1, 2), pts=0.0)
    b = source(n_players=40, gws=(1, 2), pts=0.0, id_offset=11)
    check_blend(["solio", "review"], [a, b], a.iloc[0:0], rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_blend_reports_placeholder_ids(capsys):
    rep = Report()
    a = source(n_players=40, gws=(1, 2))
    a.loc[0, "ID"] = 99999
    b = a.copy()
    check_blend(["solio", "review"], [a, b], a.iloc[0:0], rep)
    assert "placeholder IDs >=10000" in capsys.readouterr().out


# ----------------------------------------------------------- load() and main


def test_load_strips_a_byte_order_mark(tmp_path):
    """The real solio.csv arrives with a UTF-8 BOM."""
    p = tmp_path / "solio.csv"
    p.write_text("ID,Name\n1,Alpha\n", encoding="utf-8-sig")
    assert list(vs.load(p).columns) == ["ID", "Name"]


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["validate_sources.py", *argv])
    return vs.main()


def write_sources(tmp_path, **frames):
    for name, df in frames.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


def test_main_exits_zero_on_clean_sources(tmp_path, monkeypatch, capsys):
    write_sources(tmp_path, solio=source(team_of=two_teams), review=source(team_of=two_teams))
    assert run_main(monkeypatch, ["--sources", "solio", "review", "--data-dir", str(tmp_path)]) == 0
    assert "VALIDATION PASSED — 0 warning(s)" in capsys.readouterr().out


def test_main_exits_one_on_a_pre_decayed_source(tmp_path, monkeypatch, capsys):
    write_sources(
        tmp_path,
        solio=source(team_of=two_teams, pts=lambda i, k, gw: 5.0 * 0.97**k),
        review=source(team_of=two_teams),
    )
    assert run_main(monkeypatch, ["--sources", "solio", "review", "--data-dir", str(tmp_path)]) == 1
    assert "VALIDATION FAILED — 1 failure(s)" in capsys.readouterr().out


def test_main_exits_one_when_a_source_file_is_missing(tmp_path, monkeypatch, capsys):
    assert run_main(monkeypatch, ["--sources", "nope", "--data-dir", str(tmp_path)]) == 1
    assert "missing source file" in capsys.readouterr().out


def test_main_strict_promotes_warnings_to_failure(tmp_path, monkeypatch):
    """A 3-GW source only warns (decay test skipped), so it gates only under --strict."""
    write_sources(tmp_path, solio=source(gws=(1, 2, 3), team_of=two_teams))
    base = ["--sources", "solio", "--data-dir", str(tmp_path)]
    assert run_main(monkeypatch, base) == 0
    assert run_main(monkeypatch, [*base, "--strict"]) == 1


def test_main_fails_on_a_halved_blend(tmp_path, monkeypatch, capsys):
    a = source(gws=(1, 2, 3), team_of=two_teams)
    b = source(gws=(1, 2), team_of=two_teams)
    write_sources(tmp_path, solio=a, review=b)
    blended(a, [3], 0.5).to_csv(tmp_path / "mixed.csv", index=False)

    code = run_main(monkeypatch, ["--sources", "solio", "review", "--data-dir", str(tmp_path), "--mixed", "mixed.csv"])
    assert code == 1
    assert "HALVED" in capsys.readouterr().out


def test_main_gates_a_mixed_file_with_no_xmins_columns(tmp_path, monkeypatch, capsys):
    """End to end: the gate reports it and exits 1, instead of a KeyError traceback
    the user cannot tell from a broken validator (runbook §1.2)."""
    a = source(gws=(1, 2, 3), team_of=two_teams)
    b = source(gws=(1, 2), team_of=two_teams)
    write_sources(tmp_path, solio=a, review=b)
    blended(a, [3], 1.0).drop(columns=["3_xMins"]).to_csv(tmp_path / "mixed.csv", index=False)

    code = run_main(monkeypatch, ["--sources", "solio", "review", "--data-dir", str(tmp_path), "--mixed", "mixed.csv"])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL  mixed: 1 gameweek(s) carry Pts but no xMins (3_xMins)" in out
    assert "VALIDATION FAILED — 1 failure(s)" in out


def test_main_warns_when_the_mixed_file_is_absent(tmp_path, monkeypatch, capsys):
    write_sources(tmp_path, solio=source(team_of=two_teams), review=source(team_of=two_teams))
    code = run_main(monkeypatch, ["--sources", "solio", "review", "--data-dir", str(tmp_path), "--mixed", "gone.csv"])
    assert code == 0
    assert "not found, blend checks skipped" in capsys.readouterr().out


def test_main_warns_when_the_bootstrap_is_absent(tmp_path, monkeypatch, capsys):
    write_sources(tmp_path, solio=source(team_of=two_teams), review=source(team_of=two_teams))
    code = run_main(
        monkeypatch,
        ["--sources", "solio", "review", "--data-dir", str(tmp_path), "--bootstrap", str(tmp_path / "gone.json")],
    )
    assert code == 0
    assert "availability checks skipped" in capsys.readouterr().out


# -------------------------------------------------------- check_contiguity


def test_check_contiguity_passes_an_unbroken_run():
    rep = Report()
    check_contiguity("review", source(gws=(5, 6, 7, 8)), rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_contiguity_does_not_require_the_run_to_start_at_one():
    """A mid-season export legitimately starts at the next gameweek, so the run
    is not pinned to any particular first number — only to having no holes."""
    rep = Report()
    check_contiguity("review", source(gws=(5, 6, 7)), rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_contiguity_allows_sources_with_different_horizon_lengths():
    """The real passing case: review GW5-18 against solio GW5-16. A short TAIL
    is not a gap, and read_mixed renormalises per gameweek (PROJECT.md §4.6)."""
    rep = Report()
    check_contiguity("review", source(gws=tuple(range(5, 19))), rep)
    check_contiguity("solio", source(gws=tuple(range(5, 17))), rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_contiguity_fails_on_an_interior_gap():
    """5, 6, 8 — the signal is the hole at 7, which `gameweeks()` cannot see
    because it reads whatever `_Pts` columns happen to exist."""
    rep = Report()
    check_contiguity("review", source(gws=(5, 6, 8)), rep)
    assert len(rep.failures) == 1
    assert "GW7" in rep.failures[0]
    assert "GW5-GW8" in rep.failures[0]


def test_check_contiguity_names_every_missing_gameweek():
    rep = Report()
    check_contiguity("solio", source(gws=(5, 6, 9, 10, 13)), rep)
    assert "GW7, GW8, GW11, GW12" in rep.failures[0]
    assert "carries 5 of 9" in rep.failures[0]


def test_check_contiguity_is_a_failure_not_a_warning():
    """House rule: FAIL where there is no usable degraded mode. Everything
    downstream weights by POSITION — scenario_generator.load_blend slices
    `shared[:horizon]` and both risk solvers use `decay ** i` over the columns
    present — so a gap silently reweights the objective and reaches one week
    further into the season. No flag makes that right."""
    rep = Report()
    check_contiguity("review", source(gws=(1, 2, 4)), rep)
    assert rep.warnings == [] and len(rep.failures) == 1


def test_check_contiguity_says_nothing_about_a_single_gameweek_source():
    """One gameweek has no interior, so no gap is possible."""
    rep = Report()
    check_contiguity("review", source(gws=(7,)), rep)
    assert rep.failures == [] and rep.warnings == []


def test_check_contiguity_handles_a_frame_carrying_no_gameweeks_at_all():
    rep = Report()
    check_contiguity("review", pd.DataFrame({"ID": [1], "Name": ["x"]}), rep)
    assert rep.failures == [] and rep.warnings == []


def test_main_fails_when_a_source_is_missing_an_interior_gameweek(tmp_path, monkeypatch, capsys):
    write_sources(
        tmp_path,
        solio=source(gws=(1, 2, 3, 5), team_of=two_teams),
        review=source(gws=(1, 2, 3, 5), team_of=two_teams),
    )
    code = run_main(monkeypatch, ["--sources", "solio", "review", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL  solio: GW1-GW5 has 1 missing gameweek(s) (GW4)" in out
    assert "FAIL  review: GW1-GW5 has 1 missing gameweek(s) (GW4)" in out
    assert "VALIDATION FAILED" in out


def test_main_fails_when_the_blend_dropped_a_gameweek_the_sources_both_carry(tmp_path, monkeypatch, capsys):
    """The case the gate was blind to: both sources carry GW3, so the halved-GW
    check has nothing single-source to compare, and a blend that dropped GW3
    outright just looked like a shorter horizon."""
    a = source(gws=(1, 2, 3, 4), team_of=two_teams)
    write_sources(tmp_path, solio=a, review=source(gws=(1, 2, 3, 4), team_of=two_teams))
    a.drop(columns=["3_Pts", "3_xMins"]).to_csv(tmp_path / "mixed.csv", index=False)

    code = run_main(monkeypatch, ["--sources", "solio", "review", "--data-dir", str(tmp_path), "--mixed", "mixed.csv"])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL  mixed: GW1-GW4 has 1 missing gameweek(s) (GW3)" in out


def test_main_does_not_report_contiguity_for_a_mixed_file_that_was_not_supplied(tmp_path, monkeypatch, capsys):
    """check_blend gets a `dfs[0].iloc[0:0]` placeholder when --mixed is absent.
    That placeholder is not a blend, so it must not be reported on as one."""
    write_sources(tmp_path, solio=source(team_of=two_teams), review=source(team_of=two_teams))
    assert run_main(monkeypatch, ["--sources", "solio", "review", "--data-dir", str(tmp_path)]) == 0
    assert "mixed: GW" not in capsys.readouterr().out
