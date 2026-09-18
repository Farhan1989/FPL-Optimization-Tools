"""Contract tests for solio_enrich.py.

Everything here is offline. `main()` imports `requests` lazily inside the
fetch branch precisely so the `--from-file` paths never touch the network;
`no_network` below installs a stub that explodes if that ever regresses.

Fixtures are small and synthetic on purpose: `data/enrich.json` is
regenerated weekly, so nothing may depend on its values.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest

import solio_enrich
from solio_enrich import fill_cs, parse_feed, parse_page, team_code

CONTRACT_KEYS = {"gameweek", "generated", "teams", "defcon"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to reach the network from a test is a test bug."""

    def boom(*_args, **_kwargs):
        raise AssertionError("solio_enrich tests must never hit the network")

    stub = types.ModuleType("requests")
    stub.get = boom
    monkeypatch.setitem(sys.modules, "requests", stub)


def feed_row(team, gf, ga, cs=None, opponents=()):
    row = {"team": team, "prGoalsFor": gf, "prGoalsAgainst": ga, "fixtures": [{"opponent": o} for o in opponents]}
    if cs is not None:
        row["csProb"] = cs
    return row


def make_feed(attacking=(), clean_sheets=(), defcon=(), gameweek=7, generated="2026-08-19T23:29:26.663Z"):
    return {
        "gameweek": gameweek,
        "generatedAt": generated,
        "bestAttackingFixtures": list(attacking),
        "bestCleanSheets": list(clean_sheets),
        "topDefCon": [{"name": n, "team": t, "prDefConProb": p} for n, t, p in defcon],
    }


def wide_feed(n_teams, gameweek=7, defcon=(("Murillo", "Nott'm Forest", 0.65),)):
    """A feed with exactly `n_teams` teams that end up carrying a CS value."""
    codes = [
        "Arsenal",
        "Aston Villa",
        "Bournemouth",
        "Brentford",
        "Brighton",
        "Chelsea",
        "Coventry",
        "Crystal Palace",
        "Everton",
        "Fulham",
        "Hull",
        "Ipswich",
    ]
    rows = [feed_row(codes[i], 1.0 + i / 10, 1.0, cs=0.4) for i in range(n_teams)]
    return make_feed(attacking=rows, defcon=defcon, gameweek=gameweek)


# ------------------------------------------------------------------ team_code


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Arsenal", "ARS"),
        ("Nott'm Forest", "NFO"),
        ("Man Utd", "MUN"),
        ("Spurs", "TOT"),
        ("Tottenham", "TOT"),  # both spellings collapse to one code
    ],
)
def test_team_code_maps_full_names(name, expected):
    assert team_code(name) == expected


@pytest.mark.parametrize("code", ["ARS", "NFO", "WHU", "XYZ"])
def test_team_code_passes_three_letter_codes_through(code):
    """Solio writes full names in `team` and 3-letter codes in `opponent`."""
    assert team_code(code) == code


@pytest.mark.parametrize("name", [None, "", "Barcelona", "ars", "Ars", "ARSE", "Arsenal FC", "AR"])
def test_team_code_returns_none_for_unrecognised(name):
    """An unmapped side must be DROPPED, not silently pollute the team map."""
    assert team_code(name) is None


# -------------------------------------------------------------------- fill_cs


def test_fill_cs_poisson_fallback_for_missing_cs():
    teams = {"COV": {"gf": 0.51, "ga": 2.57}}
    fill_cs(teams)
    assert teams["COV"]["cs"] == round(math.exp(-2.57), 4)


def test_fill_cs_never_overwrites_a_published_cs():
    teams = {"ARS": {"gf": 2.57, "ga": 0.51, "cs": 0.5982}}
    fill_cs(teams)
    assert teams["ARS"]["cs"] == 0.5982  # NOT exp(-0.51) = 0.6005


def test_fill_cs_leaves_teams_without_goals_against_alone():
    teams = {"ARS": {"gf": 2.0}}
    fill_cs(teams)
    assert "cs" not in teams["ARS"]


# ------------------------------------------------------------------ parse_feed


def test_parse_feed_contract_keys():
    out = parse_feed(make_feed(attacking=[feed_row("Arsenal", 2.57, 0.51, cs=0.5982)], defcon=[("Murillo", "NFO", 0.65)]))
    assert set(out) == CONTRACT_KEYS
    assert out["gameweek"] == 7
    assert out["generated"] == "2026-08-19T23:29:26.663Z"
    assert out["teams"]["ARS"] == {"gf": 2.57, "ga": 0.51, "cs": 0.5982}
    assert out["defcon"] == {"Murillo|NFO": 0.65}


def test_parse_feed_mirrors_opponent_goals():
    """A team appearing only as someone else's opponent gets mirrored gf/ga."""
    out = parse_feed(make_feed(attacking=[feed_row("Arsenal", 2.57, 0.51, cs=0.5982, opponents=["COV"])]))
    assert out["teams"]["COV"]["gf"] == 0.51
    assert out["teams"]["COV"]["ga"] == 2.57
    assert out["teams"]["COV"]["cs"] == round(math.exp(-2.57), 4)


@pytest.mark.parametrize("reverse", [False, True])
def test_parse_feed_mirroring_never_overwrites_published_numbers(reverse):
    """Mirroring is a fallback: a team's own row wins whichever order it arrives in."""
    rows = [
        feed_row("Arsenal", 2.57, 0.51, cs=0.5982, opponents=["COV"]),
        feed_row("Coventry", 0.80, 2.10, cs=0.1200, opponents=["ARS"]),
    ]
    out = parse_feed(make_feed(attacking=list(reversed(rows)) if reverse else rows))
    assert out["teams"]["COV"] == {"gf": 0.80, "ga": 2.10, "cs": 0.12}
    assert out["teams"]["ARS"] == {"gf": 2.57, "ga": 0.51, "cs": 0.5982}


def test_parse_feed_drops_unmapped_team():
    out = parse_feed(make_feed(attacking=[feed_row("Real Madrid", 2.0, 1.0, cs=0.4), feed_row("Arsenal", 2.57, 0.51, cs=0.5982)]))
    assert set(out["teams"]) == {"ARS"}


def test_parse_feed_skips_rows_missing_goals():
    out = parse_feed(make_feed(attacking=[{"team": "Arsenal", "prGoalsFor": 2.0}, feed_row("Chelsea", 1.9, 0.9, cs=0.4)]))
    assert set(out["teams"]) == {"CHE"}


def test_parse_feed_poisson_fallback_when_cs_prob_absent():
    out = parse_feed(make_feed(attacking=[feed_row("Arsenal", 2.57, 0.51)]))
    assert out["teams"]["ARS"]["cs"] == round(math.exp(-0.51), 4)


def test_parse_feed_unions_both_tables():
    """The two tables are each a top-10; the union widens coverage."""
    out = parse_feed(
        make_feed(
            attacking=[feed_row("Arsenal", 2.57, 0.51, cs=0.5982)],
            clean_sheets=[feed_row("Everton", 0.90, 0.60, cs=0.5488)],
        )
    )
    assert set(out["teams"]) == {"ARS", "EVE"}


def test_parse_feed_rounds_to_four_places():
    out = parse_feed(make_feed(attacking=[feed_row("Arsenal", 2.5712345, 0.5123456, cs=0.59823456)], defcon=[("Murillo", "NFO", 0.6512345)]))
    assert out["teams"]["ARS"] == {"gf": 2.5712, "ga": 0.5123, "cs": 0.5982}
    assert out["defcon"]["Murillo|NFO"] == 0.6512


@pytest.mark.parametrize(
    "row",
    [
        {"name": "Murillo", "team": "Real Madrid", "prDefConProb": 0.65},  # unmapped team
        {"name": "Murillo", "team": "NFO"},  # no probability
        {"team": "NFO", "prDefConProb": 0.65},  # no name
    ],
)
def test_parse_feed_drops_incomplete_defcon_rows(row):
    d = make_feed(attacking=[feed_row("Arsenal", 2.57, 0.51, cs=0.5982)])
    d["topDefCon"] = [row]
    assert parse_feed(d)["defcon"] == {}


def test_parse_feed_defcon_key_uses_team_code_not_full_name():
    out = parse_feed(make_feed(attacking=[feed_row("Arsenal", 2.57, 0.51, cs=0.6)], defcon=[("Gabriel", "Arsenal", 0.55)]))
    assert out["defcon"] == {"Gabriel|ARS": 0.55}


def test_parse_feed_coerces_gameweek_to_int():
    assert parse_feed(make_feed(gameweek="6"))["gameweek"] == 6


def test_parse_feed_empty_feed_yields_empty_contract():
    out = parse_feed({})
    assert out == {"gameweek": None, "generated": None, "teams": {}, "defcon": {}}


def test_parse_feed_tolerates_null_sections():
    out = parse_feed({"gameweek": 6, "bestAttackingFixtures": None, "bestCleanSheets": None, "topDefCon": None})
    assert out["teams"] == {} and out["defcon"] == {}


# ------------------------------------------------------------------ parse_page

MD_PAGE = """
# Solio Data — Gameweek 7

Generated: 2026-08-19T23:29:26.663Z

## Best attacking fixtures

| # | Team | Fixtures | G For | G Against |
|---|---|---|---|---|
| 1 | Arsenal | vs COV | 2.57 | 0.51 |

## Best clean sheet odds

| # | Team | Fixtures | G Against | CS % |
|---|---|---|---|---|
| 1 | Arsenal | vs COV | 0.51 | 60% |

## Top projected DefCon

| # | Player | Team | Pos | Price | DefCon % |
|---|---|---|---|---|---|
| 1 | Murillo | NFO | D | 4.8 | 65% |
"""


def test_parse_page_contract_keys():
    out = parse_page(MD_PAGE)
    assert set(out) == CONTRACT_KEYS
    assert out["gameweek"] == 7
    assert out["generated"] == "2026-08-19T23:29:26.663Z"
    assert out["defcon"] == {"Murillo|NFO": 0.65}


def test_parse_page_mirrors_opponent_goals():
    teams = parse_page(MD_PAGE)["teams"]
    assert teams["ARS"] == {"gf": 2.57, "ga": 0.51, "cs": 0.6}
    assert teams["COV"]["gf"] == 0.51
    assert teams["COV"]["ga"] == 2.57
    assert teams["COV"]["cs"] == round(math.exp(-2.57), 4)


def test_parse_page_and_parse_feed_agree_on_the_same_data():
    """The two parsers must produce the identical contract. CS values here are
    exact at integer percent so the markdown table's quantisation is not in play."""
    feed = make_feed(
        attacking=[feed_row("Arsenal", 2.57, 0.51, cs=0.60, opponents=["COV"])],
        defcon=[("Murillo", "NFO", 0.65)],
        gameweek=7,
    )
    assert parse_feed(feed) == parse_page(MD_PAGE)


def test_parse_page_without_gameweek_heading():
    assert parse_page("## Best attacking fixtures\n")["gameweek"] is None


def test_parse_page_ignores_unparseable_numeric_cells():
    text = MD_PAGE.replace("| 1 | Arsenal | vs COV | 2.57 | 0.51 |", "| 1 | Arsenal | vs COV | n/a | 0.51 |")
    teams = parse_page(text)["teams"]
    assert "gf" not in teams.get("ARS", {})  # attacking row dropped, CS row still lands
    assert teams["ARS"]["cs"] == 0.6


def test_parse_page_drops_unmapped_team_rows():
    text = MD_PAGE.replace("| 1 | Arsenal | vs COV | 2.57 | 0.51 |", "| 1 | Real Madrid | vs COV | 2.57 | 0.51 |")
    assert "Real Madrid" not in parse_page(text)["teams"]


@pytest.mark.parametrize("section", ["| 1 | Arsenal | vs COV | 2.57 | 0.51 |", "| 1 | Arsenal | vs COV | 0.51 | 60% |"])
def test_parse_page_accepts_a_team_cell_holding_a_code(section):
    """Both parsers normalise through team_code(), which accepts an already
    3-letter uppercase code. The markdown path used to call TEAM_CODE.get()
    directly and silently drop such a row — an escape hatch (§8 Phase 5) that
    read fewer input shapes than the primary path it stands in for."""
    text = MD_PAGE.replace(section, section.replace("Arsenal", "ARS"))
    assert parse_page(text)["teams"]["ARS"] == {"gf": 2.57, "ga": 0.51, "cs": 0.6}


def test_parse_page_agrees_with_parse_feed_on_a_coded_team_cell():
    """The cross-validation §8 Phase 5 claims: identical contract, either shape."""
    text = MD_PAGE.replace("| Arsenal |", "| ARS |")
    feed = make_feed(
        attacking=[feed_row("Arsenal", 2.57, 0.51, cs=0.60, opponents=["COV"])],
        defcon=[("Murillo", "NFO", 0.65)],
        gameweek=7,
    )
    assert parse_page(text) == parse_feed(feed)


# ----------------------------------------------------------------------- main


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["solio_enrich.py", *argv])
    return solio_enrich.main()


def test_main_from_json_file_writes_parse_feed_output(tmp_path, monkeypatch):
    src = tmp_path / "page.json"
    feed = wide_feed(12)
    src.write_text(json.dumps(feed))
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == parse_feed(feed)


def test_main_creates_missing_output_directories(tmp_path, monkeypatch):
    src = tmp_path / "page.json"
    src.write_text(json.dumps(wide_feed(12)))
    out = tmp_path / "nested" / "deeper" / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 0
    assert out.exists()


def test_main_md_extension_uses_the_markdown_parser(tmp_path, monkeypatch, capsys):
    src = tmp_path / "page.md"
    # widen the markdown page so it clears the >=10-teams-with-CS floor
    rows = "\n".join(
        f"| {i + 1} | {n} | vs {o} | 2.0 | 0.5 |"
        for i, (n, o) in enumerate([("Arsenal", "COV"), ("Chelsea", "EVE"), ("Fulham", "HUL"), ("Leeds", "IPS"), ("Liverpool", "BUR")])
    )
    src.write_text(MD_PAGE.replace("| 1 | Arsenal | vs COV | 2.57 | 0.51 |", rows))
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 0
    assert "md feed" in capsys.readouterr().out
    assert json.loads(out.read_text())["gameweek"] == 7


def test_main_md_flag_forces_the_legacy_path_on_a_json_file(tmp_path, monkeypatch, capsys):
    """--md must not silently fall back to JSON: a JSON file forced through the
    markdown parser parses thin and is refused."""
    src = tmp_path / "page.json"
    src.write_text(json.dumps(wide_feed(12)))
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--md", "--out", str(out)]) == 1
    assert "parsed thin" in capsys.readouterr().err
    assert not out.exists()


def test_main_json_parser_used_for_unknown_extension(tmp_path, monkeypatch, capsys):
    src = tmp_path / "page.txt"
    src.write_text(json.dumps(wide_feed(12)))
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 0
    assert "json feed" in capsys.readouterr().out


@pytest.mark.parametrize("suffix", [".MD", ".Md"])
def test_main_extension_sniffing_is_case_insensitive(tmp_path, monkeypatch, suffix):
    src = tmp_path / f"page{suffix}"
    src.write_text(json.dumps(wide_feed(12)))
    out = tmp_path / "enrich.json"

    # markdown parser on JSON text => thin => refused, which proves .MD routed to it
    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 1
    assert not out.exists()


def test_main_refuses_unparseable_json(tmp_path, monkeypatch, capsys):
    src = tmp_path / "page.json"
    src.write_text("{not json at all")
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 1
    assert "did not parse" in capsys.readouterr().err
    assert not out.exists()


# --- the thin-parse refusal: a deliberate safety property -------------------


def test_main_refuses_when_gameweek_is_missing(tmp_path, monkeypatch, capsys):
    src = tmp_path / "page.json"
    src.write_text(json.dumps(wide_feed(12, gameweek=None)))
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 1
    assert "gw=None" in capsys.readouterr().err
    assert not out.exists()


def test_main_refuses_with_fewer_than_ten_teams_carrying_cs(tmp_path, monkeypatch, capsys):
    src = tmp_path / "page.json"
    src.write_text(json.dumps(wide_feed(9)))
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 1
    assert "teams-with-cs=9" in capsys.readouterr().err
    assert not out.exists()


def test_main_accepts_exactly_ten_teams_with_cs(tmp_path, monkeypatch):
    """Boundary: 10 is enough, 9 is not."""
    src = tmp_path / "page.json"
    src.write_text(json.dumps(wide_feed(10)))
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 0
    assert len(json.loads(out.read_text())["teams"]) == 10


def test_main_refuses_when_defcon_is_empty(tmp_path, monkeypatch, capsys):
    src = tmp_path / "page.json"
    src.write_text(json.dumps(wide_feed(12, defcon=())))
    out = tmp_path / "enrich.json"

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 1
    assert "defcon=0" in capsys.readouterr().err
    assert not out.exists()


def test_main_thin_refusal_leaves_an_existing_output_untouched(tmp_path, monkeypatch):
    """A refused run must never clobber last week's good enrichment."""
    src = tmp_path / "page.json"
    src.write_text(json.dumps(wide_feed(9)))
    out = tmp_path / "enrich.json"
    out.write_text('{"gameweek": 5}')

    assert run_main(monkeypatch, ["--from-file", str(src), "--out", str(out)]) == 1
    assert json.loads(out.read_text()) == {"gameweek": 5}


# --------------------------------------------------- shape of the live sample

REAL_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "enrich.json"


@pytest.mark.skipif(not REAL_SAMPLE.exists(), reason="data/enrich.json not present")
def test_checked_in_sample_still_matches_the_contract():
    """Shape only — the user regenerates data/enrich.json weekly."""
    d = json.loads(REAL_SAMPLE.read_text())
    assert set(d) == CONTRACT_KEYS
    assert isinstance(d["gameweek"], int)
    assert sum(1 for t in d["teams"].values() if "cs" in t) >= 10
    assert d["defcon"]
    for code, t in d["teams"].items():
        assert len(code) == 3 and code.isupper()
        assert {"gf", "ga"} <= set(t)
        assert 0.0 <= t["cs"] <= 1.0
    for key, p in d["defcon"].items():
        name, _, team = key.partition("|")
        assert name and len(team) == 3 and team.isupper()
        assert 0.0 <= p <= 1.0
