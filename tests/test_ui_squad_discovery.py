"""Squad auto-discovery in the UI console.

runbook §1.4 makes `stochastic_solver.py` THE transfer decision, yet its
output was the one thing the *Fill from…* dropdown could not offer: the log
printed player NAMES, and names are ambiguous in this dataset ("Palmer"
resolves to two FPL ids this season). The solver now states its own fifteen
as ids; these tests pin that line's shape and the trust grading built on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "fpl-solver-ui"
if str(UI) not in sys.path:
    sys.path.insert(0, str(UI))

import parsers  # noqa: E402

SQUAD = [12, 31, 109, 115, 124, 154, 249, 277, 368, 411, 447, 464, 467, 469, 565]
LIVE = list(SQUAD)
OLD = [*SQUAD[:-1], 999]  # a squad holding one player the live squad no longer does
BODY = "[2stage] 6/200 scenarios (stratified), pool 109, GWs 5-6, lam=0.3\n"


def log(ids, *, header=True, seed=None, squad_free=False):
    cmd = "uv run python stochastic_solver.py --scenario-dir scenarios/"
    if seed:
        cmd += " --squad " + ",".join(str(i) for i in seed)
    if squad_free:
        cmd += " --preseason"
    head = f"# FPL Solver Console run record\n# step         stochastic\n# command      {cmd}\n\n" if header else ""
    return head + BODY + "[2stage] stage-1 squad IDs: " + ",".join(str(i) for i in ids) + "\n"


# ------------------------------------------------------------- the ID line


def test_reads_the_ids_the_solver_stated():
    assert parsers._ids_from_stage1_ids_line(log(SQUAD)) == SQUAD


def test_ignores_a_line_that_is_not_a_full_squad():
    assert parsers._ids_from_stage1_ids_line(log(SQUAD[:14])) == []


def test_ignores_a_line_with_a_duplicate():
    assert parsers._ids_from_stage1_ids_line(log([*SQUAD[:14], SQUAD[0]])) == []


def test_absent_line_is_not_an_error():
    assert parsers._ids_from_stage1_ids_line(BODY) == []


def test_tolerates_spaces_after_the_commas():
    text = BODY + "[2stage] stage-1 squad IDs: " + ", ".join(str(i) for i in SQUAD) + "\n"
    assert parsers._ids_from_stage1_ids_line(text) == SQUAD


# -------------------------------------------------------------- grading


def entry(text, *, live_ids=LIVE, current_gw=5, horizon=5):
    return parsers._stage1_ids_entry("stochastic", SQUAD, parsers.read_log_provenance(text), horizon, current_gw, live_ids)


def test_seeded_with_the_live_squad_is_fresh():
    e = entry(log(SQUAD, seed=LIVE))
    assert e["trust"] == parsers.TRUST_FRESH
    assert not e["stale"]


def test_seeded_with_a_superseded_squad_is_stale():
    e = entry(log(SQUAD, seed=OLD))
    assert e["trust"] == parsers.TRUST_STALE
    assert e["stale"]
    assert "no longer contains" in e["note"]


def test_a_headerless_log_is_unknown_not_fresh():
    """`prov` is always a truthy dict carrying header=False, so an early
    version of this graded headerless logs FRESH. Unverifiable is not fine."""
    e = entry(log(SQUAD, header=False))
    assert e["trust"] == parsers.TRUST_UNKNOWN
    assert "unverified" in e["label"]


def test_a_squad_free_solve_is_not_treated_as_a_stale_seed():
    e = entry(log(SQUAD, squad_free=True))
    assert e["trust"] == parsers.TRUST_FRESH


def test_a_solve_for_played_gameweeks_is_stale_whatever_seeded_it():
    e = entry(log(SQUAD, seed=LIVE), horizon=1)
    assert e["trust"] == parsers.TRUST_STALE
    assert "GW01" in e["label"] and "GW05" in e["label"]


@pytest.mark.parametrize("missing", ["live", "gw"])
def test_unreadable_state_is_unknown(missing):
    e = entry(log(SQUAD, seed=LIVE), live_ids=[] if missing == "live" else LIVE, current_gw=None if missing == "gw" else 5)
    assert e["trust"] == parsers.TRUST_UNKNOWN
    assert "unverified" in e["label"]


def test_the_entry_keeps_the_api_contract():
    e = entry(log(SQUAD, seed=LIVE))
    assert set(e) >= {"key", "label", "ids", "note"}
    assert len(e["ids"]) == parsers.SQUAD_SIZE
