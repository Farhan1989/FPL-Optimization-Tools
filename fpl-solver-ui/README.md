# Solver console

A local web UI over the existing CLI tools. It does not reimplement any solver
logic — it uploads inputs, edits `user_settings.json`, shells out to the scripts
already in the repo, streams their output, and renders what they write.

## Run it

The repo uses `uv`, so add the UI's dependencies to the project rather than
creating a second environment — `pipeline.json` invokes `uv run python`, which
resolves against the repo's own environment:

```bash
cd FPL-Optimization-Tools
uv add fastapi "uvicorn[standard]" python-multipart

# Look at it first, with sample plans and no real data touched:
uv run python fpl-solver-ui/demo_data.py
SOLVER_ROOT=fpl-solver-ui/demo uv run uvicorn app:app \
  --app-dir fpl-solver-ui --port 8711

# Then for real, from the repo root:
uv run uvicorn app:app --app-dir fpl-solver-ui --port 8711
```

Open <http://127.0.0.1:8711>.

Defaults, all overridable by environment variable:

| Variable | Default | What it is |
|---|---|---|
| `SOLVER_ROOT` | parent of `fpl-solver-ui/` | working directory for every command |
| `FPL_DATA_DIR` | `{ROOT}/data` | where uploads land |
| `FPL_SETTINGS` | `{ROOT}/data/user_settings.json` | the settings file |
| `FPL_RESULTS_DIR` | `{ROOT}/results` | where plans are read from |
| `FPL_ARCHIVE_DIR` | `{ROOT}/../fpl-archive` | snapshot root, for `{ARCHIVE}` |

Bind to localhost. There is no authentication, and the run endpoint executes
commands — do not expose it beyond your own machine.

## The five panels

**01 Sources** — drop CSVs into the data folder. Row counts and a guessed source
label are shown so a truncated export is obvious before you solve on it.

**02 Settings** — reads `user_settings.json` and builds a form from whatever keys
are in it. Booleans get toggles, numbers get numeric inputs, lists and objects get
a JSON box. Nothing is hardcoded, so new settings appear on their own. Every save
writes a timestamped `.bak-` copy first. Keys listed in `INVARIANTS` in `app.py`
are marked and warn on save — `gap` is there because a non-zero gap lets the
solver stop while candidate plans still differ by more than the reported margin,
which makes the rankings in panel 05 meaningless. This applies to the stock
solver's settings only; `stochastic_solver.py` is invoked with `--gap 0.01`
deliberately, since the two-stage problem won't close to zero in useful time.

**03 Run** — one button per step, in runbook order, streamed live over SSE. *Run
all steps* goes top to bottom and stops at the first non-zero exit, so a failed
validation gate cannot leak into a solve. Steps come from `pipeline.json`.

**04 Plans** — one column per gameweek on a shared rail. Transfers read downward:
who leaves, arrow, who arrives. Chips are flagged on the rail itself.

**05 Compare** — expected points per gameweek per variant. Rows within 0.4 xPts
of the leader are shaded as tied rather than ranked.

## Two places you'll need to edit

**`pipeline.json`** — the commands, transcribed from runbook §B. The flags come
from the runbook, not from reading your argparse, so a first pass with each
button is worth doing before trusting *Run all*.

Placeholders `{ROOT}`, `{DATA}`, `{RESULTS}`, `{SETTINGS}` and `{ARCHIVE}` expand
at launch; `{ARCHIVE}` resolves to the newest directory under the archive root
containing `bootstrap_slim.json`, which is what `archive_solve.py` prints.

Anything else in braces is a runtime parameter, declared in that step's `params`
and rendered as a field on the step card — squad IDs, free transfers, bank, and
so on. Substitution happens *after* the command is split, so a value always
becomes exactly one argument and can't inject extra ones. A blank optional
parameter removes its own token along with the flag before it, so `--note {note}`
disappears rather than passing an empty string.

Commands run without a shell, so no pipes or redirects; wrap anything needing
them in a script.

**`parsers.py`** — how solver output becomes something the UI can draw. Two
readers are tried:

1. `results/*.plan.json` in the canonical shape documented at the top of the
   file. This is the reliable path — if each variant dumps one of these, nothing
   can be misread.
2. `results/*.csv` in long format, one row per player per gameweek. Column names
   are matched loosely against `COLUMN_ALIASES`; add your own spellings there.
   Files without both a week and a name column are ignored, so projection inputs
   sitting in the same folder won't be mistaken for plans.

The CSV reader assumes a row flagged `transfer_out` is leaving that gameweek and
so isn't part of that week's squad, and that a row flagged `transfer_in` is.
Buys and sells are paired by position where possible. If your solver writes a
different convention, the JSON path avoids the guesswork entirely.

## Adding a variant

Add a step to `pipeline.json`. It shows up as a button, and once it writes into
the results folder it appears as a new row in the comparison table and a new tab
in plans. Nothing else needs changing.

## What this doesn't do

No scheduling — that's the EventBridge/Batch path, and this UI is the manual
counterpart to it. No results history: it reads the current contents of the
results folder, and `archive_solve.py` remains what preserves a week.

## Fixed in this pass

**Plans and Compare now have a data source.** `cvar_solver.py` and
`stochastic_solver.py` print to stdout and write no plan files, and nothing in
the pipeline wrote to a top-level `results/` at all — so both panels were
permanently empty. The UI now parses the solver output it already captures:
each run's stdout is saved as `.runs/<step_id>.latest.log`, and `parsers.py`
reads every `Solution N` block out of it. With `num_iterations: 3` in
`user_settings.json` that gives exactly the comparison the runbook asks for —
near-tied candidate plans, ranked honestly. `FPL_RESULTS_DIR` also now defaults
to `data/results`, which is where the repo actually writes.

**`chip_planner.py score` had nothing to read.** `plans/` is populated by
`chip_planner.py enumerate`, which had no step — so the chips button always
failed. Added as C1, with the C2 scoring step after it.

**Stale scenario mixing.** Steps can declare `"clean": ["scenarios"]`, cleared
before the step runs. Regenerating with a smaller `--scenarios` used to leave
the higher-numbered files from the previous run in place, and every downstream
tool globs the whole directory. `chip_enumerate` clears `plans/` for the same
reason: logs from an earlier projection vintage are not comparable with fresh
ones. Paths are constrained to inside `SOLVER_ROOT`.

**Invariants were incomplete.** `gap` alone was covered; PROJECT.md section 4
also lists `decay_base`, `horizon` and `data_weights`. All four now warn on
save. The stock-solver-only caveat still stands: `stochastic_solver.py` is
called with `--gap 0.01` deliberately.

**`horizon` lived in two places** — a step parameter and a solver setting, free
to diverge silently. The settings file now supplies the field's default.

**Preseason had no path.** `stochastic_solver.py` was always invoked with
`--squad`, never `--preseason`, so GW1 and wildcards fell back to the CLI.
Added as A7, disabled by default, with `--vss`.

**`validate_deep` moved** from position seven to directly after the archive
step, and relabelled B2b — it needs the `mixed.csv` that the archive step's
solve produces, so it cannot run earlier, but it should not run after the
decisions it is meant to inform.

Still worth checking on your side: `git -C ../fpl-archive remote -v`. The
archive step passes `--push` unconditionally, which silently does nothing if no
remote is configured.
