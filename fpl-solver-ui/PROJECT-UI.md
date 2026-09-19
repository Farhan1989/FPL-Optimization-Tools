# Solver Console — session record and decision log

Companion to `PROJECT.md`. Covers the web UI built over the existing CLI tools.
Written 15 Aug 2026.

---

## 1. What was built

A local FastAPI app at `fpl-solver-ui/`, sitting inside the
`FPL-Optimization-Tools` repo. It reimplements no solver logic — it uploads
inputs, edits settings, shells out to the existing scripts, streams their
output, and renders what they write.

| File | Role |
|---|---|
| `app.py` | HTTP API: uploads, settings read/write, run launcher, SSE streaming |
| `parsers.py` | Adapter — solver output to canonical plan shape |
| `pipeline.json` | Declarative step list; the source of the run buttons |
| `static/index.html` `styles.css` `app.js` | Frontend, no build step |
| `demo_data.py` | Sample plans for inspecting the UI without real data |

Five panels: **Sources** (CSV upload), **Settings** (form over
`data/user_settings.json`), **Run** (a button per runbook step, streamed),
**Plans** (gameweek columns with transfer arrows), **Compare** (score table).

---

## 2. Decisions and rationale

**Local server, not a browser artifact.** The UI must write to `data/`, read
`results/`, and execute solver scripts. A sandboxed browser page can do none of
those.

**No build step.** Vanilla JS with fonts from a CDN. A weekly-use internal tool
should not acquire an npm dependency tree that rots between seasons.

**Commands live in `pipeline.json`, not in code.** Flags change; the runbook is
the authority. Editing JSON to fix a flag is a smaller act than editing Python,
and the file is committable alongside a week's solve as part of the record of
how that solve was produced.

**Substitution happens after `shlex.split`, never before.** Parameter values
become exactly one argv token each, so a squad field containing shell
metacharacters is inert. Commands run with no shell at all — no pipes, no
redirects.

**`{ARCHIVE}` resolves by file, not by name.** It finds the newest directory
containing `bootstrap_slim.json` rather than parsing timestamp directory names,
so a changed naming scheme doesn't silently break it.

**The noise band is a first-class UI concept.** Plans within 0.4 xPts of the
leader are shaded as tied rather than ranked, and the count surfaces in the
header. This encodes the project's own empirical finding — validated against
Solio's chip tree, where four full-season plans separated by ~0.4 points over 19
gameweeks. A table that ranks by the second decimal would actively mislead.

**`gap: 0` is marked as an invariant and warns on save.** Scoped to the stock
solver's settings only. `stochastic_solver.py` is deliberately invoked at
`--gap 0.01`, hardcoded in the pipeline, because the two-stage problem will not
close to zero in useful time.

**Run all stops at the first non-zero exit.** A failed validation gate must not
leak into a solve.

---

## 3. Corrections made mid-session

The first pipeline was written from memory of the tool names rather than from
`runbook.md`, and was wrong in four ways. All four are fixed, but the pattern is
the point: **memory of a project is not the same as its documentation.**

1. **Missing `uv run`.** Every runbook command is `uv run python <script>`.
   Bare `python` would have failed on every button.
2. **Invented flags.** `run_solver.py --objective expected_points --tag ev` and
   similar do not exist. Replaced with the real §B sequence.
3. **Wrong settings path.** The runbook puts settings at
   `data/user_settings.json`; the app defaulted to the repo root.
4. **No runtime parameters.** `--squad`, `--fts`, `--itb` and `{ARCHIVE}` change
   every week and cannot live in static config. Step cards now render fields for
   them, and values persist so *Run all* reuses them.

This is the fourth instance in this project of output being described more
confidently than it was verified — consistent with the existing note about
claiming files were delivered without publishing them. The mitigation is the
same: check the artifact, not the description.

---

## 4. Limitations

- **`parsers.py` is unverified against real solver output.** It reads
  `results/*.plan.json` in a documented canonical shape reliably; the CSV
  fallback matches column names loosely against `COLUMN_ALIASES` and guesses at
  transfer conventions. If the Plans panel is empty after a successful solve,
  this is why.
- **Pipeline flags are transcribed from the runbook, not read from argparse.**
  Each button needs one manual first run before *Run all* is trustworthy.
- **The visual design was never rendered.** The browser CDN was unreachable in
  the build sandbox, so no screenshot was taken. Layout, spacing, and the
  transfer-arrow rendering are unconfirmed.
- **No authentication.** The run endpoint executes commands. Bind to localhost
  only.
- **No results history.** It reads the current contents of `results/`.
  `archive_solve.py` remains what preserves a week. Run logs under `.runs/` do
  now carry a `#` provenance header (command, ids, timestamps, exit status), so
  a finished run can at least be identified after the server stops.
- **Squad-fill trust is three-valued, and the third value is load-bearing.**
  Every *Fill from…* entry now carries `trust` of `fresh` / `unknown` / `stale`,
  shown in the label (`note` is only a hover tooltip) and sorted on, so a
  trustworthy entry never sits below an untrustworthy one. `unknown` means
  *we could not check*, not *fine*: a log with no `#` provenance header, a
  recorded command with no `--squad`, or a live API that did not answer. The
  evidence for a derived entry is the recorded `--squad` compared against the
  live squad; failing that, which base squad its first-gameweek Sells could
  actually be applied to, which proves staleness with no header at all.
  Nothing is ever dropped — an unverifiable entry is still usually the right
  one, and removing it just sends you back to typing fifteen IDs.
- **Preseason entries are squad-free, not stale-seeded — but they do expire.**
  A `stage-1 squad` or `EV solve squad` entry comes from a solve that was
  handed no team at all (`--preseason`), so there is no seed to be stale. What
  dates it is its own horizon: a 15 built for GW1–4 is a historical answer
  once GW5 is live, and it is labelled with the gameweeks it was built for
  rather than as a stale seed. During an actual preseason or wildcard the same
  entry grades `fresh`.
- **Still unverifiable by design:** a run whose squad is fetched by the script
  rather than passed on the command line (`archive_solve.py` reads picks from
  the API itself). Its `--squad` will never appear in a header, so such an
  entry can only ever reach `fresh` through the Sells-against-live check being
  extended to prove equality, which it cannot do today.
- **Python 3.14 concern (unresolved).** `highspy` and `sasoptpy` may lack wheels
  for 3.14; if the repo environment is on it, that is worth checking
  independently of this UI.

---

## 5. Open items

1. Run each pipeline step once manually and correct any flag mismatches.
2. Confirm the Plans panel renders real solver output; if not, either fix
   `COLUMN_ALIASES` or have each variant emit a canonical `*.plan.json`.
3. Review the rendered design and adjust — it has never been seen.
4. Decide whether `pipeline.json` should be committed by `archive_solve.py` as
   part of a week's snapshot.
5. Unchanged from before: VSS computation in `stochastic_solver.py` has still
   never produced a number on real hardware.

---

## 6. Relationship to the cloud plan

This UI is the manual counterpart to the EventBridge/Batch design, not a
replacement. Scheduling, the variant sweep, and the GitHub Issues results log
remain the automated path. The overlap worth watching: `pipeline.json` and the
Lambda dispatcher's variant matrix describe the same commands in two places, and
will drift unless one is derived from the other.
