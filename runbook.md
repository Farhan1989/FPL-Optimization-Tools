# FPL Pipeline Runbook — 2026/27

How to actually run this, week by week. `PROJECT.md` says *why* each decision
was made; this says *what to type*.

Every sequence is given twice: **UI** (the local console) and **CLI** (the same
commands, for scripting or when the UI misbehaves). They run identical
commands — the UI is a launcher, not a reimplementation.

Unfamiliar term? PROJECT.md §2 is a glossary.

---

## 0. Setup — once

**Layout.** Two sibling directories, deliberately separate so upstream pulls
never collide with your weekly snapshots:

```
~/Code/Personal/
├── FPL-Optimization-Tools/     solver repo + all custom tools at its root
│   ├── archive_solve.py  validate_sources.py  solio_enrich.py
│   ├── scenario_generator.py  cvar_solver.py  stochastic_solver.py
│   ├── chip_planner.py
│   ├── fpl-solver-ui/          the console (the static/ subdirectory matters)
│   ├── data/                   review.csv, solio.csv, user_settings.json
│   ├── scenarios/  plans/      generated, safe to delete
│   └── run/  dev/              upstream code, do not edit
└── fpl-archive/                separate git repo, created automatically
```

**Dependencies.** `sasoptpy` is no longer an upstream dependency; the two risk
solvers still need it:

```bash
cd ~/Code/Personal/FPL-Optimization-Tools
uv add sasoptpy
uv sync
uv run python -c "import sasoptpy, highspy; print('ok')"
```

**Archive remote.** `--push` silently does nothing without one, and the
ownership snapshots are the irreplaceable part:

```bash
git -C ../fpl-archive remote -v          # empty = nothing is backed up
```

**Start the UI:**

```bash
uv run uvicorn app:app --app-dir fpl-solver-ui --port 8711
```

then open `http://127.0.0.1:8711`. Localhost only.

**Settings to confirm** in `data/user_settings.json`, or the UI Settings panel,
which warns if you break an invariant:

| Setting | Value | Why |
|---|---|---|
| `datasource` | `"mixed"` | blend both sources |
| `data_weights` | `{"review": 1, "solio": 1}` | §4.1 |
| `gap` | `0` | §4.3 — without it, plan ordering is not real |
| `horizon` | `12` | §4.6 |
| `decay_base` | `0.87` | §5 |
| `num_iterations` | `3` | gives the Compare panel something to compare |
| `secs` | `900`+ | per-solve time limit |

---

## 1. Every gameweek

Run 24–48h before the deadline. Repeat on deadline morning if there is major
team news — projections move.

### 1.1 Refresh the sources — manual, both routes

Download into `data/`:

- **FPL Review** export → `data/review.csv`. Keep the elevenify dial fixed all
  season (§4.4) and record its value in the note below.
- **Solio** projection download → `data/solio.csv`.

Fixture alignment between the two files is your responsibility. The validator
checks it; it cannot fix it.

### 1.2 Archive, validate, EV solve

One command: snapshots everything, refuses to solve on bad data, runs the stock
solver, commits.

**UI:** step **B2**. Fill *Note* (e.g. `review elevenify dial = 50%`), press
**Run**. This is the gate — *Run all steps* stops here if validation fails.

**CLI:**
```bash
uv run python archive_solve.py --validate --push --note "review elevenify dial = 50%"
```

If validation fails, read `validation.log` in the new snapshot and fix the
data. Do not `--force` past it unless you know exactly why it fired.

Note the snapshot path it prints — `<ARCHIVE>` below means that directory.

### 1.3 Enrich, then generate scenarios

**UI:** steps **B3a** then **B3b**. B3b takes *Scenarios* (200), *Horizon* (12,
defaulted from your settings) and *Seed* (42). It clears `scenarios/` first.

**CLI:**
```bash
uv run python solio_enrich.py --out data/enrich.json
rm -rf scenarios/
uv run python scenario_generator.py --sources review solio --data-dir data \
    --scenarios 200 --out scenarios/ --horizon 12 --seed 42 \
    --enrich data/enrich.json
```

Check the printed calibration line: **bias should be within ±0.05.** If not,
something upstream changed and the scenarios are no longer a mean-preserving
spread of your projections.

The `rm -rf` matters on the CLI. Regenerating with a *smaller* `--scenarios`
leaves higher-numbered files from the previous run in place, and every
downstream tool globs the whole directory. The UI does this for you.

### 1.4 The transfer decision

**UI:** step **B4**. Use the **Fill from…** dropdown above *Squad IDs* and pick
*Your team* — no typing. Set free transfers and bank.

**CLI:**
```bash
uv run python stochastic_solver.py --scenario-dir scenarios/ \
    --squad "id1,...,id15" --fts 2 --itb 0.5 \
    --weeks 5 --use-scenarios 16 --lam 0.3 --gap 0.01 --secs 900 \
    --bootstrap <ARCHIVE>/bootstrap_slim.json
```

Read the **stage-1 line** — that is the move you commit to. Treat recourse
branches as informational: at 16 scenarios almost every branch is 1/16, which
is sixteen bespoke reactions rather than a plan tree. Consensus below roughly
6/16 is not signal.

Keep `--use-scenarios` at 12–16. Stage 2 grows linearly in scenario count; 100
produces a model about eight times larger that will not converge.

### 1.5 Cross-check — only when the two solvers disagree

Same recommendation from B2 and B4? Done. If they differ, score both squads.

**UI:** step **B5a**, twice — the dropdown offers *EV solve squad* and *after
this week's move*.

**CLI:**
```bash
uv run python cvar_solver.py --scenario-dir scenarios/ \
    --bootstrap <ARCHIVE>/bootstrap_slim.json --weeks 4 --evaluate "<15 ids>"
```

You get `E[Δ]`, `sd`, `CVaR₂₀%` and `P(Δ>0)`. The EV move maximises
expectation; the stochastic move protects the tail. Disagreement tells you how
assumption-sensitive the week is — information, not error.

**Read these with the standard error in mind.** With sd ≈ 18 over 200
scenarios the SE on each mean is ≈ 1.3, so a difference under ~2.5 points is
not distinguishable. The squads also share most of their players, so a paired
comparison would be sharper — not yet implemented.

### 1.6 Optional: a tail-optimised alternative

**UI:** step **B5b**. **CLI:**
```bash
uv run python cvar_solver.py --scenario-dir scenarios/ \
    --bootstrap <ARCHIVE>/bootstrap_slim.json --weeks 4 \
    --lam 0.5 --alpha 0.2 --compare-ev
```

Builds the squad that optimises the tail directly, printing λ=0 alongside.
**Caveat:** this squad is chosen *on* the scenarios it is then scored on, so
its numbers are optimistically biased. For a fair comparison see §3.

### 1.7 Execute

Enter transfer, lineup and captain on the FPL site before the deadline.

---

## 2. Preseason, wildcards and free hits

Same shape, but there is no current squad, so the initial-15 tools replace the
transfer decision.

1. §1.1 – §1.3 unchanged.
2. **Build the squad under uncertainty** — UI step **A7**, or:
   ```bash
   uv run python stochastic_solver.py --scenario-dir scenarios/ --preseason \
       --weeks 4 --use-scenarios 16 --lam 0.3 --gap 0.01 --secs 900 \
       --bootstrap <ARCHIVE>/bootstrap_slim.json --vss
   ```
   `--vss` triples the runtime and prices what uncertainty modelling bought.
3. **Score it against the EV squad** — §1.5.
4. **Consider the tail-optimised alternative** — §1.6.
5. Decide, enter, and log the reasoning in PROJECT.md §8.

---

## 3. Testing a squad honestly (out-of-sample)

A squad chosen to score well on 200 scenarios *will* score well on those 200.
The only real test is a scenario set it has never seen.

**CLI:**
```bash
uv run python scenario_generator.py --sources review solio --data-dir data \
    --scenarios 200 --out scenarios_v/ --horizon 12 --seed 99 \
    --enrich data/enrich.json

uv run python cvar_solver.py --scenario-dir scenarios_v/ \
    --bootstrap <ARCHIVE>/bootstrap_slim.json --weeks 4 --evaluate "<15 ids>"
```

In the UI: change *Seed* to 99, re-run B3b, then re-run B5a for each candidate.
Two minutes, and it is the difference between "this squad looks best" and "this
squad is best."

---

## 4. Chips

**First set (WC/FH/BB/TC) expires at the GW19 deadline, 13:30 GMT Sat 2 Jan.**
Unused chips are lost. There are essentially no blanks or doubles before then,
so first-set timing is a player-distribution problem, not a calendar one
(§4.7).

Run before finalising the GW1 squad — a BB-GW1 candidate cannot wait — then
roughly fortnightly.

**UI:** steps **C1** then **C2**. **CLI:**
```bash
uv run python chip_planner.py enumerate \
    --candidates "bb:1,fh:3,wc:7" "bb:2,wc:4,fh:8" --workers 2

uv run python chip_planner.py score --plans plans/ \
    --scenario-dir scenarios/ --bootstrap <ARCHIVE>/bootstrap_slim.json

uv run python chip_planner.py tc-table --scenario-dir scenarios/
```

Start narrow — each combination is a full solve and grids multiply. With
`--bb 1 2 3 --fh 3 4 8 --wc 4 7` you get 48 solves.

**Heed the verdict line.** When the top two plans sit inside the noise band the
table is telling you it cannot separate them; decide on team news, fixture
certainty, and the fact that an early chip carries no execution risk. Rank TC
weeks by the tail table, not by mean — you play TC for the haul, not the
average.

From ~GW15 the deadline guard matters: commit remaining chips to their best
surviving week rather than letting them expire.

**Second set (~GW25+):** wait for cup draws, then enumerate the 2–4 plausible
blank/double calendars by hand and score under each.

---

## 5. Periodic work

| When | What |
|---|---|
| ~GW6 | Calibrate scenario priors against realised outcomes: P(blank), P(haul), DefCon rates, clean-sheet frequency by position. Adjust `SHARE` in `scenario_generator.py` |
| ~GW10 | Score Review vs Solio accuracy from the archive. This is the evidence that would justify moving off 50/50 or retuning the elevenify dial |
| Never mid-season | Change the elevenify dial, blend weights, `decay_base` or `horizon` without recording it. They silently redefine what the archive means |

---

## 6. When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: sasoptpy` | no longer an upstream dependency | `uv add sasoptpy` |
| Validation fails on fixture structure | sources disagree about blanks/doubles | realign the exports; do not `--force` |
| Solve never finishes | too many scenarios, or `--weeks` too long | `--use-scenarios 16`, `--weeks 5`, set `--secs` |
| Solve stops at a gap | time limit hit | the objective is a lower bound; raise `--secs` or accept the bracket |
| Terminal closed, solve vanished | `uv run` dies with its shell | run from the UI, or `caffeinate -is nohup … &` |
| Stop button does nothing | HiGHS ignores SIGTERM mid C-call | fixed — escalates to SIGKILL after 4s; verify with `pgrep -fl stochastic_solver` |
| UI: "Couldn't reach the server" | server down, or a render error | check the terminal; hard-refresh |
| Plans/Compare empty or single-row | risk solvers write no plan files; `num_iterations` is 1 | expected — set `num_iterations: 3` |
| Chip `score` finds no plans | `enumerate` has not run | run C1 first |

---

## 7. Quick reference

| Tool | Question it answers |
|---|---|
| `archive_solve.py` | What is the EV-optimal plan — and preserve everything |
| `validate_sources.py` | Is the data safe to solve on |
| `solio_enrich.py` | Published clean-sheet and DefCon probabilities |
| `scenario_generator.py` | What might actually happen |
| `cvar_solver.py` | How risky is this squad against the field |
| `stochastic_solver.py` | Which move now is best given I will adapt later |
| `chip_planner.py` | When to play each chip, and whether the answer is real |

**UI step map:** B2 archive · B2b deep validation · B3a enrich · B3b scenarios ·
B4 transfer decision · A7 preseason squad · B5a score a squad · B5b tail-safe
alternative · C1 enumerate chips · C2 score chips.
