# FPL Pipeline Runbook — 2026/27

All commands run from the repo root (`FPL-Optimization-Tools/`).
Files: `archive_solve.py`, `validate_sources.py`, `scenario_generator.py`,
`cvar_solver.py`, `stochastic_solver.py` sit next to `paths.py`.

`<ARCHIVE>` below means the newest snapshot directory, e.g.
`../fpl-archive/2026-27/GW01/20260804T124510Z`.

---

## A. Preseason (once, before the GW1 deadline — Fri 21 Aug, 18:30 UK)

**A1. Refresh sources.** Download from both sites into `data/`:
- FPL Review export (GW1–14, elevenify dial at your chosen value) → `data/review.csv`
- Solio projection download → `data/solio.csv`

Check `data/user_settings.json`: `datasource: "mixed"`, `data_weights: {"review": 1, "solio": 1}`, `gap: 0`, `horizon: 12`. Set `num_iterations: 3` so the EV solver gives you genuinely distinct plans.

**A2. Validate + archive + EV solve** (one command):

    uv run python archive_solve.py --validate --note "review elevenify dial = 50%"

This snapshots ownership/prices/fixtures, gates on validation, runs the stock
solver, and commits everything to the archive. Note the printed `<ARCHIVE>` path.

**A3. Deep validation** (optional but cheap — adds the availability cross-check):

    uv run python validate_sources.py --sources solio review \
        --mixed mixed.csv --bootstrap <ARCHIVE>/bootstrap_slim.json

**A4. Generate scenarios (with published-data enrichment):**

    uv run python solio_enrich.py --out data/enrich.json
    uv run python scenario_generator.py --sources review solio \
        --data-dir data --scenarios 200 --out scenarios/ --horizon 12 \
        --enrich data/enrich.json

Check the printed calibration line: bias should be within ±0.05.

**A5. Risk-score the EV squad.** Take the 15 IDs from the A2 plan:

    uv run python cvar_solver.py --scenario-dir scenarios/ \
        --bootstrap <ARCHIVE>/bootstrap_slim.json --weeks 4 \
        --evaluate "411,426,..."

**A6. CVaR alternative squad:**

    uv run python cvar_solver.py --scenario-dir scenarios/ \
        --bootstrap <ARCHIVE>/bootstrap_slim.json --weeks 4 \
        --lam 0.5 --alpha 0.2 --compare-ev

**A7. Stochastic cross-check + VSS:**

    uv run python stochastic_solver.py --scenario-dir scenarios/ --preseason \
        --weeks 4 --use-scenarios 16 --lam 0 --gap 0.01 \
        --bootstrap <ARCHIVE>/bootstrap_slim.json --vss

**A8. Decide.** You now have: the EV plan (A2), its tail profile (A5), a
risk-adjusted alternative (A6), and the uncertainty-aware pick with a VSS
number telling you how much the stochastic view added (A7). Pick, enter on
the FPL site, done. Repeat A1–A7 once more on deadline day morning with
fresh CSVs — projections move as team news lands.

---

## B. Every mid-season gameweek

Run the sequence 24–48h before the deadline; re-run A-style on deadline
morning if there is major team news.

**B1. Refresh** `data/review.csv` and `data/solio.csv`.

**B2. Archive + validate + EV solve:**

    uv run python archive_solve.py --validate --push \
        --note "review elevenify dial = 50%"

If validation fails, read `validation.log` — fix the data, don't `--force`
past it unless you understand exactly why it fired.

**B3. Enrichment + scenarios:**

    uv run python solio_enrich.py --out data/enrich.json
    uv run python scenario_generator.py --sources review solio \
        --data-dir data --scenarios 200 --out scenarios/ --horizon 12 \
        --enrich data/enrich.json

The enrich step pulls Solio's published team clean-sheet odds and per-player
DefCon trigger probabilities (free public endpoint, 4-hourly) and replaces
the generator's inferred values for the first gameweek, means conserved. If
the fetch or parse fails it refuses to write and the generator runs
unenriched — never blocked, just less refined.

**B4. Stochastic transfer decision.** Current squad IDs, banked FTs, bank:

    uv run python stochastic_solver.py --scenario-dir scenarios/ \
        --squad "id1,id2,...,id15" --fts 2 --itb 0.5 \
        --weeks 5 --use-scenarios 16 --lam 0.3 --gap 0.01 \
        --bootstrap <ARCHIVE>/bootstrap_slim.json

Read the stage-1 line (the committed move) and the recourse branches (what
you'd likely do next week in each world).

**B5. Cross-check vs the EV plan.** If B2 and B4 recommend the same
transfer, done — high confidence. If they disagree, score both squads:

    uv run python cvar_solver.py --scenario-dir scenarios/ \
        --bootstrap <ARCHIVE>/bootstrap_slim.json --weeks 4 \
        --evaluate "<post-transfer squad A ids>"
    # repeat for squad B

Take the trade you're happy with: the EV move maximises expectation, the
stochastic/CVaR move protects the tail. Disagreement is information about
how assumption-sensitive this week is, not an error.

**B6. Execute** transfer + lineup + captain on the FPL site.

---

## C. Special weeks and chips

**Two chip sets, two different problems.** The first set (WC/FH/BB/TC)
expires at the GW19 deadline (13:30 GMT, Sat 2 Jan) and plays out on a flat
calendar — no blanks or doubles — so it is a *player-distribution* problem
with a hard expiry. The second set is a *calendar* problem: blanks and
doubles from ~GW19 onward dictate usage, and those firm up with cup draws.

**First-set workflow** (`chip_planner.py`):

    # 1. enumerate timings incl. community-consensus candidates
    uv run python chip_planner.py enumerate --bb 1 2 3 --fh 3 4 8 --wc 4 7 \
        --candidates "bb:1,fh:3,wc:7" "bb:2,wc:4,fh:8"

    # 2. score all plans under scenarios: win-share / regret / CVaR
    uv run python chip_planner.py score --plans plans/ \
        --scenario-dir scenarios/ --bootstrap <ARCHIVE>/bootstrap_slim.json

    # 3. Triple Captain weeks by TAIL mass, not mean
    uv run python chip_planner.py tc-table --scenario-dir scenarios/

Timing: run before finalising the GW1 squad (BB-GW1 candidates cannot wait),
then re-run fortnightly — the horizon reaches the full first-set window
around GW5–7. Heed the verdict line: when the top plans sit inside the noise
band, the table is telling you it's a coin flip — decide on team news and
fixture certainty, and treat the certainty of an early chip (you fully
control the GW1 squad) as a legitimate tiebreaker. From ~GW15, the deadline
guard matters: commit remaining chips to their best surviving week rather
than letting them expire.

**Second-set chips (~GW25+):** wait for cup draws, then enumerate over the
2–4 plausible blank/double calendars by hand and run `score` under each.

- **Wildcard weeks:** treat as preseason — run the full A sequence with the
  CVaR/stochastic solves in squad-from-scratch mode.
- **Mid-season budget caveat:** `stochastic_solver.py` and chip plans use
  buy prices only. When your squad carries large sell-price discounts,
  sanity-check affordability in the stock solver or on the FPL site.

## D. Periodic (not weekly)

- **After ~6 GWs:** calibrate the scenario generator against realised
  outcomes from the archive (P(blank), P(haul), DefCon rates, CS frequency
  by position). Adjust `SHARE` priors in `scenario_generator.py` if needed.
- **After ~10 GWs:** score review vs solio accuracy from archived
  projections vs realised points; revisit the 50/50 weights and the
  elevenify dial with actual evidence.
- **Never mid-season:** change the elevenify dial or blend weights without
  recording it via `--note` — it silently redefines what `review.csv` means
  in your archive.

## E. Quick reference — what each tool answers

| Tool | Question it answers |
|---|---|
| `archive_solve.py` | What does the EV-optimal plan look like, and preserve everything |
| `validate_sources.py` | Is the data safe to solve on |
| `scenario_generator.py` | What might actually happen, with realistic variance and correlation |
| `cvar_solver.py` | How risky is this squad vs the field; what would a tail-safe squad be |
| `stochastic_solver.py` | Which move NOW is best given I'll adapt later; was uncertainty worth modelling (VSS) |
| `chip_planner.py` | When to play each first-set chip — and whether the answer is real or a coin flip |
| `solio_enrich.py` | Published CS odds + DefCon probabilities to sharpen scenario decomposition |
