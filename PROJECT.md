# FPL Optimisation Pipeline — Project Record

**Season:** 2026/27 · **Goal:** consistent top-10k (best so far: ~2,500)
**Base:** `solioanalytics/open-fpl-solver` (Apache 2.0), unmodified.
All tooling here wraps or extends it; nothing patches `dev/solver.py`.

This file is the durable record. Code docstrings say *what* each tool does;
this says **why**, and what must not be changed. If you are picking this up
after a gap — or you are an AI assistant with no prior context — read this
first.

---

## 1. Why this project exists

The stated goal is *consistent* top-10k, not a high ceiling. That is a
statement about reducing left-tail variance, not raising expected points.
A single 2,500 finish is weak evidence of skill: a genuine top-10k process
has a wide outcome distribution, and 2,500 one year and 40,000 the next can
come from the identical process.

Everything below follows from that framing. It is why the roadmap ends in a
risk-aware objective rather than, say, price-change optimisation.

### The central insight

A linear expected-value objective **cannot** be made rank-aware by
subtracting effective ownership:

```
Σ_p Pts[p]·(x[p] − EO[p])  =  Σ_p Pts[p]·x[p] − Σ_p Pts[p]·EO[p]
                                                └──── constant ────┘
```

EO is exogenous, so the second term has no decision variable and the argmax
is unchanged. **Effective ownership does not enter through expectation. It
enters through variance.** Owning the 70%-owned captain barely moves your
expected differential; it collapses its variance. This is the whole reason
steps 2–4 exist.

---

## 2. Glossary

Read this first if any term below is unfamiliar. Everything else in this file
assumes it.

### The game

**Gameweek (GW)** — one round of Premier League fixtures. 38 per season.
**Deadline** — 90 minutes before the first kickoff of a gameweek; your team is
locked at that moment.
**Squad / XI / bench** — you own 15 players (2 GK, 5 DEF, 5 MID, 3 FWD) and
field 11 each week. The other 4 sit on the bench in a set order.
**Formation** — the legal shapes for an XI: exactly 1 GK, 3–5 DEF, 2–5 MID,
1–3 FWD, totalling 11.
**Autosub** — if a starter does not play, the first eligible bench player
replaces them automatically, provided the formation stays legal.
**Captain / vice** — captain scores double. Vice takes over if the captain
does not play.
**Free transfer (FT)** — one per gameweek, bankable up to five. Extra
transfers cost −4 points each ("a hit").
**Chip** — a one-shot power-up. **WC** wildcard (unlimited free transfers for
one week, permanent), **FH** free hit (unlimited for one week, squad reverts),
**BB** bench boost (all 15 score), **TC** triple captain (captain scores 3×).
For 2026/27 there are two sets; the first expires at the GW19 deadline.
**Blank / double gameweek** — a team has no fixture, or two. Chips are usually
timed around these; there are essentially none before GW19.
**DefCon** — defensive contribution. 2 points for a defender reaching 10
combined clearances/blocks/interceptions/tackles, or a mid/forward reaching 12
including ball recoveries. Introduced 2025/26; it revalued cheap defenders.
**BPS / bonus** — a hidden per-match index; the top three get 3/2/1 bonus
points. Reworked for 2026/27, so bonus projections are the least reliable
component early in the season.
**Price changes** — player prices drift with net transfers. You keep only 50%
of any profit when selling, so team value is slow to build and easy to lose.

### The data

**Projection / xPts** — a model's expected points for a player in a gameweek.
**xMins** — expected minutes. Drives everything else: no minutes, no points.
**FPL Review** — one of the two projection sources. ML plus performance data
plus market odds. Publishes ~14 gameweeks ahead.
**Solio** — the other source. Market-odds based with a stochastic model.
Publishes ~19 gameweeks ahead and a free 4-hourly data endpoint.
**elevenify** — a third model, now built into FPL Review as a slider rather
than a separate feed. See §4.4.
**Blend** — the 50/50 average of Review and Solio that the solver optimises
on, written to `mixed.csv`.
**Enrichment** — replacing two inferred quantities (team clean-sheet odds,
per-player DefCon probability) with Solio's published values. See
`solio_enrich.py`.

### The optimisation

**Solver** — software that finds the best decision subject to constraints.
Here: which 15 players, which XI, which transfers.
**MILP** — mixed-integer linear program. The mathematical form of the problem:
a linear objective over variables that must be whole numbers (you cannot own
0.4 of a player). **HiGHS** is the program that solves it.
**Objective** — the quantity being maximised. For the stock solver, decayed
expected points.
**Constraint** — a rule the answer must obey: 15 players, £100.0m budget, max
3 per club, legal formation.
**Decision variable** — something the solver chooses, e.g. "is player p in the
squad in gameweek w". Almost all are binary (0 or 1).
**Objective value / xPts total** — the score of a plan. Only comparable within
one data source and one settings set; never across.
**Gap** — the distance between the best solution found and the best possible.
`gap: 0` means proven optimal. A 2% gap means up to 2% better may exist. See
§4.3 — this matters more than it sounds.
**Horizon** — how many gameweeks ahead the solver plans.
**`decay_base`** — future gameweeks are discounted by this factor per week
(0.87 here), because plans change and distant projections are less reliable.
**Pool / pruning** — restricting the solver to the most plausible ~150 players
so it finishes in reasonable time.

### The risk layer

**Scenario** — one sampled version of how a set of gameweeks could actually
turn out: who plays, who scores, who keeps a clean sheet. 200 of them
approximate the distribution of outcomes.
**Monte Carlo** — estimating a distribution by sampling from it many times.
**Effective ownership (EO)** — the share of the field that owns a player,
counting captaincy twice. The field's expected score is built from this.
**Field** — the notional average manager you are ranked against. Modelled here
from the ownership snapshot in the archive.
**Delta (Δ)** — your score minus the field's, per scenario. Rank moves on Δ,
not on raw points.
**E[Δ]** — average Δ across scenarios. Positive means you beat the field on
average.
**CVaR (Conditional Value at Risk)** — the average Δ in the worst α fraction
of scenarios. `CVaR₂₀% = −15` means: in the worst fifth of worlds, you finish
15 points behind the field. **This is the number a consistency goal should be
judged on.**
**λ (lambda)** — the dial between maximising expectation (λ=0) and protecting
the tail (λ=1).
**α (alpha)** — which tail fraction CVaR looks at. 0.2 = worst 20%.
**Two-stage stochastic program** — a model where this week's decision is made
once, before you know what happens, and later weeks are allowed to adapt per
scenario.
**Non-anticipativity** — the rule that you cannot make today's decision
differently in each scenario, because today you do not know which one you are
in. It is what makes the model honest.
**Recourse** — the adapting later-stage decisions.
**VSS (Value of the Stochastic Solution)** — how many points planning under
uncertainty gains over planning on averages. Measured here at roughly +3 over
four gameweeks. See §4.9.
**In-sample vs out-of-sample** — a squad chosen to score well on a set of
scenarios will score well on *those* scenarios. Only a fresh scenario set
(different `--seed`) tests whether it is genuinely better.
**Noise band** — a margin inside which two results cannot be told apart.
Ordering within it is not signal.
**Win share / regret** — for chip plans: how often each plan is best across
scenarios, and how far behind it falls when it is not.

---

## 3. Architecture and tools

```
data/review.csv  data/solio.csv          ← manual download, weekly
        │
        ├─ validate_sources.py           ← gate: 6 checks, exit 1 on failure
        │
        ├─ archive_solve.py              ← snapshot + EV solve + git commit
        │      └─ ../fpl-archive/<season>/GW<nn>/<utc-ts>/
        │
        └─ scenario_generator.py         ← S sampled outcome paths
                   │
                   │      ▲ solio_enrich.py  ← published CS/DefCon probs
                   ├─ cvar_solver.py         ← squad vs field, tail-aware
                   ├─ stochastic_solver.py   ← two-stage transfer decision
                   └─ chip_planner.py        ← chip timing: enumerate → score
```

### What each tool is for

| Tool | Answers | Notes |
|---|---|---|
| `archive_solve.py` | What is the expected-points optimal plan — and preserve everything about this week | Wraps the stock solver. Snapshots ownership/prices/fixtures, gates on validation, commits to the archive repo |
| `validate_sources.py` | Is the data safe to solve on | Six checks; exit 1 blocks the solve |
| `solio_enrich.py` | Published clean-sheet and DefCon probabilities | Free 4-hourly endpoint; refuses to write on a thin parse |
| `scenario_generator.py` | What might actually happen | Emits S sampled outcome paths in the solver's own schema |
| `cvar_solver.py` | How risky is this squad against the field; what would a tail-safe squad look like | Squad + XI + captain only. No transfers, no chips |
| `stochastic_solver.py` | Which move *now* is best given I will adapt later; was uncertainty worth modelling | Two-stage. Stage 1 committed, stage 2 adapts per scenario |
| `chip_planner.py` | When to play each first-set chip, and whether the answer is real | Enumerate then score; reports win share and a noise verdict |
| `fpl-solver-ui/` | All of the above with buttons instead of flags | Local FastAPI console, no network exposure |

### The archive

`archive_solve.py` writes to a **separate git repo** (`../fpl-archive`), not
into the solver repo, so upstream pulls never conflict with weekly snapshots.
Each run creates `<season>/GW<nn>/<utc-timestamp>/` containing:

- `raw_review.csv`, `raw_solio.csv` — both sources exactly as consumed
- `mixed.csv` — the blend actually optimised on
- `bootstrap_slim.json` — **ownership and prices at the deadline.** This is the
  one artefact that cannot be reconstructed later, and the reason the archive
  exists
- `solutions.csv`, `solver_stdout.log`, `validation.log`
- `manifest.json` — settings, git SHA, artefact hashes, free-text note

Operational sequences live in `runbook.md`. Do not duplicate them here.

---

## 4. Decision log

Each entry: what was decided, why, and what evidence supports it.

### 4.1 Blend: Review + Solio, 50/50 — KEEP

Empirically insensitive. Sweeping the Review weight 0.4–0.6 produces an
**identical top-15 by EV**; 0.3–0.7 changes 1–3 of the top 50. Theory
agrees: for two forecasts of equal accuracy, 50/50 is minimum-variance
*regardless of correlation*, and estimated "optimal" weights routinely
underperform equal weights out of sample (weight-estimation error exceeds
the theoretical gain).

Measured source correlation on points: **0.748** — independent enough for
blending to do real work.

Revisit only with realised-outcome evidence (§7).

### 4.2 `decay_solio.py` — DELETED, do not resurrect

It applied a 0.97/GW multiplier to Solio to "match Review". Review has **no
decay**. Evidence, all from within-Review tests that never referenced Solio:

| Test | Review | If 0.97 decay were present |
|---|---|---|
| Log-linear fit, per-GW factor | 0.9945 | 0.9700 |
| Paired per-player GW38/GW29 (n=146) | 0.968 | 0.760 |

Effect of the bug: the nominal 50/50 silently drifted to ~57/43 in Review's
favour by late season. Decay now lives in exactly one place: `decay_base`.

**Related, and important:** Review's *xMins* genuinely decline (median
GW14/GW1 = 0.852) but that is **real minutes modelling, not decay** — the
ratio is dispersed (spread 0.107) with some players *rising* (injury
returns, e.g. Murillo 62→81). A uniform multiplier would show ~zero spread.
Do not flatten Review's minutes to match Solio; that would delete the most
differentiated signal in the blend. `validate_sources.py` tests for this
explicitly, because points-per-90 is blind to a decay applied to Pts and
xMins together.

### 4.3 `gap: 0` — REQUIRED

Previously the solver terminated at ~0.11% gap while candidate plans were
separated by ~0.46 xPts — i.e. **inside the gap**, making the plan ranking
meaningless. With `gap: 0` the first 26/27 run found its true optimum only
in the final 17 seconds of a 714s solve (incumbent −387.96 → optimum
−388.02). Small, but real, and invisible without it.

### 4.4 elevenify — USE THE REVIEW DIAL, not a third source

elevenify team ratings are built into FPL Review with a user-set blend dial.
Adding elevenify separately would double-count it at an unknown ratio.

**Dial arithmetic** (Solio is market-odds based, so it carries markets too):

```
final = (1 − 0.5·d)·markets + 0.5·d·elevenify        d = dial fraction
```

So a **50% dial gives ~75/25 markets/elevenify**, not 50/50. For true 50/50
exposure the dial goes to 100%. Current setting: **record it in every
`--note`** — the dial changes what `review.csv` *means*.

Sources agree broadly (Spearman 0.879 on goal difference); disagreement
concentrates mid-table, and markets is systematically higher on both goals
scored and conceded (elevenify is more bullish on clean sheets).

The earlier concern that elevenify excludes defensive contributions applies
to his own player tables on the Substack, **not** to this dial — Review
computes player points itself, DefCon included.

### 4.5 Mikkel Tokvam — RULED OUT as a solver source

The repo's own converter (`dev/data_parser.py`) hard-codes minutes:

```python
df.loc[df["Pos"].isin(["G", "D"]), "Weighted minutes"] = "90"
```

Every goalkeeper and defender is nailed at 90, every gameweek. The format
has a single minutes column, not one per GW. Blending it at 20% would drag
rotation-risk defenders back toward phantom minutes (White ARS: Review 28,
Solio 8, 50/50 blend 18, 40/40/20 blend **32**).

Compounding: the author acknowledges defender ratings are weaker; the tool
is positioned for attacking assets. In the DefCon era that is the wrong
specialisation, and Farhan's differential exposure is *entirely* cheap
defenders. Also fragile: fuzzy name matching with a hardcoded team dict that
lacks the 26/27 promoted sides, and `read_data` swallows parse failures
silently (`except Exception: continue`), so a broken Mikkel feed would not
error — it would quietly become a two-source blend.

Keep the subscription for written analysis if valued. Do not feed it to the
solver at any weight.

### 4.6 Horizon 12 — KEEP (Solio offers 19)

At `decay_base` 0.87, GW19 carries ~8% weight. Extending costs solve time
for negligible gain and breaks alignment with Review's 14.

**Verified safe:** gameweeks present in only one source are **not halved** —
the blend normalises by available weight (`weight` column drops to 0.5 but
mixed/solio ratio is 0.997). Single-source *players* are likewise not
halved. This was a live concern and is now closed.

### 4.7 Chips: two sets, two different problems — SOLVE-THEN-EVALUATE

26/27 has two chip sets. The first (WC/FH/BB/TC) **expires at the GW19
deadline** and plays out on a flat calendar — essentially no blanks/doubles
before GW19. The second set's value is dictated by spring blanks/doubles,
unknown until cup draws resolve.

Consequences:

- **First set is a player-distribution problem with a hard expiry**, not a
  calendar problem. On a flat calendar, chip weeks are separated only by the
  points *distribution* (TC is an option on the captain's right tail;
  ranking TC weeks by mean is close to meaningless). Use-it-or-lose-it makes
  it finite-horizon optimal stopping: the play/hold threshold falls as GW19
  approaches. The classic failure is dumping unused chips into GW17–18.
- **BB-GW1 has a structural argument**: bench engineering costs zero when
  the squad is built from scratch. Community consensus (BB1/FH3/WC7 or
  BB2/WC4/FH8) is treated as *candidates to score*, not answers to adopt —
  `chip_planner.py enumerate --candidates` puts them on identical footing
  with solver-found timings.
- **A stochastic chip MILP was rejected**: chip binaries inside a
  scenario-replicated model explode (FH's shadow squad doubles it again),
  and two-stage recourse would play TC on weeks it already "knows" the haul
  lands — look-ahead bias in its worst form. The tractable design is
  **enumerate deterministically (stock solver, `gap: 0`), then evaluate
  every plan under the 200 scenarios**, reporting win-share and regret
  rather than a ranked xPts column. When the top plans sit inside the noise
  band, the honest answer is "coin flip — decide on team news", and the
  tool says so explicitly.
- FH's first-half value is substantially *insurance* (injury crisis,
  fixture chaos), which no expectation-maximising solve prices. Policy:
  hold as insurance to ~GW15–16, then commit to the best surviving week.
- **Second set:** wait for cup draws (~GW25+), enumerate the 2–4 plausible
  blank/double calendars by hand, score under each.

### 4.8 Deterministic backtesting — ABANDONED, deliberately

No historical projection CSVs exist. Realised outcomes are recoverable from
the FPL API (and `vaastav/Fantasy-Premier-League`); **ex-ante projections are
not**. Three seasons would have been weak evidence anyway — the quantity of
interest resolves once per season, so n≈3.

This is precisely why `archive_solve.py` was built first: it makes the
missing dataset exist going forward.

---

## 5. Invariants — do not change mid-season

Changing any of these silently makes the archive non-comparable, which
defeats its purpose.

| Setting | Value | Note |
|---|---|---|
| elevenify dial | *(record in `--note`)* | Redefines `review.csv` |
| `data_weights` | `{review: 1, solio: 1}` | See §4.1 |
| `decay_base` | 0.87 | |
| `horizon` | 12 | |
| `gap` | 0 | See §4.3 |
| scenario `--seed` | fixed per run, logged | Reproducibility |

If one must change, record it in `--note` **and** add a line to §7 below.

---

## 6. Known limitations

**`scenario_generator.py`** — component priors (`SHARE`, `ASSIST_FRACTION`,
tilt scale) are documented constants from FPL scoring composition, *not*
calibrated to 26/27. `--enrich` (from `solio_enrich.py`) replaces the two
weakest inferred quantities — team CS probability and per-player DefCon —
with Solio's published values, but only for the FIRST horizon gameweek and
only for listed players/teams; later weeks remain prior-based. The parser now
reads the typed JSON feed (`latest.json`) rather than markdown tables, so
layout drift is no longer a failure mode; it still refuses to write on a thin
parse rather than emitting garbage. Extending enrichment beyond the first
gameweek needs a multi-GW source — see §7. Calibration verified only that sampled means reproduce
the blended projection (bias −0.02); the *shape* of the tails is unverified
until realised outcomes exist.

**`cvar_solver.py`** — field model is static ownership from one snapshot;
real EO drifts. Captaincy model (share ∝ own×proj among top 12) is an
approximation. No transfers, no chips: it is a squad/lineup/captain tool for
GW1, wildcards, and plan evaluation.

**`stochastic_solver.py`** — buy prices only, so the 50% sell-profit rule is
unmodelled; exact preseason, approximate mid-season. No chips. Two-stage
models have perfect foresight *within* each scenario, which overvalues
flexibility — hence `--max-recourse-transfers` (default 1) as a temper.
**VSS has never been successfully computed** (see §7).

**All scenario tools** — scenario reduction is stratified + moment-matched,
but small S still carries Monte Carlo noise in captain choice. Use ≥200
generated scenarios; treat marginal captain flips as noise.

**`chip_planner.py`** — `enumerate` wraps the stock solver's parallel
pattern but has **never run against the live FPL API** (built in a sandbox
without API access; parser and scoring validated on real solver logs and
scenarios). Scoring ignores autosubs (flat lineup totals) and uses buy
prices. Second-set calendar machinery is a manual process, not code.

**Chips remain out of scope in `cvar_solver.py` and
`stochastic_solver.py` — and this is a trap, not a footnote.** Both score
XI + captain (11 players); Bench Boost scores 15. So every CVaR figure
understates a BB squad by its entire bench, and understates *good* benches
most — precisely the axis a BB week selects on. `--evaluate --weeks N` also
holds a squad fixed for N gameweeks, which penalises a squad built as a
short-lived chip vehicle before a wildcard.

**Do not use a risk-solver squad in a chip week.** In August 2026 the
stochastic squad benched a backup goalkeeper with zero expected minutes — one
slot of fifteen dead on the week the chip was spent — while the EV squad,
which planned the chip, benched a playing keeper. Chip weeks belong to the
stock solver and `chip_planner.py`.

---

## 7. Open items

1. ~~**Compute VSS.**~~ **DONE: +3.10** (bracket [3.10, 8.45]; the stochastic
   solve hit its time limit at 2.60% gap while the comparison arm solved
   exactly, so the figure is a conservative lower bound). Corroborated by an
   earlier independent run at +2.90. This is the largest single effect measured
   in the project — larger than the plan spread, the chip spread and the decay
   bug combined. Remaining work: the dual bound moves only ~1.3 points in 900s,
   which is a weak LP relaxation rather than a bad incumbent. Tightening it
   needs valid inequalities linking scenarios — real work, not a one-liner.
1. **Re-score all candidate squads out-of-sample.** The CVaR squads were
   *selected* on the same 200 scenarios they were then scored on, which is an
   optimistically biased estimate; the EV and stochastic squads were scored
   out-of-sample. Regenerate with `--seed 99` and evaluate all four against the
   fresh set. Until then the only fair comparison in that run is λ=0 vs λ=0.5,
   which shares its in-sample status: **−0.60 E[Δ] for +2.71 CVaR₂₀%**, the
   risk trade working exactly as designed.
1. **Re-run every `--evaluate` from before the formation fix.** Those numbers
   were produced with a lineup that could be illegal, and are inflated.
2. **Calibrate scenario priors** (~GW6). Compare sampled P(blank), P(haul),
   DefCon hit rates and clean-sheet frequency by position against realised
   outcomes from the archive. Adjust `SHARE` in `scenario_generator.py`.
3. **Score source accuracy** (~GW10). Archived projections vs realised
   points, per source, per position, per horizon distance. This is the
   evidence that would justify moving off 50/50 (§4.1) or retuning the
   elevenify dial (§4.4).
4. **26/27 BPS rework.** Bonus is redistributed toward GK, full-backs and
   attackers. All projection models are miscalibrated on the bonus component
   until enough 26/27 data accumulates — expect the noisiest signal to be
   bonus for roughly the first 6–8 gameweeks.
5. **Commit and log the first-set chip plan.** Run `chip_planner.py`
   enumerate + score before GW1 (BB-GW1 candidates can't wait); record the
   chosen plan in §7 — it constrains every subsequent week. Verify
   `enumerate` on its first live run (§6). Deadline guard from ~GW15.

---

## 8. Change log

Newest last. Anything that changes an invariant in §5 must be recorded here.

### Phase 1 — infrastructure (Aug 2026)

- **`archive_solve.py`** built. Weekly snapshot of ownership, prices, fixtures,
  team state, both raw sources, the blend, the plan and the full settings, into
  a separate git repo. Motivated by having no historical projections to
  backtest against (§4.8): the point is that next year this data exists.
- **`validate_sources.py`** built as a gate. Six checks: source decay, xMins
  multiplier, fixture structure, blend normalisation, single-source coverage,
  API availability.
- **`decay_solio.py` retired** after measurement showed Review has no decay
  (§4.2). The script had been silently drifting a 50/50 blend to ~57/43.
- **`gap: 0` adopted** after finding candidate plans separated by less than the
  solver's own optimality gap (§4.3).

### Phase 2 — modelling under uncertainty (Aug 2026)

- **`scenario_generator.py`** built. Decomposes projections into components,
  samples team-level events first so teammates correlate, and re-samples player
  outcomes conditional on them. Four mean-bias bugs found and fixed during
  calibration (conceded-penalty double count, assist starvation, clean-sheet
  pooling leakage, truncation). Final calibration bias −0.02 points.
- **`cvar_solver.py`** built. Optimises `(1−λ)·E[Δ] + λ·CVaR_α[Δ]` against an
  effective-ownership field via the Rockafellar–Uryasev linearisation, so the
  model stays a MILP.
- **`stochastic_solver.py`** built. Two-stage program with a shared stage 1 and
  scenario-indexed recourse, plus `--vss`.
- **`chip_planner.py`** built (§4.7). Enumerate deterministically, then score
  every plan across the scenario set; report win share and regret rather than a
  ranked points column.

### Phase 3 — sharpening (Aug 2026)

- **`solio_enrich.py`** added; `scenario_generator.py --enrich`. Replaces the
  two weakest inferred quantities with published values. Discovered that a flat
  positional prior had one cheap defender's DefCon probability wrong by a
  factor of four — precisely the archetype this strategy relies on.
- **`sasoptpy` re-added** to `pyproject.toml`. Upstream dropped it when moving
  to `requires-python >=3.14`; the two custom solvers still need it. See §6.
- **`eta` bounded** to the achievable score range in both risk solvers. The
  `-1e6` placeholder let the feasibility-jump heuristic open at ~1e6 and spend
  most of the time limit climbing back.
- **Illegal-formation bug fixed** in `cvar_solver.py --evaluate`. The greedy XI
  capped positions at their maximum but never enforced the minimum, so it could
  field two defenders — a team FPL will not accept. On a constructed case it
  scored 80 where the best legal XI scores 73, i.e. it flattered every squad it
  evaluated. Replaced with exact enumeration over the eight legal formations.
  **Any `--evaluate` result from before this fix is inflated.**

### Phase 4 — the UI (Aug 2026)

`fpl-solver-ui/` built: a local FastAPI console over the same commands.
Notable fixes during review and use:

- Plans and Compare had **no data source** — the risk solvers write no plan
  files and nothing wrote to the configured results directory. The UI now
  parses the solver stdout it already captures, one plan per `Solution N`
  block, and deduplicates against the results directory.
- **`chip_planner score` had nothing to read**; an `enumerate` step was missing.
- **Stale scenario mixing.** Steps declare `"clean": [...]`, cleared before the
  step runs, because regenerating with a smaller `--scenarios` used to leave
  higher-numbered files behind for the next glob to pick up.
- **Stop did nothing.** SIGTERM cannot interrupt HiGHS mid C-call, and the
  error was swallowed by an empty catch. Now escalates to SIGKILL after four
  seconds, sweeps surviving descendants, and reports which happened.
- **`caffeinate -is`** wraps every step on macOS. Sleep suspended solves *and*
  consumed the solver's wall-clock limit.
- **Squad ID auto-fill.** Fifteen IDs typed by hand before every solve is slow
  and a mistyped ID is a valid solve of the wrong problem. Sources: your team
  from the archive snapshot, the stochastic stage-1 table, the EV squad, and
  your team with this week's transfers applied.
- **Seed field** on the scenarios step, to make out-of-sample re-scoring easy.
- Invariant warnings extended from `gap` alone to all four in §5.
- Palette moved from aubergine to the brand teal `#1C7480`, preserving every
  lightness value so contrast ratios were unchanged.

### Phase 5 — data plumbing (Aug 2026)

- **`solio_enrich.py` switched to the JSON feed.** Solio publishes an
  undocumented `api/data/latest.json` alongside `latest.md`, carrying the same
  figures as typed fields. Two gains: the markdown-layout dependency named in
  §6 disappears, and the values arrive unrounded — the markdown table quantises
  clean-sheet probability to integer percent, so `ARS 60%` is really 0.5982.
  Both parsers were run against the feed and agree to **≤0.0062 on CS and
  ≤0.0045 on DefCon**, exactly half a quantisation step, confirming they read
  the same data. The legacy path survives as `--md` with **no silent
  fallback**, so a JSON break fails loudly instead of quietly downgrading.
- **`generated` timestamp bug fixed.** The markdown regex expected a `... UTC`
  suffix while the page emits ISO-8601, so the field had been silently `None`
  in every `enrich.json` written to date. Cosmetic — nothing consumes it — but
  it was the §6 fragility demonstrating itself.

### Measurements worth remembering

| Quantity | Value | Where |
|---|---|---|
| Blend weight sensitivity | identical top-15 across 0.4–0.6 | §4.1 |
| Review implied decay | 0.9945/GW (i.e. none) | §4.2 |
| Candidate plan spread | 0.46 xPts | §4.3 |
| Chip timing spread | ~1.2 xPts, 48.5% vs 44.0% win share | §4.7 |
| Scenario calibration bias | −0.02 pts | §6 |
| Teammate correlation in scenarios | 0.57 same team, −0.03 across | §6 |
| **VSS** | **+3.10 over 4 GWs (two runs: +2.90, +3.10)** | §7 |
| λ=0 → λ=0.5 trade | −0.60 E[Δ] for +2.71 CVaR₂₀% | §7 |
