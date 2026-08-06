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

## 2. Architecture

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
                   ├─ cvar_solver.py         ← squad vs field, tail-aware
                   └─ stochastic_solver.py   ← two-stage transfer decision
```

Operational sequences live in `runbook.md`. Do not duplicate them here.

---

## 3. Decision log

Each entry: what was decided, why, and what evidence supports it.

### 3.1 Blend: Review + Solio, 50/50 — KEEP

Empirically insensitive. Sweeping the Review weight 0.4–0.6 produces an
**identical top-15 by EV**; 0.3–0.7 changes 1–3 of the top 50. Theory
agrees: for two forecasts of equal accuracy, 50/50 is minimum-variance
*regardless of correlation*, and estimated "optimal" weights routinely
underperform equal weights out of sample (weight-estimation error exceeds
the theoretical gain).

Measured source correlation on points: **0.748** — independent enough for
blending to do real work.

Revisit only with realised-outcome evidence (§6).

### 3.2 `decay_solio.py` — DELETED, do not resurrect

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

### 3.3 `gap: 0` — REQUIRED

Previously the solver terminated at ~0.11% gap while candidate plans were
separated by ~0.46 xPts — i.e. **inside the gap**, making the plan ranking
meaningless. With `gap: 0` the first 26/27 run found its true optimum only
in the final 17 seconds of a 714s solve (incumbent −387.96 → optimum
−388.02). Small, but real, and invisible without it.

### 3.4 elevenify — USE THE REVIEW DIAL, not a third source

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

### 3.5 Mikkel Tokvam — RULED OUT as a solver source

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

### 3.6 Horizon 12 — KEEP (Solio offers 19)

At `decay_base` 0.87, GW19 carries ~8% weight. Extending costs solve time
for negligible gain and breaks alignment with Review's 14.

**Verified safe:** gameweeks present in only one source are **not halved** —
the blend normalises by available weight (`weight` column drops to 0.5 but
mixed/solio ratio is 0.997). Single-source *players* are likewise not
halved. This was a live concern and is now closed.

### 3.7 Deterministic backtesting — ABANDONED, deliberately

No historical projection CSVs exist. Realised outcomes are recoverable from
the FPL API (and `vaastav/Fantasy-Premier-League`); **ex-ante projections are
not**. Three seasons would have been weak evidence anyway — the quantity of
interest resolves once per season, so n≈3.

This is precisely why `archive_solve.py` was built first: it makes the
missing dataset exist going forward.

---

## 4. Invariants — do not change mid-season

Changing any of these silently makes the archive non-comparable, which
defeats its purpose.

| Setting | Value | Note |
|---|---|---|
| elevenify dial | *(record in `--note`)* | Redefines `review.csv` |
| `data_weights` | `{review: 1, solio: 1}` | See §3.1 |
| `decay_base` | 0.87 | |
| `horizon` | 12 | |
| `gap` | 0 | See §3.3 |
| scenario `--seed` | fixed per run, logged | Reproducibility |

If one must change, record it in `--note` **and** add a line to §7 below.

---

## 5. Known limitations

**`scenario_generator.py`** — component priors (`SHARE`, `ASSIST_FRACTION`,
tilt scale) are documented constants from FPL scoring composition, *not*
calibrated to 26/27. Calibration verified only that sampled means reproduce
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
**VSS has never been successfully computed** (see §6).

**All scenario tools** — scenario reduction is stratified + moment-matched,
but small S still carries Monte Carlo noise in captain choice. Use ≥200
generated scenarios; treat marginal captain flips as noise.

**Chips are out of scope everywhere.** Chip weeks, blanks and doubles are
the stock solver's job.

---

## 6. Open items

1. **Compute VSS.** `stochastic_solver.py --vss`. If ≈0, the deterministic
   solver already captured the value and step 4 is a validation tool, not a
   replacement. Either answer is worth having.
2. **Calibrate scenario priors** (~GW6). Compare sampled P(blank), P(haul),
   DefCon hit rates and clean-sheet frequency by position against realised
   outcomes from the archive. Adjust `SHARE` in `scenario_generator.py`.
3. **Score source accuracy** (~GW10). Archived projections vs realised
   points, per source, per position, per horizon distance. This is the
   evidence that would justify moving off 50/50 (§3.1) or retuning the
   elevenify dial (§3.4).
4. **26/27 BPS rework.** Bonus is redistributed toward GK, full-backs and
   attackers. All projection models are miscalibrated on the bonus component
   until enough 26/27 data accumulates — expect the noisiest signal to be
   bonus for roughly the first 6–8 gameweeks.

---

## 7. Change log

| Date | Change | Rationale |
|---|---|---|
| 2026-08 | Pipeline built (steps 1–4), `decay_solio.py` retired, `gap: 0` set | Initial build |

*(Append here whenever an invariant in §4 changes.)*
