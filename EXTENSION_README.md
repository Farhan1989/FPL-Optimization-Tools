## Validate sources

uv run python validate_sources.py --sources solio review

## Archive solves

uv run python archive_solve.py --validate --note "review elevenify dial = 50%"

## Scenario generator

uv run python scenario_generator.py --sources review solio --scenarios 200 --out scenarios/ --horizon 12

## CVar Solver

uv run python cvar_solver.py --scenario-dir scenarios/ --bootstrap ../fpl-archive/2026-27/GW01/<ts>/bootstrap_slim.json --weeks 4 --lam 0.5 --alpha 0.2 --compare-ev

## Stochastic Solver

uv run python stochastic_solver.py --scenario-dir scenarios/ --preseason --weeks 3 --use-scenarios 12 --lam 0 --gap 0.01 --bootstrap ../fpl-archive/2026-27/GW01/<ts>/bootstrap_slim.json --vss