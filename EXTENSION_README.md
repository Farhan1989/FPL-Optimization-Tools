## Validate sources

uv run python validate_sources.py --sources solio review

## Archive solves

uv run python archive_solve.py --validate --note "review elevenify dial = 50%"

## Scenario generator

uv run python scenario_generator.py --sources review solio --scenarios 200 --out scenarios/ --horizon 12