#!/usr/bin/env python3
"""
archive_solve.py — run open-fpl-solver and archive everything needed to
reconstruct the decision later.

Captures, per gameweek:
  * raw projection sources (review.csv, solio.csv, ...) as consumed
  * the blended mixed.csv the solver actually optimised on
  * the full effective settings (comprehensive + user + overlay)
  * every solution incl. per-GW squads, and objective under several decay bases
  * solver stdout
  * a slim bootstrap-static snapshot -> DEADLINE OWNERSHIP + PRICES
  * fixtures
  * your realised picks for the previous gameweek

Ownership at deadline is the single most valuable thing here: it is not
retrievable after the fact, and it is required for any effective-ownership
or variance work later.

Usage
-----
    python archive_solve.py                      # detect GW, solve, archive, commit
    python archive_solve.py --gw 7               # force gameweek
    python archive_solve.py --push               # also git push the archive
    python archive_solve.py --no-solve           # snapshot data only, no solve
    python archive_solve.py --dry-run            # show plan, touch nothing

Settings are NOT mutated. Logging flags are injected via a temporary
--config overlay, which solve.py merges above user_settings.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
RUN_DIR = REPO_ROOT / "run"

FPL_API = "https://fantasy.premierleague.com/api"

# July onward belongs to the season starting that calendar year.
SEASON_ROLLOVER_MONTH = 7

# Forced into every solve so the archive is self-describing.
FORCED_OPTIONS = {
    "export_data": "mixed.csv",
    "solutions_file": "solutions.csv",
    "save_squads": True,
    "solutions_file_player_type": "name",
    "report_decay_base": [0.85, 0.90, 0.95, 1.0],
    "print_squads": True,
    "print_transfer_chip_summary": True,
}

# Kept from bootstrap-static. Everything else is noise for our purposes.
ELEMENT_FIELDS = [
    "id",
    "web_name",
    "first_name",
    "second_name",
    "team",
    "element_type",
    "now_cost",
    "cost_change_start",
    "cost_change_event",
    "selected_by_percent",
    "transfers_in_event",
    "transfers_out_event",
    "status",
    "chance_of_playing_next_round",
    "total_points",
    "form",
    "ep_next",
    "ep_this",
    "minutes",
]


# ---------------------------------------------------------------- utilities


def log(msg: str) -> None:
    print(f"[archive] {msg}", flush=True)


def season_label(today: dt.date) -> str:
    """FPL seasons span Aug->May. July onward counts as the new season."""
    start = today.year if today.month >= SEASON_ROLLOVER_MONTH else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def solver_commit() -> str | None:
    try:
        return git(["rev-parse", "HEAD"], REPO_ROOT).stdout.strip()
    except Exception:
        return None


# ------------------------------------------------------------------ network


def fetch_json(url: str, timeout: int = 30):
    """Isolated so a network failure never costs us the local artifacts."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "fpl-archive/1.0"})
    resp.raise_for_status()
    return resp.json()


def detect_gameweek() -> int | None:
    try:
        boot = fetch_json(f"{FPL_API}/bootstrap-static/")
    except Exception as exc:
        log(f"could not reach FPL API for GW detection: {exc}")
        return None

    events = boot.get("events", [])
    for ev in events:
        if ev.get("is_next"):
            return ev["id"]
    for ev in events:
        if not ev.get("finished"):
            return ev["id"]
    return events[-1]["id"] if events else None


def snapshot_network(dest: Path, team_id: int | None, gw: int) -> dict:
    """Best-effort. Records what worked and what didn't."""
    status: dict[str, str] = {}

    # --- bootstrap-static, slimmed. Ownership + prices at deadline.
    try:
        boot = fetch_json(f"{FPL_API}/bootstrap-static/")
        slim = {
            "captured_utc": dt.datetime.now(dt.UTC).isoformat(),
            "gameweek": gw,
            "teams": [{k: t.get(k) for k in ("id", "name", "short_name", "strength")} for t in boot.get("teams", [])],
            "events": [
                {k: e.get(k) for k in ("id", "name", "deadline_time", "finished", "is_next", "average_entry_score")} for e in boot.get("events", [])
            ],
            "elements": [{k: el.get(k) for k in ELEMENT_FIELDS} for el in boot.get("elements", [])],
        }
        (dest / "bootstrap_slim.json").write_text(json.dumps(slim, indent=1))
        status["bootstrap"] = f"ok ({len(slim['elements'])} elements)"
    except Exception as exc:
        status["bootstrap"] = f"FAILED: {exc}"

    # --- fixtures
    try:
        fixtures = fetch_json(f"{FPL_API}/fixtures/")
        (dest / "fixtures.json").write_text(json.dumps(fixtures, indent=1))
        status["fixtures"] = "ok"
    except Exception as exc:
        status["fixtures"] = f"FAILED: {exc}"

    # --- your own team. Picks for the UPCOMING gw 404 until the deadline
    #     passes, so we take the previous gw, which is what lets us
    #     reconstruct counterfactuals.
    if team_id:
        try:
            entry = fetch_json(f"{FPL_API}/entry/{team_id}/")
            (dest / "entry.json").write_text(json.dumps(entry, indent=1))
            status["entry"] = "ok"
        except Exception as exc:
            status["entry"] = f"FAILED: {exc}"

        try:
            hist = fetch_json(f"{FPL_API}/entry/{team_id}/history/")
            (dest / "entry_history.json").write_text(json.dumps(hist, indent=1))
            status["entry_history"] = "ok"
        except Exception as exc:
            status["entry_history"] = f"FAILED: {exc}"

        if gw and gw > 1:
            try:
                picks = fetch_json(f"{FPL_API}/entry/{team_id}/event/{gw - 1}/picks/")
                (dest / f"picks_gw{gw - 1:02d}.json").write_text(json.dumps(picks, indent=1))
                status["picks_prev_gw"] = "ok"
            except Exception as exc:
                status["picks_prev_gw"] = f"FAILED: {exc}"
    else:
        status["entry"] = "skipped (no team_id)"

    return status


# -------------------------------------------------------------- settings io


def load_effective_settings() -> dict:
    with (DATA_DIR / "comprehensive_settings.json").open() as fh:
        opts = json.load(fh)
    with (DATA_DIR / "user_settings.json").open() as fh:
        opts.update(json.load(fh))
    return opts


def discover_sources(opts: dict) -> list[str]:
    """Which raw projection CSVs feed this solve."""
    ds = opts.get("datasource", "solio")
    if ds == "mixed":
        weights = opts.get("data_weights") or {}
        return sorted(weights.keys())
    return [ds]


# ------------------------------------------------------------------- solving


def build_overlay(dest: Path) -> Path:
    """Absolute paths, because solve.py runs with cwd=run/."""
    overlay = dict(FORCED_OPTIONS)
    overlay["solutions_file"] = str(dest / "solutions.csv")
    path = dest / "_overlay_config.json"
    path.write_text(json.dumps(overlay, indent=2))
    return path


def run_validator(sources: list[str], dest: Path, mixed: bool) -> int:
    """Gate the solve. Writes its output into the archive."""
    cmd = ["uv", "run", "python", "validate_sources.py", "--sources", *sources]
    if mixed:
        cmd += ["--mixed", "mixed.csv"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    out = proc.stdout + proc.stderr
    print(out)
    (dest / "validation.log").write_text(out)
    return proc.returncode


def run_solver(overlay: Path, dest: Path) -> tuple[int, float]:
    cmd = ["uv", "run", "python", "solve.py", "--config", str(overlay)]
    log(f"running: {' '.join(cmd)}  (cwd={RUN_DIR})")
    started = dt.datetime.now(dt.UTC)

    logfile = dest / "solver_stdout.log"
    with logfile.open("w") as fh:
        fh.write(f"# {' '.join(cmd)}\n# started {started.isoformat()}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=RUN_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()

    elapsed = (dt.datetime.now(dt.UTC) - started).total_seconds()
    return proc.returncode, elapsed


def collect_outputs(dest: Path, sources: list[str], expect_mixed: bool) -> dict:
    """
    Copy inputs and solver products into the archive.

    Raw sources are archived alongside the blend deliberately: with only
    mixed.csv you cannot re-derive alternative weightings, and you lose the
    cross-model disagreement signal entirely.
    """
    collected: dict[str, str] = {}

    def grab(src: Path, name: str) -> None:
        if src.exists():
            shutil.copy2(src, dest / name)
            collected[name] = sha256_file(src)[:16]
        else:
            collected[name] = "MISSING"

    for src_name in sources:
        grab(DATA_DIR / f"{src_name}.csv", f"raw_{src_name}.csv")

    # export_data / solutions_file may land in data/ or in run/.
    # mixed.csv only exists when blending, so don't warn about it otherwise.
    wanted = [("solutions.csv", "solutions.csv")]
    if expect_mixed:
        wanted.insert(0, ("mixed.csv", "mixed.csv"))
    for fname, archived_as in wanted:
        for candidate in (DATA_DIR / fname, RUN_DIR / fname, REPO_ROOT / fname, dest / fname):
            if candidate.exists() and candidate.parent != dest:
                shutil.copy2(candidate, dest / archived_as)
                collected[archived_as] = sha256_file(candidate)[:16]
                break
        else:
            if (dest / archived_as).exists():
                collected[archived_as] = sha256_file(dest / archived_as)[:16]
            else:
                collected.setdefault(archived_as, "MISSING")

    grab(DATA_DIR / "user_settings.json", "user_settings.json")
    grab(DATA_DIR / "comprehensive_settings.json", "comprehensive_settings.json")

    results_dir = DATA_DIR / "results"
    if results_dir.is_dir():
        payload = [p for p in results_dir.iterdir() if p.is_file() and p.name != ".gitkeep"]
        if payload:
            out = dest / "results"
            out.mkdir(exist_ok=True)
            for p in payload:
                shutil.copy2(p, out / p.name)
            collected["results/"] = f"{len(payload)} files"

    return collected


# ------------------------------------------------------------------ archive


def ensure_archive_repo(root: Path, dry: bool) -> None:
    if (root / ".git").exists():
        return
    if dry:
        log(f"[dry-run] would git init {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    git(["init"], root)

    # A fresh machine often has no global git identity; without one every
    # commit fails with exit 128. Set a local fallback only if needed.
    if git(["config", "user.email"], root, check=False).returncode != 0:
        git(["config", "user.email", "fpl-archive@localhost"], root, check=False)
        git(["config", "user.name", "FPL Archive"], root, check=False)
        log("no git identity found — set a local one for this repo")

    (root / ".gitattributes").write_text("*.csv -diff\n*.json -diff\n")
    (root / "README.md").write_text(
        "# FPL solver archive\n\n"
        "Per-gameweek snapshots from open-fpl-solver.\n\n"
        "Layout: `<season>/GW<nn>/<utc-timestamp>/`\n\n"
        "`bootstrap_slim.json` holds ownership and prices at deadline. "
        "That data is not retrievable retrospectively.\n"
    )
    git(["add", "-A"], root)
    git(["commit", "-m", "Initialise FPL solver archive"], root, check=False)
    log(f"initialised archive repo at {root}")


def commit_archive(root: Path, rel: Path, gw: int, push: bool, dry: bool) -> None:
    if dry:
        log(f"[dry-run] would commit {rel}")
        return
    git(["add", "-A"], root)
    result = git(["commit", "-m", f"GW{gw:02d} solve — {rel.as_posix()}"], root, check=False)
    if result.returncode != 0 and "nothing to commit" not in (result.stdout + result.stderr):
        log(f"commit warning: {result.stdout.strip()} {result.stderr.strip()}")
    else:
        log("committed")
    if push:
        res = git(["push"], root, check=False)
        log("pushed" if res.returncode == 0 else f"push failed: {res.stderr.strip()}")


# --------------------------------------------------------------------- main


def main() -> int:  # noqa: PLR0912, PLR0915
    ap = argparse.ArgumentParser(description="Run and archive an FPL solve.")
    ap.add_argument("--archive-root", type=Path, default=REPO_ROOT.parent / "fpl-archive")
    ap.add_argument("--gw", type=int, default=None, help="override gameweek detection")
    ap.add_argument("--team-id", type=int, default=None, help="overrides user_settings team_id")
    ap.add_argument("--no-solve", action="store_true", help="snapshot data only")
    ap.add_argument("--no-network", action="store_true", help="skip all API calls")
    ap.add_argument("--note", type=str, default=None, help="free-text note recorded in the manifest, e.g. 'review elevenify dial = 50%%'")
    ap.add_argument("--validate", action="store_true", help="run validate_sources.py first and abort on failure")
    ap.add_argument("--force", action="store_true", help="continue even if validation fails")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    opts = load_effective_settings()
    sources = discover_sources(opts)
    team_id = args.team_id or opts.get("team_id")

    gw = args.gw
    if gw is None and not args.no_network:
        gw = detect_gameweek()
    if gw is None:
        log("gameweek unknown — pass --gw explicitly")
        return 2

    now = dt.datetime.now(dt.UTC)
    rel = Path(season_label(now.date())) / f"GW{gw:02d}" / now.strftime("%Y%m%dT%H%M%SZ")
    dest = args.archive_root / rel

    log(f"gameweek {gw} | sources {sources} | horizon {opts.get('horizon')} | decay {opts.get('decay_base')}")
    log(f"destination {dest}")

    if args.dry_run:
        log("[dry-run] stopping before any writes")
        return 0

    ensure_archive_repo(args.archive_root, args.dry_run)
    dest.mkdir(parents=True, exist_ok=True)

    net_status: dict[str, str] = {}
    if not args.no_network:
        log("snapshotting FPL API (ownership, prices, fixtures, picks)...")
        net_status = snapshot_network(dest, team_id, gw)
        for k, v in net_status.items():
            log(f"  {k}: {v}")
    else:
        net_status = {"all": "skipped (--no-network)"}

    val_rc = None
    if args.validate:
        log("validating sources...")
        val_rc = run_validator(sources, dest, opts.get("datasource") == "mixed")
        if val_rc != 0:
            if args.force:
                log("validation FAILED — continuing because --force was given")
            else:
                log("validation FAILED — aborting solve. Re-run with --force to override.")
                (dest / "manifest.json").write_text(
                    json.dumps(
                        {
                            "captured_utc": now.isoformat(),
                            "gameweek": gw,
                            "aborted": "validation_failed",
                            "validation_exit": val_rc,
                            "note": args.note,
                        },
                        indent=2,
                    )
                )
                commit_archive(args.archive_root, rel, gw, args.push, args.dry_run)
                return 1

    rc, elapsed = 0, 0.0
    if not args.no_solve:
        overlay = build_overlay(dest)
        rc, elapsed = run_solver(overlay, dest)
        log(f"solver exited {rc} in {elapsed:.1f}s")
    else:
        log("skipping solve (--no-solve)")

    collected = collect_outputs(dest, sources, opts.get("datasource") == "mixed")

    manifest = {
        "captured_utc": now.isoformat(),
        "gameweek": gw,
        "season": season_label(now.date()),
        "solver_commit": solver_commit(),
        "solver_exit_code": rc,
        "solve_seconds": round(elapsed, 1),
        "team_id": team_id,
        "datasource": opts.get("datasource"),
        "data_weights": opts.get("data_weights"),
        "sources_archived": sources,
        "key_settings": {
            k: opts.get(k)
            for k in (
                "horizon",
                "decay_base",
                "ft_value_list",
                "ft_value",
                "xmin_lb",
                "ev_per_price_cutoff",
                "itb_value",
                "bench_weights",
                "vcap_weight",
                "hit_limit",
                "weekly_hit_limit",
                "num_iterations",
                "iteration_criteria",
                "randomized",
                "randomization_strength",
                "preseason",
            )
        },
        "forced_options": FORCED_OPTIONS,
        "validation_exit": val_rc,
        "note": args.note,
        "network_snapshot": net_status,
        "artifacts": collected,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))

    missing = [k for k, v in collected.items() if v == "MISSING"]
    if missing:
        log(f"WARNING missing artifacts: {missing}")

    commit_archive(args.archive_root, rel, gw, args.push, args.dry_run)
    log(f"done -> {dest}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
