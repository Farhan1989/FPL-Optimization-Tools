"""
FPL Solver Console — local web UI for the open-fpl-solver workflow.

Run:  uvicorn app:app --reload --port 8711
Then: http://127.0.0.1:8711

Everything the UI touches lives under SOLVER_ROOT (default: parent directory).
Nothing here shells out to anything that isn't declared in pipeline.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import parsers

# ---------------------------------------------------------------- paths

HERE = Path(__file__).resolve().parent
SOLVER_ROOT = Path(os.environ.get("SOLVER_ROOT", HERE.parent)).resolve()
DATA_DIR = Path(os.environ.get("FPL_DATA_DIR", SOLVER_ROOT / "data")).resolve()
RESULTS_DIR = Path(os.environ.get("FPL_RESULTS_DIR", SOLVER_ROOT / "data" / "results")).resolve()
SETTINGS_PATH = Path(os.environ.get("FPL_SETTINGS", SOLVER_ROOT / "data" / "user_settings.json")).resolve()
# Snapshots live outside the repo, e.g. ../fpl-archive/2026-27/GW01/<timestamp>/
ARCHIVE_ROOT = Path(os.environ.get("FPL_ARCHIVE_DIR", SOLVER_ROOT.parent / "fpl-archive")).resolve()
PIPELINE_PATH = HERE / "pipeline.json"
LOG_DIR = HERE / ".runs"

for d in (DATA_DIR, RESULTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_UPLOAD_SUFFIXES = {".csv", ".json"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

# Settings that must not drift. The UI warns loudly if these are changed.
# PROJECT.md section 4. Changing any of these mid-season makes the archive
# non-comparable, which is the whole point of having one.
INVARIANTS: Dict[str, Any] = {
    "gap": 0,
    "decay_base": 0.87,
    "horizon": 12,
    "data_weights": {"review": 1, "solio": 1},
}

app = FastAPI(title="FPL Solver Console")


# ---------------------------------------------------------------- runs


class Run:
    def __init__(self, run_id: str, step_id: str, command: str):
        self.id = run_id
        self.step_id = step_id
        self.command = command
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.returncode: Optional[int] = None
        self.lines: List[str] = []
        self.subscribers: List[asyncio.Queue] = []
        self.process: Optional[asyncio.subprocess.Process] = None
        self.cancelled = False

    def emit(self, kind: str, text: str) -> None:
        payload = json.dumps({"kind": kind, "text": text})
        if kind == "line":
            self.lines.append(text)
            if len(self.lines) > 4000:
                del self.lines[:1000]
        for q in list(self.subscribers):
            q.put_nowait(payload)

    def status(self) -> str:
        if self.finished_at is None:
            return "running"
        if self.cancelled:
            return "cancelled"
        return "ok" if self.returncode == 0 else "failed"

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "step_id": self.step_id,
            "command": self.command,
            "status": self.status(),
            "returncode": self.returncode,
            "started_at": self.started_at,
            "duration": (self.finished_at or time.time()) - self.started_at,
            "tail": self.lines[-40:],
        }


RUNS: Dict[str, Run] = {}
LAST_RUN_BY_STEP: Dict[str, str] = {}
LAST_PARAMS: Dict[str, Dict[str, str]] = {}
PIPELINE_LOCK = asyncio.Lock()


def load_pipeline() -> List[Dict[str, Any]]:
    with PIPELINE_PATH.open() as fh:
        return json.load(fh)["steps"]


def find_step(step_id: str) -> Dict[str, Any]:
    for step in load_pipeline():
        if step["id"] == step_id:
            return step
    raise HTTPException(404, f"No pipeline step called '{step_id}'.")


def resolve_archive() -> Optional[Path]:
    """Newest snapshot directory — the one archive_solve.py just printed.

    Identified by the bootstrap file every snapshot contains, rather than by
    parsing timestamps, so a changed naming scheme doesn't break it."""
    try:
        found = list(ARCHIVE_ROOT.glob("*/GW*/*/bootstrap_slim.json"))
    except OSError:
        return None
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime).parent


def build_argv(command: str, params: Dict[str, str]) -> List[str]:
    """Split the template first, then substitute — so a parameter value always
    lands as exactly one argv token and can never inject extra arguments.

    An optional parameter left blank removes its own token, and the flag
    immediately preceding it, so `--note {note}` disappears entirely rather
    than passing an empty string the script would have to defend against."""
    archive = resolve_archive()
    values = {
        "ROOT": str(SOLVER_ROOT),
        "DATA": str(DATA_DIR),
        "RESULTS": str(RESULTS_DIR),
        "SETTINGS": str(SETTINGS_PATH),
        "ARCHIVE": str(archive) if archive else "",
        **{k: str(v) for k, v in (params or {}).items()},
    }
    bare = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")

    argv: List[str] = []
    for token in shlex.split(command):
        match = bare.match(token)
        if match and not values.get(match.group(1), "").strip():
            if argv and argv[-1].startswith("-"):
                argv.pop()
            continue
        for key, value in values.items():
            token = token.replace("{" + key + "}", value)
        argv.append(token)
    return argv


def clean_targets(step: Dict[str, Any]) -> List[str]:
    """Delete directories a step declares stale before it runs.

    Regenerating with a smaller --scenarios leaves the higher-numbered files
    from the previous run in place, and every downstream tool globs the whole
    directory — so a shrunk run silently mixes two vintages. Cleaning is
    declared per step rather than left to memory."""
    removed = []
    for rel in step.get("clean", []):
        target = (SOLVER_ROOT / rel).resolve()
        if SOLVER_ROOT not in target.parents or not target.is_dir():
            continue
        shutil.rmtree(target, ignore_errors=True)
        removed.append(rel)
    return removed


async def launch(step: Dict[str, Any], params: Optional[Dict[str, str]] = None) -> Run:
    argv = build_argv(step["command"], params or {})
    command = shlex.join(argv)
    run = Run(uuid.uuid4().hex[:12], step["id"], command)
    RUNS[run.id] = run
    LAST_RUN_BY_STEP[step["id"]] = run.id

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    for rel in clean_targets(step):
        run.emit("meta", f"# cleared {rel}/ before running")
    run.emit("meta", f"$ {command}")
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(SOLVER_ROOT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        run.returncode = 127
        run.finished_at = time.time()
        run.emit("line", f"Command not found: {exc}")
        run.emit("end", "failed")
        return run

    run.process = proc

    async def pump() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            run.emit("line", raw.decode("utf-8", "replace").rstrip("\n"))
        run.returncode = await proc.wait()
        run.finished_at = time.time()
        body = "\n".join(run.lines)
        (LOG_DIR / f"{run.id}.log").write_text(body)
        # Stable per-step copy: the Plans panel reads these, so a rerun
        # replaces its own plan rather than accumulating stale variants.
        (LOG_DIR / f"{run.step_id}.latest.log").write_text(body)
        run.emit("end", run.status())

    asyncio.create_task(pump())
    return run


# ---------------------------------------------------------------- api


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    return {
        "paths": {
            "root": str(SOLVER_ROOT),
            "data": str(DATA_DIR),
            "results": str(RESULTS_DIR),
            "settings": str(SETTINGS_PATH),
            "archive_root": str(ARCHIVE_ROOT),
        },
        "files": list_files(),
        "archive": str(resolve_archive() or ""),
        "settings_defaults": settings_defaults(),
        "steps": [
            {
                **s,
                "last_run": RUNS[LAST_RUN_BY_STEP[s["id"]]].summary() if s["id"] in LAST_RUN_BY_STEP else None,
                "saved_params": LAST_PARAMS.get(s["id"], {}),
            }
            for s in load_pipeline()
        ],
        "invariants": INVARIANTS,
    }


def settings_defaults() -> Dict[str, str]:
    """Parameter defaults sourced from user_settings.json.

    `horizon` exists both as a step parameter and as a solver setting; if they
    drift you generate scenarios over one horizon and solve over another, with
    nothing to flag it. The settings file wins."""
    shared = ("horizon",)
    try:
        settings = json.loads(SETTINGS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: str(settings[k]) for k in shared if k in settings}


def list_files() -> List[Dict[str, Any]]:
    out = []
    for p in sorted(DATA_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_UPLOAD_SUFFIXES:
            stat = p.stat()
            out.append(
                {
                    "name": p.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "kind": parsers.classify_source(p),
                    "rows": parsers.count_rows(p),
                }
            )
    return out


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    saved, rejected = [], []
    for f in files:
        name = Path(f.filename or "").name
        if not name:
            continue
        if Path(name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
            rejected.append({"name": name, "why": "Only .csv and .json files."})
            continue
        target = DATA_DIR / name
        size = 0
        oversize = False
        with target.open("wb") as out:
            while chunk := await f.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    oversize = True
                    break
                out.write(chunk)
        if oversize:
            target.unlink(missing_ok=True)
            rejected.append({"name": name, "why": "Larger than 64 MB."})
        else:
            saved.append(name)
    return {"saved": saved, "rejected": rejected, "files": list_files()}


@app.delete("/api/files/{name}")
async def delete_file(name: str) -> Dict[str, Any]:
    target = (DATA_DIR / Path(name).name).resolve()
    if target.parent != DATA_DIR or not target.exists():
        raise HTTPException(404, "That file isn't in the data folder.")
    target.unlink()
    return {"files": list_files()}


@app.get("/api/settings")
async def get_settings() -> Dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {"exists": False, "settings": {}, "raw": "{}"}
    raw = SETTINGS_PATH.read_text()
    try:
        settings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"{SETTINGS_PATH.name} isn't valid JSON: {exc}")
    return {"exists": True, "settings": settings, "raw": raw}


class SettingsPayload(BaseModel):
    settings: Dict[str, Any]


@app.put("/api/settings")
async def put_settings(payload: SettingsPayload) -> Dict[str, Any]:
    settings = payload.settings
    warnings = [
        f"'{k}' is set to {settings[k]}, but this project treats {k}={v} as an invariant."
        for k, v in INVARIANTS.items()
        if k in settings and settings[k] != v
    ]
    if SETTINGS_PATH.exists():
        backup = SETTINGS_PATH.with_suffix(SETTINGS_PATH.suffix + f".bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(SETTINGS_PATH, backup)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")
    return {"saved": True, "warnings": warnings, "path": str(SETTINGS_PATH)}


class RunPayload(BaseModel):
    params: Dict[str, str] = {}


@app.post("/api/run/{step_id}")
async def run_step(step_id: str, payload: Optional[RunPayload] = None) -> Dict[str, Any]:
    step = find_step(step_id)
    params = payload.params if payload else {}
    missing = [p["name"] for p in step.get("params", []) if p.get("required") and not params.get(p["name"], "").strip()]
    if missing:
        raise HTTPException(400, f"Fill in {', '.join(missing)} before running this step.")
    LAST_PARAMS[step_id] = params
    run = await launch(step, params)
    return {"run_id": run.id, "command": run.command}


@app.post("/api/cancel/{run_id}")
async def cancel(run_id: str) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run or not run.process or run.finished_at is not None:
        raise HTTPException(404, "That run has already finished.")
    run.cancelled = True
    os.killpg(os.getpgid(run.process.pid), signal.SIGTERM)
    return {"cancelled": True}


@app.get("/api/run/{run_id}")
async def run_detail(run_id: str) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "No such run.")
    return {**run.summary(), "lines": run.lines}


@app.get("/api/stream/{run_id}")
async def stream(run_id: str) -> StreamingResponse:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "No such run.")

    queue: asyncio.Queue = asyncio.Queue()
    run.subscribers.append(queue)
    backlog = list(run.lines)

    async def gen():
        try:
            for line in backlog:
                yield f"data: {json.dumps({'kind': 'line', 'text': line})}\n\n"
            if run.finished_at is not None:
                yield f"data: {json.dumps({'kind': 'end', 'text': run.status()})}\n\n"
                return
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=20)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {payload}\n\n"
                if json.loads(payload)["kind"] == "end":
                    return
        finally:
            if queue in run.subscribers:
                run.subscribers.remove(queue)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/run-all")
async def run_all() -> Dict[str, Any]:
    """Run every enabled step in order, stopping at the first failure."""
    if PIPELINE_LOCK.locked():
        raise HTTPException(409, "A full run is already in progress.")

    steps = [s for s in load_pipeline() if s.get("enabled", True)]
    run_ids: List[str] = []

    async def sequence():
        async with PIPELINE_LOCK:
            for step in steps:
                params = LAST_PARAMS.get(step["id"], {})
                if any(p.get("required") and not params.get(p["name"], "").strip() for p in step.get("params", [])):
                    continue
                run = await launch(step, params)
                run_ids.append(run.id)
                while run.finished_at is None:
                    await asyncio.sleep(0.2)
                if run.returncode != 0:
                    break

    asyncio.create_task(sequence())
    await asyncio.sleep(0.1)
    return {"steps": [s["id"] for s in steps]}


@app.get("/api/plans")
async def plans() -> Dict[str, Any]:
    found = parsers.load_plans(RESULTS_DIR, LOG_DIR)
    return {
        "plans": found,
        "comparison": parsers.build_comparison(found),
        "results_dir": str(RESULTS_DIR),
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
