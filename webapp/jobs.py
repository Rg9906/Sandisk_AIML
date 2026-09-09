"""
Live pipeline runner.

Clicking Run in the console starts a REAL pipeline process and streams its stdout back.
Nothing is simulated: the console shows the same output you would get from the terminal,
and when the job finishes the dashboard reloads from the artifacts it just wrote.

Only the presets below can ever be started. The task name is looked up in this table and
the argument vector is built here, so nothing the browser sends reaches a shell.
"""
from __future__ import annotations

import itertools
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- presets
# `est` is what we have actually measured on this machine, not a guess. `stages` drives
# the progress rail: each entry is (label, regex that marks the stage as reached).
PRESETS: dict[str, dict] = {
    "recon_smoke": {
        "label": "Recon — the five proofs",
        "blurb": "Answers the five wave-0 questions that set the architecture. "
                 "No training, pure measurement.",
        "argv": ["-m", "src.recon", "--dataset", "smoke"],
        "est": 40,
        "demo": True,
        "stages": [("Load splits", r"train\s+dies="), ("Q1 failure rate", r"^## Q1|Q1 --"),
                   ("Q2 block order", r"Q2"), ("Q3 spatial", r"Q3"),
                   ("Q4 leakage", r"Q4"), ("Q5 overlap", r"Q5"), ("Write recon.md", r"recon\.md")],
    },
    "report_smoke": {
        "label": "Full pipeline — smoke (24 wafers)",
        "blurb": "Trains Model A, Model B pooling and the attention-MIL branch, runs the "
                 "baselines, calibrates, prices the threshold and regenerates every figure. "
                 "The whole system end to end.",
        "argv": ["-m", "src.report", "--dataset", "smoke", "--epochs", "3", "--mil-budget", "180"],
        "est": 110,
        "demo": True,
        "stages": [("Load splits", r"WaferLens report"),
                   ("Train pooling branch", r"\[Branch 1\]"),
                   ("Train attention-MIL", r"\[Branch 2\]"),
                   ("Blend & select", r"Model B Unified|blend weight"),
                   ("Incumbent baselines", r"baselines\.\.\."),
                   ("Calibrate", r"calibration"),
                   ("Economics", r"economics"),
                   ("Figures & attribution", r"figures\.\.\."),
                   ("Write submission", r"wrote results"),
                   ("Complete", r"report complete")],
    },
    "ablation_smoke": {
        "label": "Ablation waterfall — smoke",
        "blurb": "Rebuilds the model one component at a time and measures what each is "
                 "actually worth.",
        "argv": ["-m", "src.ablation", "--dataset", "smoke"],
        "est": 150,
        "demo": True,
        "stages": [("Load splits", r"ablation|Load"), ("Raw parametrics", r"raw 500"),
                   ("Wafer prior", r"wafer prior"), ("Spatial factory", r"spatial factory"),
                   ("Anomaly score", r"anomaly score"), ("Block pooling", r"block pooling")],
    },
    "imbalance_smoke": {
        "label": "Imbalance ladder — smoke",
        "blurb": "Runs all six rungs, measuring after each: weighting, threshold tuning, "
                 "per-wafer weights, focal loss and SMOTE.",
        "argv": ["-m", "src.imbalance", "--dataset", "smoke"],
        "est": 160,
        "demo": True,
        "stages": [("Load splits", r"imbalance|Load"), ("Rung 0-1", r"rung|0\."),
                   ("Threshold tuning", r"tuned threshold"), ("Focal loss", r"focal"),
                   ("SMOTE", r"SMOTE"), ("Write analysis", r"imbalance_analysis")],
    },
    "report_full": {
        "label": "Full pipeline — full dataset (200 wafers)",
        "blurb": "The real thing on all 200 wafers. This is what regenerates the numbers "
                 "you submit. Not a stage demo — it takes about 40 minutes.",
        "argv": ["-m", "src.report", "--dataset", "full"],
        "est": 2400,
        "demo": False,
        "stages": [("Load splits", r"WaferLens report"),
                   ("Train pooling branch", r"\[Branch 1\]"),
                   ("Train attention-MIL", r"\[Branch 2\]"),
                   ("Blend & select", r"Model B Unified|blend weight"),
                   ("Incumbent baselines", r"baselines\.\.\."),
                   ("Calibrate", r"calibration"),
                   ("Economics", r"economics"),
                   ("Figures & attribution", r"figures\.\.\."),
                   ("Write submission", r"wrote results"),
                   ("Complete", r"report complete")],
    },
}

# Lines worth pulling out of the stream and showing as live metric cards.
METRIC_PATTERNS = [
    (r"AUC-PR=([0-9.]+)\s+F1=([0-9.]+)", ("auc_pr", "f1")),
    (r"Brier ([0-9.]+) -> ([0-9.]+)", ("brier_raw", "brier_cal")),
    (r"blend weight w_mil=([0-9.]+)", ("w_mil",)),
    (r"val AUC-PR=([0-9.]+)", ("val_auc_pr",)),
    (r"report complete in ([0-9.]+)s", ("elapsed_s",)),
]

_counter = itertools.count(1)


@dataclass
class Job:
    id: str
    task: str
    label: str
    argv: list[str]
    est: int
    stages: list
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"          # running | done | failed | cancelled
    lines: list[dict] = field(default_factory=list)
    stage_at: int = 0
    metrics: dict = field(default_factory=dict)
    returncode: int | None = None
    _proc: subprocess.Popen | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ view
    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            tail = self.lines[since:]
            return {
                "id": self.id, "task": self.task, "label": self.label,
                "status": self.status, "returncode": self.returncode,
                "started": self.started, "finished": self.finished,
                "elapsed": round((self.finished or time.time()) - self.started, 1),
                "est": self.est,
                "stages": [s[0] for s in self.stages],
                "stage_at": self.stage_at,
                "metrics": dict(self.metrics),
                "lines": tail,
                "next": len(self.lines),
                "command": "python " + " ".join(self.argv),
            }

    def cancel(self) -> bool:
        with self._lock:
            if self.status != "running" or self._proc is None:
                return False
        self._proc.terminate()
        return True


_JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()
_ACTIVE: str | None = None


def presets() -> list[dict]:
    return [{"task": k, "label": v["label"], "blurb": v["blurb"],
             "est": v["est"], "demo": v["demo"],
             "command": "python " + " ".join(v["argv"]),
             "stages": [s[0] for s in v["stages"]]}
            for k, v in PRESETS.items()]


def active() -> Job | None:
    with _JOBS_LOCK:
        j = _JOBS.get(_ACTIVE) if _ACTIVE else None
    return j if (j and j.status == "running") else None


def get(job_id: str) -> Job | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def recent(limit: int = 12) -> list[dict]:
    with _JOBS_LOCK:
        js = sorted(_JOBS.values(), key=lambda j: j.started, reverse=True)[:limit]
    return [{"id": j.id, "task": j.task, "label": j.label, "status": j.status,
             "started": j.started, "elapsed": round((j.finished or time.time()) - j.started, 1),
             "metrics": dict(j.metrics)} for j in js]


def start(task: str, on_done=None) -> Job:
    """Launch one preset. Refuses if something is already running."""
    global _ACTIVE
    if task not in PRESETS:
        raise KeyError(f"unknown task '{task}'")
    if active() is not None:
        raise RuntimeError("a job is already running -- cancel it or wait")

    spec = PRESETS[task]
    job = Job(id=f"{next(_counter):03d}-{uuid.uuid4().hex[:6]}", task=task,
              label=spec["label"], argv=spec["argv"], est=spec["est"],
              stages=spec["stages"])
    with _JOBS_LOCK:
        _JOBS[job.id] = job
        _ACTIVE = job.id

    threading.Thread(target=_run, args=(job, on_done), daemon=True).start()
    return job


def _append(job: Job, text: str, kind: str = "out") -> None:
    with job._lock:
        job.lines.append({"t": round(time.time() - job.started, 2),
                          "kind": kind, "text": text})
        # Keep memory bounded on a 40-minute run without losing the head or the tail.
        if len(job.lines) > 4000:
            del job.lines[500:1500]


def _scan(job: Job, text: str) -> None:
    """Advance the stage rail and lift any metric the line happens to carry."""
    with job._lock:
        for i in range(job.stage_at, len(job.stages)):
            if re.search(job.stages[i][1], text, re.I):
                job.stage_at = i + 1
                break
        for pattern, keys in METRIC_PATTERNS:
            m = re.search(pattern, text)
            if m:
                for k, v in zip(keys, m.groups()):
                    job.metrics[k] = float(v)


def _run(job: Job, on_done) -> None:
    global _ACTIVE
    argv = [sys.executable, *job.argv]
    _append(job, f"$ python {' '.join(job.argv)}", "cmd")
    try:
        proc = subprocess.Popen(
            argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8",
                 "PYTHONUNBUFFERED": "1"},
        )
        with job._lock:
            job._proc = proc
        for raw in proc.stdout:                       # streams as the pipeline prints
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            _append(job, line)
            _scan(job, line)
        proc.wait()
        job.returncode = proc.returncode
        job.status = "done" if proc.returncode == 0 else (
            "cancelled" if proc.returncode in (-15, 15, 143) else "failed")
    except Exception as exc:                                          # noqa: BLE001
        _append(job, f"runner error: {type(exc).__name__}: {exc}", "err")
        job.status = "failed"
        job.returncode = -1
    finally:
        job.finished = time.time()
        with job._lock:
            job.stage_at = len(job.stages) if job.status == "done" else job.stage_at
        _append(job, f"-- {job.status} in {job.finished - job.started:.1f}s "
                     f"(exit {job.returncode}) --", "cmd")
        with _JOBS_LOCK:
            if _ACTIVE == job.id:
                _ACTIVE = None
        if on_done:
            try:
                on_done(job)
            except Exception:                                         # noqa: BLE001
                pass
