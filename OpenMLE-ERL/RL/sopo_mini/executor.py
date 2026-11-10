"""Docker-free local executor: run generated code in a subprocess and score it.

Replaces the OpenMLE sandbox service and the LLM hack-check API for the offline
experiment. Safety model (weaker than Docker, adequate inside your own Slurm job):

* the program runs in a fresh temporary directory holding only train.csv/test.csv;
* test labels never enter that directory (they stay in the parent process);
* address-space and CPU-time rlimits, a wall-clock timeout, single-threaded BLAS,
  and the whole process group is killed on timeout;
* a static check rejects network, subprocess and file-deletion APIs and any reference
  to the private label directory. Rejected programs get the original "hack" penalty.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tasks import Task, compute_metric, to_utility

FORBIDDEN = re.compile(
    r"\b(socket|requests|urllib|http\.client|subprocess|os\.system|os\.popen|os\.remove|os\.unlink|"
    r"shutil\.rmtree|shutil\.move|pty|ctypes|multiprocessing\.connection)\b|_private|\.\./")
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    blocks = CODE_BLOCK.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    # Unterminated block (truncated generation): take everything after the fence.
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text or "", re.S)
    return (m.group(1) if m else "").strip()


@dataclass
class Result:
    category: str          # valid | empty | hack
    status: str            # ok | error | timeout | bad_submission | skipped
    score: float | None
    utility: float
    log: str
    runtime: float
    code: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.category == "valid" and self.status == "ok" and self.score is not None

    def feedback(self, metric: str) -> str:
        """Feedback shown to the next Improve/Debug step (mirrors format_sandbox_feedback)."""
        lines = ["## Execution Result", f"**Status**: {self.status}", ""]
        if self.score is not None:
            lines += [f"**Score** ({metric}): {self.score:.6g}", ""]
        log = self.log.strip()
        if len(log) > 2000:
            log = log[:1000] + "\n... (truncated) ...\n" + log[-1000:]
        if log:
            lines += ["**Execution Log**:", "```", log, "```"]
        return "\n".join(lines)


def _limits(mem_gb: float, cpu_s: int):
    def apply():
        import resource
        os.setsid()
        mem = int(mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 ** 2, 512 * 1024 ** 2))
    return apply


def run_program(code: str, task: Task, timeout: float = 120, mem_gb: float = 6.0, workdir: str | None = None) -> Result:
    if not code.strip():
        return Result("empty", "skipped", None, 0.0, "No code block found in the response.", 0.0, code)
    if FORBIDDEN.search(code):
        hit = FORBIDDEN.search(code).group(0)
        return Result("hack", "skipped", None, 0.0, f"Rejected: forbidden API `{hit}`.", 0.0, code)
    tmp = tempfile.mkdtemp(prefix="sopo_run_", dir=workdir)
    try:
        for f in ("train.csv", "test.csv"):
            shutil.copy(task.public_dir / f, os.path.join(tmp, f))
        with open(os.path.join(tmp, "main.py"), "w") as fh:
            fh.write(code)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "TMPDIR": tmp, "PYTHONHASHSEED": "0",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}
        start = time.monotonic()
        proc = subprocess.Popen([sys.executable, "-I", "main.py"], cwd=tmp, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, preexec_fn=_limits(mem_gb, int(timeout) + 5))
        try:
            out, _ = proc.communicate(timeout=timeout)
            status = "ok" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, _ = proc.communicate()
            status = "timeout"
        runtime = time.monotonic() - start
        log = (out or b"").decode("utf-8", "replace")
        if status == "timeout":
            log += f"\nTimeoutError: program exceeded the {timeout:.0f}s time limit."
        score = None
        if status == "ok":
            score, msg = _score(os.path.join(tmp, "submission.csv"), task)
            if score is None:
                status = "bad_submission"
                log += "\n" + msg
        private = task.private()
        util = to_utility(score, private) if score is not None else 0.0
        return Result("valid", status, score, util, log[-6000:], runtime, code)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _score(path: str, task: Task):
    if not os.path.exists(path):
        return None, "submission.csv was not created."
    private = task.private()
    try:
        sub = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        return None, f"submission.csv could not be read: {e}"
    if task.target not in sub.columns or "id" not in sub.columns:
        return None, f"submission.csv must have columns id,{task.target}; got {list(sub.columns)[:10]}"
    y = np.asarray(private["y_test"])
    if len(sub) != len(y):
        return None, f"submission.csv has {len(sub)} rows, expected {len(y)}."
    sub = sub.sort_values("id")
    if not np.array_equal(sub["id"].to_numpy(), np.arange(len(y))):
        return None, "submission ids do not match test.csv ids."
    pred = sub[task.target].to_numpy()
    try:
        if pd.isna(pred).any():
            return None, "submission contains missing predictions."
        return compute_metric(task.metric, y, pred), ""
    except Exception as e:  # noqa: BLE001
        return None, f"could not score submission: {e}"


class Executor:
    def __init__(self, workers: int, timeout: float, mem_gb: float, workdir: str | None = None):
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers))
        self.timeout, self.mem_gb, self.workdir = timeout, mem_gb, workdir
        self.calls = 0

    def run_many(self, jobs: list[tuple[str, Task]]) -> list[Result]:
        self.calls += len(jobs)
        futs = [self.pool.submit(run_program, c, t, self.timeout, self.mem_gb, self.workdir) for c, t in jobs]
        return [f.result() for f in futs]
