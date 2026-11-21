"""Run the inference test on real OpenMLE task packages with the paper's own prompts and scoring.

What is reused verbatim from the released code (imported, not copied):
  * prompts      prompt_builder.build_prompt (Draft / Improve / Debug) + the released public prompt
                 template OpenMLE-Gym/openmle_gym/templates/selfvalid0327-sft4.json
  * code parse   reward_func_utils.extract_code
  * judge        reward_func_utils._build_hack_check_messages / _parse_hack_check_completion
                 (sent to the policy's own endpoint, because the GPT judge is not available)
  * scoring      OpenMLE-Gym/openmle-sandbox/node_workers/read_and_metric.py with the task's
                 utils/metric.py and data/private/test_answer.csv, run exactly as the sandbox
                 dispatcher runs it (DATA_DIR=<task>/data/public, cwd = job dir, submission.csv in cwd)
  * reward       single step: group-adaptive power_clip (adaptive_reward_advantage_utils, launcher
                 defaults top1_top16, lower shift 0.25, alpha 1) + generate_mle.compute_mode_rewards
                 semantics (improve bonus 0.5 * delta, -1 empty, -0.5 hack, 0 unchanged code)
                 lookahead / ground truth: the static leaderboard power_clip used for SOPO
                 continuation nodes (generate_mle.compute_static_score_reward)

What differs from the paper's deployment (say so in any write-up):
  * programs run as local processes on the Slurm node (optionally inside Apptainer), not in the
    Docker AIO sandbox;
  * the hack judge is the policy model, not GPT;
  * the resource paragraph of the public system prompt is rewritten to the real job limits
    unless --keep-original-resources is given.
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL_DIR = HERE.parent
REPO = RL_DIR.parent.parent
sys.path.insert(0, str(RL_DIR))

# Released launcher defaults (scripts/run_openmle_rl_*.sh); must be set before reward_func_utils imports.
for _k, _v in {"POWER_CLIP_ALPHA": "1", "REWARD_MAPPING_MODE": "power_clip",
               "STATIC_REWARD_MAPPING_MODE": "power_clip", "VALIDATION_TEST_GAP_PENALTY_ENABLED": "0",
               "ENABLE_SELF_VALIDATION": "0"}.items():
    os.environ.setdefault(_k, _v)

import reward_func_utils as rfu  # noqa: E402
from adaptive_reward_advantage_utils import score_to_group_adaptive_reward  # noqa: E402

from executor import Result  # noqa: E402

def _first(env, *cands):
    if os.getenv(env):
        return Path(os.environ[env])
    for c in cands:
        if c.exists():
            return c
    return cands[0]


# Repo layout, or the cluster layout (<W>/sopo_rl = RL dir, <W>/sopo_gym_files = copied Gym files).
TEMPLATE = _first("OPENMLE_TEMPLATE", REPO / "OpenMLE-Gym" / "openmle_gym" / "templates" / "selfvalid0327-sft4.json",
                  RL_DIR.parent / "sopo_gym_files" / "templates" / "selfvalid0327-sft4.json")
SCORER = _first("OPENMLE_SCORER", REPO / "OpenMLE-Gym" / "openmle-sandbox" / "node_workers" / "read_and_metric.py",
                RL_DIR.parent / "sopo_gym_files" / "node_workers" / "read_and_metric.py")
ADAPTIVE = dict(mode=os.getenv("ADAPTIVE_REWARD_BOUND_MODE", "top1_top16"),
                lower_shift_ratio=float(os.getenv("ADAPTIVE_REWARD_BOUND_LOWER_SHIFT_RATIO", "0.25")),
                single_finite_reward_one=os.getenv("ADAPTIVE_BOUND_SINGLE_FINITE_REWARD_ONE", "1") != "0")
IMPROVE_DELTA_BONUS_COEF = float(os.getenv("IMPROVE_DELTA_BONUS_COEF", "0.5"))
FALLBACK_BEST, FALLBACK_WORST = 100.0, -100.0  # METRIC_STATIC_FALLBACK_SIGNED_*
FORBIDDEN = re.compile(r"data/private|test_answer|read_and_metric|\.\./private")


# ---------------------------------------------------------------- tasks

@dataclass
class OpenMLETask:
    name: str
    root: Path
    description: str
    data_description: str
    metadata: dict
    public_system: str
    public_user: str
    is_openmle: bool = True

    @property
    def metric(self) -> str:  # used only for log lines
        return "score"


def _higher_is_better_from_metric(path: Path):
    m = re.search(r"higher_is_better\s*(?::\s*bool)?\s*=\s*(True|False)", path.read_text())
    return None if m is None else m.group(1) == "True"


def _fallback_data_description(public: Path) -> str:
    """Same layout as the consumer inputs: file list with line counts, then a few header lines."""
    lines, heads = [], []
    for p in sorted(public.iterdir()):
        if p.is_file():
            try:
                with p.open(errors="replace") as f:
                    n = sum(1 for _ in f)
            except OSError:
                n = 0
            lines.append(f"{p.name} ({n} lines)")
            if p.suffix == ".csv":
                with p.open(errors="replace") as f:
                    head = [next(f, "").rstrip()[:500] for _ in range(3)]
                heads.append(f"-> {p.name} has {n} rows; first lines:\n" + "\n".join(head))
        elif p.is_dir():
            lines.append(f"{p.name}/ ({sum(1 for _ in p.rglob('*'))} files)")
    return "```\n" + "\n".join(lines) + "```\n\n" + "\n\n".join(heads)


def _public_prompts(resources: str | None):
    msgs = json.loads(TEMPLATE.read_text())
    system = next(m["content"] for m in msgs if m["role"] == "system")
    user = next(m["content"] for m in msgs if m["role"] == "user")
    if resources:
        system = re.sub(r"You are running in a sandboxed environment with the following resources:.*?\n(?=Carefully)",
                        "You are running in a sandboxed environment with the following resources:\n" + resources + "\n",
                        system, flags=re.S)
    return system, user


def load_openmle_tasks(roots, names=None, cpu_only=True, require_bounds=False, resources=None, limit=0):
    system, user = _public_prompts(resources)
    pkgs = []
    for r in map(Path, roots):
        pkgs += [r] if (r / "data" / "public").is_dir() else sorted(p for p in r.iterdir() if (p / "data" / "public").is_dir())
    wanted = set(names) if names else None
    tasks, skipped = [], {}
    for pkg in pkgs:
        if wanted and pkg.name not in wanted:
            continue
        if not ((pkg / "utils" / "metric.py").exists() and (pkg / "data" / "private" / "test_answer.csv").exists()):
            skipped[pkg.name] = "no metric.py or private answer"
            continue
        meta_path = pkg / "info" / "task_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.setdefault("task_name", pkg.name)
        if meta.get("higher_is_better") is None:
            meta["higher_is_better"] = _higher_is_better_from_metric(pkg / "utils" / "metric.py")
        meta["data_dir"] = str(pkg.resolve())
        if meta.get("higher_is_better") is None:
            skipped[pkg.name] = "metric direction unknown"
            continue
        if cpu_only and str(meta.get("cpu_gpu", "CPU")).upper() != "CPU":
            skipped[pkg.name] = f"needs {meta.get('cpu_gpu')}"
            continue
        best, worst = rfu._preferred_static_bounds_signed(meta, priority="leaderboard")
        if require_bounds and not (rfu._finite(best) and rfu._finite(worst)):
            skipped[pkg.name] = "no leaderboard/theoretical score bounds in info/task_metadata.json"
            continue
        dd = pkg / "info" / "data_description.txt"
        tasks.append(OpenMLETask(
            name=pkg.name, root=pkg.resolve(),
            description=(pkg / "data" / "public" / "description.txt").read_text(errors="replace")
            if (pkg / "data" / "public" / "description.txt").exists() else "",
            data_description=dd.read_text(errors="replace") if dd.exists() else _fallback_data_description(pkg / "data" / "public"),
            metadata=meta, public_system=system, public_user=user))
        if limit and len(tasks) >= limit:
            break
    for k, v in skipped.items():
        print(f"skip task {k}: {v}", flush=True)
    return tasks


# ---------------------------------------------------------------- prompts

class _ParentShim:
    """Minimal stand-in for program_database.Program as read by prompt_builder."""

    def __init__(self, node):
        self.code = node.code
        self.status_code = node.result.extra.get("status_code", -1)
        self.payload = node.result.extra.get("payload", {})


def build_messages(op: str, task: OpenMLETask, parent=None) -> list[dict]:
    from prompt_builder import build_prompt
    system, user = build_prompt(op, task.description, task.data_description, task.public_system, task.public_user,
                                virtual_data_dir=str(task.root), parent_program=_ParentShim(parent) if parent else None)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- rewards

def static_utility(score, meta) -> float:
    """compute_static_score_reward: leaderboard-first power_clip, used for SOPO continuation nodes."""
    if score is None or not rfu._finite(score):
        return 0.0
    m = {k: v for k, v in meta.items() if not k.startswith("dynamic_bound")}
    return float(rfu.score2reward_with_static_priority(float(score), m, mode=os.environ["STATIC_REWARD_MAPPING_MODE"],
                                                        priority="leaderboard", fallback_best=FALLBACK_BEST,
                                                        fallback_worst=FALLBACK_WORST))


def has_static_bounds(meta) -> bool:
    best, worst = rfu._preferred_static_bounds_signed(meta, priority="leaderboard")
    return rfu._finite(best) and rfu._finite(worst)


def task_scale(scores, meta, low_pct=0.1):
    """Frozen per-task utility scale for tasks without leaderboard bounds (most released packages).

    The paper's training reward maps scores with group-adaptive bounds (top1 / top16), so it needs no
    static bounds; but the static mapping used for SOPO continuation nodes then falls back to +-100 and
    becomes ~0.5 + score/200, which makes lookahead gains vanish. For the analysis every node of a task is
    mapped with one fixed scale instead: best = best score seen for the task, worst = its 10th percentile,
    power_clip (alpha=1) in between. Returns a function score -> utility in [0, 1]."""
    signed = sorted(rfu.signed_score(float(x), meta) for x in scores if x is not None and rfu._finite(x))
    if not signed:
        return None
    best = signed[-1]
    worst = signed[min(len(signed) - 1, int(low_pct * len(signed)))]
    if best - worst < 1e-12:
        worst = best - max(abs(best), 1.0) * 1e-6

    def f(score):
        if score is None or not rfu._finite(score):
            return 0.0
        v = (rfu.signed_score(float(score), meta) - worst) / (best - worst)
        return float(min(1.0, max(0.0, v)))
    return f


def rescale_records(records):
    """Re-map utility / static_value of every node with one frozen scale per task (see task_scale)."""
    by_task = {}
    for rec in records:
        if rec.get("static_bounds"):
            continue
        sc = by_task.setdefault(rec["task"], [])
        for sib in rec["siblings"]:
            sc.append(sib.get("score"))
            for ch in [sib["discovery"], *sib["repeats"]]:
                sc += [n.get("score") for n in ch]
        sc.append(rec.get("parent_score"))
    scales = {}
    for rec in records:
        if rec.get("static_bounds"):
            continue
        meta = {"higher_is_better": rec.get("higher_is_better", True)}
        f = scales.setdefault(rec["task"], task_scale(by_task[rec["task"]], meta))
        if f is None:
            continue
        if rec.get("parent_score") is not None:
            rec["parent_utility"] = f(rec["parent_score"])
        for sib in rec["siblings"]:
            sib["utility"] = f(sib.get("score")) if sib["eligible"] else 0.0
            for ch in [sib["discovery"], *sib["repeats"]]:
                for n in ch:
                    n["static_value"] = f(n.get("score")) if n["eligible"] else 0.0
    return records


class History:
    """Per-task finite test scores seen so far (the rollout's historical_scores for adaptive bounds)."""

    def __init__(self):
        self.scores: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def get(self, task):
        with self.lock:
            return list(self.scores.get(task, []))

    def add(self, task, scores):
        with self.lock:
            self.scores.setdefault(task, []).extend(float(s) for s in scores if s is not None and rfu._finite(s))


def group_single_step_rewards(task: OpenMLETask, op: str, parent, results: list[Result], history: History):
    """Original GSPO reward of each sibling: adaptive group bound -> power_clip -> compute_mode_rewards."""
    scores = [r.score if r.eligible else None for r in results]
    hist = history.get(task.name)
    parent_base = parent.result.extra.get("base_reward") if parent is not None else None
    out = []
    for r, s in zip(results, scores):
        if r.category == "empty":
            out.append(-1.0)
            continue
        if r.category in ("hack", "hack_verify"):
            out.append(-0.5)
            continue
        if s is None:
            base = 0.0
        else:
            base, _ = score_to_group_adaptive_reward(
                score=s, metadata=task.metadata, historical_scores=hist,
                current_group_scores=[x for x in scores if x is not None], mode=ADAPTIVE["mode"],
                reward_mapping_mode=os.environ["REWARD_MAPPING_MODE"], use_score2reward=True,
                lower_shift_ratio=ADAPTIVE["lower_shift_ratio"],
                single_finite_reward_one=ADAPTIVE["single_finite_reward_one"])
        r.extra["base_reward"] = base
        if op in ("improve", "debug") and parent is not None:
            if base == 0.0 or r.code.strip() == parent.code.strip():
                out.append(0.0)
                continue
            ref = parent_base if parent_base is not None else parent.utility
            out.append(base + IMPROVE_DELTA_BONUS_COEF * max(base - float(ref), 0.0))
        else:
            out.append(base)
    history.add(task.name, scores)
    return out


# ---------------------------------------------------------------- execution (sandbox equivalent)

def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run(cmd, cwd, env, timeout, log_path):
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            _kill(proc)
            proc.wait()
            return None, True


class LocalSandbox:
    """Reproduces the dispatcher contract of openmle-sandbox for one job and returns (status_code, payload)."""

    def __init__(self, job_timeout=3600, workdir=None, python=sys.executable, scorer_python=sys.executable,
                 apptainer_image=None, program_gpus="", mem_gb=0):
        self.job_timeout, self.python, self.scorer_python = job_timeout, python, scorer_python
        self.workdir = workdir or tempfile.gettempdir()
        self.apptainer_image, self.program_gpus, self.mem_gb = apptainer_image, program_gpus, mem_gb

    def _program_cmd(self, job, public):
        inner = f"export DATA_DIR={shlex.quote(str(public))} && {shlex.quote(self.python)} main.py"
        if self.mem_gb:
            inner = f"ulimit -v {int(self.mem_gb * 1024 * 1024)} && " + inner
        if self.apptainer_image:
            return ["apptainer", "exec", "--containall", "--cleanenv", "--no-home", "--bind",
                    f"{public}:{public}:ro,{job}:{job}", "--pwd", str(job), self.apptainer_image, "bash", "-c", inner]
        return ["bash", "-c", inner]

    def run(self, code: str, task: OpenMLETask):
        t0 = time.time()
        job = Path(tempfile.mkdtemp(prefix=f"{task.name[:40]}_", dir=self.workdir))
        try:
            (job / "main.py").write_text(code)
            public = task.root / "data" / "public"
            env = {"PATH": os.environ.get("PATH", ""), "HOME": str(job), "PYTHONUNBUFFERED": "1",
                   "CUDA_VISIBLE_DEVICES": self.program_gpus, "OMP_NUM_THREADS": os.environ.get("PROGRAM_THREADS", "6"),
                   "MKL_NUM_THREADS": os.environ.get("PROGRAM_THREADS", "6"), "LANG": "C.UTF-8",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            stdout = job / "stdout.txt"
            rc, timed_out = _run(self._program_cmd(job, public), job, env, self.job_timeout, stdout)
            out = stdout.read_text(errors="replace")[-20000:]
            log = "\n--- SANDBOX STDOUT START ---\n" + out + "\n--- SANDBOX STDOUT END ---\n"
            run_time = time.time() - t0
            if timed_out:
                return 200, {"status": "timeout", "running_time": run_time,
                             "result": {"result": "timeout", "score": None, "run_log": log + f"\nTimeout after {self.job_timeout}s"}}
            if rc != 0:
                return 200, {"status": "failed", "running_time": run_time,
                             "result": {"result": "code_execution_error", "score": None, "run_log": log}}
            sub = job / "submission.csv"
            if not sub.exists():
                return 200, {"status": "failed", "running_time": run_time,
                             "result": {"result": "submission_missing", "score": None, "run_log": log}}
            score_log = job / "score.txt"
            cmd = [self.scorer_python, str(SCORER), "-d", str(task.root),
                   "-a", str(task.root / "data" / "private" / "test_answer.csv"), "-p", str(sub)]
            senv = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1", "LANG": "C.UTF-8"}
            _run(cmd, job, senv, 1800, score_log)
            stext = score_log.read_text(errors="replace")
            m = re.search(r"##SCORE##\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)", stext)
            score = float(m.group(1)) if m else None
            if score is not None and not math.isfinite(score):
                score = None
            status = "success" if score is not None else "failed"
            log += "\n--- OUTPUT START ---\n" + stext[-4000:] + "\n--- OUTPUT END ---\n"
            return 200, {"status": status, "running_time": time.time() - t0,
                         "result": {"result": "success" if score is not None else "scoring_failed",
                                    "score": score, "run_log": log}}
        finally:
            shutil.rmtree(job, ignore_errors=True)


# ---------------------------------------------------------------- judge and policy (OpenAI-compatible server)

def _chat(url, model, messages, n=1, max_tokens=1024, temperature=0.0, timeout=3600, seed=None):
    import httpx
    body = dict(model=model, messages=messages, n=n, max_tokens=max_tokens, temperature=temperature)
    if seed is not None:
        body["seed"] = seed
    for attempt in range(5):
        try:
            r = httpx.post(url.rstrip("/") + "/chat/completions", json=body, timeout=timeout,
                           headers={"Authorization": "Bearer " + os.getenv("MODEL_API_KEY", "EMPTY")})
            r.raise_for_status()
            return r.json()["choices"]
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            print(f"model request failed ({e}); retry {attempt + 1}", flush=True)
            time.sleep(5 * (attempt + 1))


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).split("</think>")[-1]


class _Choice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class ModelJudge:
    def __init__(self, url, model, max_tokens=8192):
        self.url, self.model, self.max_tokens = url, model, max_tokens

    def __call__(self, code: str):
        msgs = rfu._build_hack_check_messages(code, require_holdout_validation=False)
        try:
            ch = _chat(self.url, self.model, msgs, max_tokens=self.max_tokens, temperature=0.0)
            content = strip_thinking(ch[0]["message"].get("content") or "")
        except Exception as e:  # noqa: BLE001  (fail open, like the original helper)
            return True, "valid", f"judge unavailable: {e}"
        comp = type("C", (), {"choices": [_Choice(content)]})()
        return rfu._parse_hack_check_completion(comp, require_holdout_validation=False)


@dataclass
class Gen:
    text: str
    prompt_ids: list = field(default_factory=list)
    response_ids: list = field(default_factory=list)


class OpenAIPolicy:
    def __init__(self, url, model, concurrency=16, seed=0):
        self.url, self.model, self.seed = url, model, seed
        self.pool = ThreadPoolExecutor(max_workers=concurrency)
        self.calls = 0

    def generate(self, prompts, n, max_new_tokens, temperature=1.0, top_p=1.0):
        def one(i, msgs):
            ch = _chat(self.url, self.model, msgs, n=n, max_tokens=max_new_tokens, temperature=temperature,
                       seed=self.seed * 1000003 + self.calls * 7919 + i)
            return [Gen(c["message"].get("content") or "") for c in ch]
        futs = [self.pool.submit(one, i, m) for i, m in enumerate(prompts)]
        self.calls += 1
        return [f.result() for f in futs]


def extract(text: str) -> str:
    try:
        return rfu.extract_code(strip_thinking(text)).strip()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------- executor used by infer_credit

class OpenMLEExecutor:
    def __init__(self, sandbox: LocalSandbox, judge=None, workers=8):
        self.sandbox, self.judge = sandbox, judge
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers))
        self.calls = 0

    def _one(self, code: str, task: OpenMLETask) -> Result:
        t0 = time.time()
        if not code.strip():
            return Result("empty", "skipped", None, 0.0, "No code generated", 0.0, code, {"status_code": -2,
                          "payload": {"error": "empty_code", "reason": "No code generated"}})
        if FORBIDDEN.search(code):
            return Result("hack", "skipped", None, 0.0, "reads private evaluation files", 0.0, code,
                          {"status_code": -1, "payload": {"error": "hack_detected", "reason": "reads private files"}})
        if self.judge is not None:
            ok, cat, reason = self.judge(code)
            if not ok:
                return Result(cat, "skipped", None, 0.0, reason, time.time() - t0, code,
                              {"status_code": -1, "payload": {"error": "hack_detected", "reason": reason},
                               "judge_reason": reason})
        status_code, payload = self.sandbox.run(code, task)
        res = payload.get("result") or {}
        score = res.get("score")
        status = {"success": "ok", "timeout": "timeout"}.get(payload.get("status"), "error")
        if payload.get("status") == "success" and score is None:
            status = "bad_submission"
        log = rfu.get_clear_log(res.get("run_log"))
        val = rfu.extract_validation_score(log)
        return Result("valid", status, score, self.utility(score, task) if score is not None else 0.0,
                      log[-4000:], time.time() - t0, code,
                      {"status_code": status_code, "payload": payload, "validation_score": val,
                       "result": res.get("result")})

    history = None
    scales: dict = {}

    def utility(self, score, task):
        if has_static_bounds(task.metadata):
            return static_utility(score, task.metadata)
        f = self.scales.get(task.name)
        return f(score) if f else 0.5  # before seeding there is no scale yet; refreshed by freeze_scales

    def freeze_scales(self, db, tasks):
        """Fix each task's run-time utility scale from its seed programs (used for parent selection only)."""
        for t in tasks:
            if has_static_bounds(t.metadata):
                continue
            nodes = db.programs(t.name)
            f = task_scale([n.result.score for n in nodes if n.result.eligible], t.metadata)
            if f is None:
                continue
            self.scales[t.name] = f
            for n in nodes:
                n.result.utility = f(n.result.score) if n.result.eligible else 0.0

    def group_rewards(self, task, op, parent, results):
        if self.history is None:
            self.history = History()
        return group_single_step_rewards(task, op, parent, results, self.history)

    def run_many(self, jobs):
        self.calls += len(jobs)
        futs = [self.pool.submit(self._one, c, t) for c, t in jobs]
        return [f.result() for f in futs]


# ---------------------------------------------------------------- scripted policy for pipeline tests

_FAKE_PROGRAMS = {
    "crash": "raise RuntimeError('boom')\n",
    "sample": ("import os, shutil\nd = os.environ['DATA_DIR']\n"
               "shutil.copy(os.path.join(d, 'sample_submission.csv'), 'submission.csv')\n"
               "print('Final Validation Score: 0.0')\n"),
    "model": ("import os, numpy as np, pandas as pd\n"
              "from sklearn.ensemble import HistGradientBoostingClassifier as C, HistGradientBoostingRegressor as R\n"
              "d = os.environ['DATA_DIR']\ntr = pd.read_csv(os.path.join(d, 'train.csv'))\n"
              "te = pd.read_csv(os.path.join(d, 'test.csv'))\nss = pd.read_csv(os.path.join(d, 'sample_submission.csv'))\n"
              "idc, tgt = ss.columns[0], ss.columns[1]\n"
              "feats = [c for c in te.columns if c in tr.columns and c != idc and pd.api.types.is_numeric_dtype(tr[c])]\n"
              "y = tr[tgt]\nreg = pd.api.types.is_float_dtype(y) and y.nunique() > 20\n"
              "m = (R() if reg else C()).fit(tr[feats], y.astype(float) if reg else y.astype(str))\n"
              "p = m.predict(te[feats])\nif not reg:\n    p = pd.Series(p).map(lambda v: type(y.iloc[0])(v) if not isinstance(y.iloc[0], (bool, np.bool_)) else v == 'True')\n"
              "pd.DataFrame({idc: te[idc], tgt: p}).to_csv('submission.csv', index=False)\n"
              "print('Final Validation Score: 0.5')\n"),
}
_NEXT = {"crash": "sample", "sample": "model", "model": "model"}


class FakeOpenMLEPolicy:
    def __init__(self, seed=0):
        import random
        self.rng = random.Random(seed)

    def generate(self, prompts, n, max_new_tokens=0, temperature=1.0, top_p=1.0):
        out = []
        for msgs in prompts:
            user = msgs[-1]["content"]
            row = []
            for _ in range(n):
                cur = next((k for k, v in _FAKE_PROGRAMS.items() if v.strip()[:60] in user), None)
                kind = _NEXT[cur] if cur and self.rng.random() < 0.7 else self.rng.choice(list(_FAKE_PROGRAMS))
                row.append(Gen(f"Plan: {kind}\n```python\n{_FAKE_PROGRAMS[kind]}```"))
            out.append(row)
        return out
