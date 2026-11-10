"""Fixed-budget search evaluation on held-out tasks (the quantity SOPO targets).

For every held-out task and search seed, run a small greedy evolutionary search with
the checkpoint as operator: `--drafts` initial drafts, then Improve (a good scored
program) / Debug (a failed one) / Draft until `--budget` programs have been executed.
Record the best utility found after each program, i.e. the search curve.

All (task, seed) searches advance in lockstep so each search step is one batched
generation call. The same seeds are used for every checkpoint, so arms are compared
on identical task/seed pairs (paired comparison in report.py).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from executor import Executor, extract_code  # noqa: E402
from operators import ProgramDB, build_messages  # noqa: E402
from tasks import load_tasks  # noqa: E402


def choose(db: ProgramDB, task: str, t: int, drafts: int, rng: random.Random):
    if t < drafts:
        return "draft", None
    u = rng.random()
    if u < 0.7:
        p = db.pick_improve(task, temperature=0.1)
        if p is not None:
            return "improve", p
    if u < 0.9:
        p = db.pick_debug(task)
        if p is not None:
            return "debug", p
    return "draft", None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", required=True, help="checkpoint dir or HF id; 'fake' for scripted policy")
    ap.add_argument("--out", required=True, help="jsonl output")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--drafts", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--mem-gb", type=float, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if args.model == "fake" or (Path(args.model) / "FAKE").exists():
        from policy import FakePolicy
        policy = FakePolicy(0)
    else:
        from policy import HFPolicy
        policy = HFPolicy(args.model, device=args.device, gen_batch=args.gen_batch, train=False)
    tasks = load_tasks(args.tasks, args.split)
    executor = Executor(args.workers, args.timeout, args.mem_gb)
    runs = []
    for task in tasks:
        for seed in range(args.seeds):
            rng = random.Random(f"{task.name}/{seed}")
            runs.append(dict(task=task, seed=seed, rng=rng, db=ProgramDB(rng), curve=[], best=0.0, ops=[],
                             draft_utils=[], valid=[]))
    for t in range(args.budget):
        picks = [choose(r["db"], r["task"].name, t, args.drafts, r["rng"]) for r in runs]
        prompts = [build_messages(op, r["task"], p) for r, (op, p) in zip(runs, picks)]
        gens = policy.generate(prompts, 1, args.max_new_tokens, args.temperature)
        results = executor.run_many([(extract_code(g[0].text), r["task"]) for g, r in zip(gens, runs)])
        for r, (op, p), g, res in zip(runs, picks, gens, results):
            r["db"].add(r["task"].name, op, res.code, g[0].text, res, p)
            if res.eligible:
                r["best"] = max(r["best"], res.utility)
            r["curve"].append(r["best"])
            r["ops"].append(op)
            r["valid"].append(res.eligible)
            if op == "draft":
                r["draft_utils"].append(res.utility if res.eligible else 0.0)
        print(f"[eval {args.model}] node {t + 1}/{args.budget} mean best="
              f"{sum(r['best'] for r in runs) / len(runs):.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in runs:
            fh.write(json.dumps(dict(task=r["task"].name, seed=r["seed"], curve=r["curve"], ops=r["ops"],
                                     valid=r["valid"], draft_utils=r["draft_utils"])) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
