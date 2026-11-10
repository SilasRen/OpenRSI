"""Single-GPU RL with the original single-step reward (baseline) or SOPO.

One code path for every arm, so the only difference between arms is the reward:

  --sopo-steps 0                     original OpenMLE reward (control)
  --sopo-steps 2 --reward-mode best  SOPO simple version: best utility the branch reaches
  --sopo-steps 2 --reward-mode credit --lam 0.9
                                     SOPO lambda version (lambda=0 == control reward)
  --curriculum                       replay (task, operator, parent) states by SOPO credit
                                     variance among siblings that ran (sopo_bandit idea)

Each step: pick B states (task + operator + parent from the per-task program DB),
sample G siblings, execute them locally, continue every valid sibling K more steps
along its own branch with the real Improve/Debug prompts (no gradient), turn the
branch into a reward with the same `combine_reward` used by the full SOPO code,
group-normalise, and take one policy-gradient step on the root responses only.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # for sopo.combine_reward

from executor import Executor, extract_code  # noqa: E402
from operators import ProgramDB, build_messages, immediate_reward  # noqa: E402
from tasks import load_tasks  # noqa: E402
from sopo import combine_reward  # noqa: E402


def node_dict(result, op=None):
    return {"static_value": result.utility, "eligible": result.eligible, "code_category": result.category,
            "status": result.status, "score": result.score, **({"operator": op} if op else {})}


class Curriculum:
    """Variance-of-credit bandit over (task, op, parent) states; uniform with prob `explore`."""

    def __init__(self, rng: random.Random, explore: float = 0.3, decay: float = 0.99, ema: float = 0.2):
        self.rng, self.explore, self.decay, self.ema = rng, explore, decay, ema
        self.arms: dict[tuple, dict] = {}
        self.clock = 0

    def observe(self, key, rewards, eligible):
        vals = [r for r, ok in zip(rewards, eligible) if ok]
        var = statistics.pvariance(vals) if len(vals) >= 2 else 0.0
        self.clock += 1
        prev = self.arms.get(key)
        pri = var if prev is None else (1 - self.ema) * prev["priority"] + self.ema * var
        self.arms[key] = {"priority": pri, "updated": self.clock}

    def draw(self):
        if not self.arms or self.rng.random() < self.explore:
            return None
        keys = list(self.arms)
        w = [self.arms[k]["priority"] * self.decay ** (self.clock - self.arms[k]["updated"]) for k in keys]
        if sum(w) <= 1e-12:
            return None
        return self.rng.choices(keys, weights=w, k=1)[0]


def run_lookahead(policy, executor, items, steps, args):
    """items: list of (task, root Node-like). Returns per-item list of node dicts (root first)."""
    from operators import Node

    chains = [[node_dict(n.result)] for _, n in items]
    current = [n for _, n in items]
    active = [i for i, (_, n) in enumerate(items) if n.result.category == "valid"]
    for _depth in range(steps):
        if not active:
            break
        prompts, ops = [], []
        for i in active:
            op = "improve" if current[i].result.eligible else "debug"
            ops.append(op)
            prompts.append(build_messages(op, items[i][0], current[i]))
        gens = policy.generate(prompts, 1, args.max_new_tokens, args.temperature)
        jobs = [(extract_code(g[0].text), items[i][0]) for g, i in zip(gens, active)]
        results = executor.run_many(jobs)
        for i, op, g, (code, _), res in zip(active, ops, gens, jobs, results):
            child = Node(-1, items[i][0].name, op, code, g[0].text, res, None)
            chains[i].append(node_dict(res, op))
            current[i] = child
    return chains


def main(argv=None):
    ap = argparse.ArgumentParser(description="SOPO small-scale RL")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--fake-policy", action="store_true", help="scripted policy, no GPU (pipeline test)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--groups", type=int, default=4, help="states per step (B)")
    ap.add_argument("--group-size", type=int, default=8, help="siblings per state (G)")
    ap.add_argument("--sopo-steps", type=int, default=0)
    ap.add_argument("--reward-mode", choices=["best", "credit"], default="best")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--op-probs", default="draft=0.5,improve=0.25,debug=0.25")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--mem-gb", type=float, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    rng = random.Random(args.seed)
    tasks = load_tasks(args.tasks, "train")
    by_name = {t.name: t for t in tasks}
    probs = {k: float(v) for k, v in (kv.split("=") for kv in args.op_probs.split(","))}
    if args.fake_policy:
        from policy import FakePolicy
        policy = FakePolicy(args.seed)
    else:
        import torch
        from policy import HFPolicy
        torch.manual_seed(args.seed)
        policy = HFPolicy(args.model, device=args.device, lr=args.lr, gen_batch=args.gen_batch)
    executor = Executor(args.workers, args.timeout, args.mem_gb)
    db = ProgramDB(rng)
    cur = Curriculum(rng) if args.curriculum else None
    log = open(out / "train_log.jsonl", "a")

    for step in range(1, args.steps + 1):
        t0 = time.time()
        states = []
        for _ in range(args.groups):
            key = cur.draw() if cur else None
            if key is not None and key[2] is not None and key[2] in db.nodes:
                task, op = by_name[key[0]], key[1]
                parent = db.nodes[key[2]]
            elif key is not None and key[2] is None:
                task, op, parent = by_name[key[0]], key[1], None
            else:
                task = rng.choice(tasks)
                op, parent = db.select(task.name, probs)
            states.append((task, op, parent))
        prompts = [build_messages(op, task, parent) for task, op, parent in states]
        gens = policy.generate(prompts, args.group_size, args.max_new_tokens, args.temperature)
        jobs = [(extract_code(g.text), states[i][0]) for i, row in enumerate(gens) for g in row]
        results = executor.run_many(jobs)

        flat_gens, rewards_all, stats = [], [], []
        roots = []
        k = 0
        for i, (task, op, parent) in enumerate(states):
            for g in gens[i]:
                res = results[k]
                k += 1
                node = db.add(task.name, op, res.code, g.text, res, parent)
                roots.append((task, node, op, parent, g))
        if args.sopo_steps > 0:
            chains = run_lookahead(policy, executor, [(t, n) for t, n, *_ in roots], args.sopo_steps, args)
        else:
            chains = [[node_dict(n.result)] for _, n, *_ in roots]

        advantages = []
        for i, (task, op, parent) in enumerate(states):
            grp = roots[i * args.group_size:(i + 1) * args.group_size]
            grp_chains = chains[i * args.group_size:(i + 1) * args.group_size]
            rewards = []
            for (t, node, _op, par, g), chain in zip(grp, grp_chains):
                imm = immediate_reward(op, node.result, par)
                if args.sopo_steps > 0 and node.result.category == "valid":
                    r, _ = combine_reward(chain, imm, args.reward_mode, args.lam)
                else:
                    r = imm
                rewards.append(r)
                flat_gens.append(g)
            mu = statistics.fmean(rewards)
            sd = statistics.pstdev(rewards)
            advantages += [0.0 if sd < 1e-6 else (r - mu) / (sd + 1e-6) for r in rewards]
            rewards_all += rewards
            elig = [n.result.eligible for _, n, *_ in grp]
            if cur is not None:
                cur.observe((task.name, op, parent.id if parent else None), rewards, elig)
            stats.append({
                "task": task.name, "op": op, "parent": parent.id if parent else None,
                "rewards": rewards,
                "root_utility": [n.utility for _, n, *_ in grp],
                "branch_best": [max([c["static_value"] for c in ch if c["eligible"]], default=0.0)
                                for ch in grp_chains],
                "valid_rate": sum(elig) / len(elig),
            })
        upd = policy.update(flat_gens, advantages)
        rec = {"step": step, "time": time.time() - t0, "executions": executor.calls,
               "mean_reward": statistics.fmean(rewards_all), "groups": stats, **upd}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[step {step}] reward={rec['mean_reward']:.3f} valid="
              f"{statistics.fmean(s['valid_rate'] for s in stats):.2f} execs={executor.calls} "
              f"t={rec['time']:.0f}s", flush=True)
    policy.save(str(out / "final"))
    print(f"saved {out / 'final'}")


if __name__ == "__main__":
    main()
