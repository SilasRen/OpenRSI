"""Inference-only check: is the multi-step (lookahead) reward a better GSPO signal than the single-step one?

No training, no optimizer. A frozen model plays the operator; everything runs on one GPU with the
local executor (no Docker, sandbox service or API).

For every state (task + Draft/Improve/Debug operator + parent program):

1. Sample G sibling responses (the GSPO group) and execute them. Their single-step reward r0 is
   exactly what the original RL step feeds into the group-normalised advantage.
2. Discovery: continue every runnable sibling K more steps along its own branch (Improve if it
   scored, Debug if it crashed) and turn the branch into multi-step rewards:
     best        : highest utility on the branch
     credit(lam) : r0 + sum_d lam**d * new-record gain at depth d     (the multiplicative discount)
   for every K' <= K.
3. Ground truth: continue every runnable sibling K steps again, R times, independently of
   discovery. V = mean over repeats of the best utility the branch reaches. V is "how good a search
   starting point this sibling really is", which is what SOPO claims to reward.

Then, per group, compare the two GSPO signals against V:
  * Spearman(reward, V) among runnable siblings. GSPO's advantage (r-mean)/std is a monotone map of
    the reward inside a group, so this is also the advantage's rank agreement with V.
  * Pairwise accuracy: for sibling pairs whose V differs by > --pair-margin, how often the sign of
    the advantage difference agrees with V.
  * Top-1 pick: V of the sibling with the largest advantage under each signal.
  * How often the multi-step reward exceeds the single-step reward, and how often the group's
    ranking (top-1) changes.

Discovery and ground-truth continuations are independent samples, so a multi-step reward cannot
look good merely by sharing noise with V. Both include the root's own utility.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from executor import Executor, extract_code  # noqa: E402
from operators import ProgramDB, build_messages, immediate_reward  # noqa: E402
from tasks import load_tasks  # noqa: E402
from sopo import combine_reward  # noqa: E402
from train import node_dict, run_lookahead  # noqa: E402


# ---------------------------------------------------------------- statistics helpers

def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def advantages(rewards):
    mu, sd = statistics.fmean(rewards), statistics.pstdev(rewards)
    return [0.0 if sd < 1e-6 else (r - mu) / (sd + 1e-6) for r in rewards]


def pair_accuracy(adv, value, margin):
    hit = tot = 0
    for i in range(len(adv)):
        for j in range(i + 1, len(adv)):
            dv = value[i] - value[j]
            if abs(dv) <= margin:
                continue
            tot += 1
            da = adv[i] - adv[j]
            hit += 1.0 if da * dv > 0 else (0.5 if abs(da) < 1e-9 else 0.0)
    return (hit, tot)


def top1_value(adv, value, rng):
    best = max(adv)
    cands = [i for i, a in enumerate(adv) if a >= best - 1e-9]
    return value[rng.choice(cands)]


def bootstrap_ci(xs, rng, n=2000):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return statistics.fmean(xs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


# ---------------------------------------------------------------- data collection

def branch_best(chain):
    return max([c["static_value"] for c in chain if c["eligible"]], default=0.0)


def build_states(policy, executor, tasks, args, rng, out):
    """Seed a small program DB per task with drafts, then sample (task, op, parent) states."""
    path = out / "states.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    db = ProgramDB(rng)
    prompts = [build_messages("draft", t) for t in tasks]
    gens = policy.generate(prompts, args.seed_drafts, args.max_new_tokens, args.temperature)
    jobs = [(extract_code(g.text), tasks[i]) for i, row in enumerate(gens) for g in row]
    results = executor.run_many(jobs)
    k = 0
    for i, row in enumerate(gens):
        if hasattr(executor, "group_rewards"):  # stores the original base reward used later as parent reference
            executor.group_rewards(tasks[i], "draft", None, results[k:k + len(row)])
        for g in row:
            db.add(tasks[i].name, "draft", results[k].code, g.text, results[k], None)
            k += 1
    if args.seed_scripted:
        # Simple fixed baselines (e.g. a linear model) give mid-quality parents with real headroom for Improve.
        from policy import _FAKE
        kinds = [x for x in args.seed_scripted.split(",") if x]
        sjobs = [(_FAKE[kind], t) for t in tasks for kind in kinds]
        sres = executor.run_many(sjobs)
        for (code, t), res, kind in zip(sjobs, sres, [kd for _ in tasks for kd in kinds]):
            db.add(t.name, "draft", code, f"SCRIPTED:{kind}\n```python\n{code}```", res, None)
    states = []
    if args.improve_only:
        lo, hi = (float(x) for x in args.parent_range.split(","))
        pool = {t.name: [n for n in db.programs(t.name) if n.result.eligible and lo <= n.utility <= hi]
                for t in tasks}
        usable = [t for t in tasks if pool[t.name]]
        if not usable:
            raise SystemExit(f"no scored parent with utility in [{lo}, {hi}]; widen --parent-range")
        for s in range(args.num_states):
            task = usable[s % len(usable)]
            states.append((task, "improve", rng.choice(pool[task.name])))
        print(f"improve-only: {len(usable)}/{len(tasks)} tasks have parents in [{lo}, {hi}]", flush=True)
    else:
        probs = {kk: float(v) for kk, v in (kv.split("=") for kv in args.op_probs.split(","))}
        for s in range(args.num_states):
            task = tasks[s % len(tasks)]
            op, parent = db.select(task.name, probs)
            states.append((task, op, parent))
    path.write_bytes(pickle.dumps(states))
    seeded = sum(r.eligible for r in results)
    print(f"seeded {len(results)} drafts ({seeded} scored); {len(states)} states: "
          f"{ {o: sum(st[1] == o for st in states) for o in ('draft', 'improve', 'debug')} }", flush=True)
    return states


def collect_chunk(policy, executor, states, args):
    from operators import Node

    prompts = [build_messages(op, task, parent) for task, op, parent in states]
    gens = policy.generate(prompts, args.group_size, args.max_new_tokens, args.temperature)
    jobs = [(extract_code(g.text), states[i][0]) for i, row in enumerate(gens) for g in row]
    results = executor.run_many(jobs)
    roots = []
    k = 0
    r0s = []
    for i, (task, op, parent) in enumerate(states):
        grp = results[k:k + len(gens[i])]
        if hasattr(executor, "group_rewards"):
            r0s += executor.group_rewards(task, op, parent, grp)
        else:
            r0s += [immediate_reward(op, res, parent) for res in grp]
        for g in gens[i]:
            res = results[k]
            k += 1
            roots.append((task, Node(-1, task.name, op, res.code, g.text, res, parent.id if parent else None)))
    discovery = run_lookahead(policy, executor, roots, args.steps, args)
    repeats = [run_lookahead(policy, executor, roots, args.steps, args) for _ in range(args.repeats)]
    records = []
    for i, (task, op, parent) in enumerate(states):
        sib = []
        for j in range(args.group_size):
            idx = i * args.group_size + j
            node = roots[idx][1]
            sib.append({
                "category": node.result.category, "status": node.result.status,
                "utility": node.utility, "eligible": node.result.eligible,
                "r0": r0s[idx],
                "score": node.result.score,
                "base_reward": node.result.extra.get("base_reward"),
                "judge": node.result.extra.get("judge_reason"),
                "log_tail": node.result.log[-400:] if node.result.status != "ok" else "",
                "discovery": discovery[idx],
                "repeats": [rep[idx] for rep in repeats],
            })
        records.append({"task": task.name, "op": op,
                        "parent_utility": parent.utility if parent else None,
                        "parent_scripted": bool(parent and parent.response.startswith("SCRIPTED:")),
                        "siblings": sib})
    return records


# ---------------------------------------------------------------- analysis

def signals(sib, op_parent_reward_fn, K, lams):
    """Return {signal_name: reward} for one sibling, using the first K lookahead steps."""
    r0 = sib["r0"]
    out = {"single": r0}
    if sib["category"] != "valid":
        for k in range(1, K + 1):
            out[f"best_K{k}"] = r0
            for lam in lams:
                out[f"credit_K{k}_l{lam:g}"] = r0
        return out
    for k in range(1, K + 1):
        chain = sib["discovery"][:k + 1]
        out[f"best_K{k}"] = combine_reward(chain, r0, "best", 1.0)[0]
        for lam in lams:
            out[f"credit_K{k}_l{lam:g}"] = combine_reward(chain, r0, "credit", lam)[0]
    return out


def sibling_value(sib, future_only):
    """V: mean best utility over the independent continuations.

    future_only=True drops the root itself, so V measures only how far the branch can go *after* this sibling,
    i.e. how good a search starting point it is. With the root included, V shares the root's own score with the
    single-step reward, which mechanically favours `single`."""
    chains = sib["repeats"] or [sib["discovery"]]
    return statistics.fmean(branch_best(ch[1:] if future_only else ch) for ch in chains)


def analyse(records, args, future_only=True):
    rng = random.Random(args.seed + 7)
    lams = [float(x) for x in args.lams.split(",")]
    names = None
    per_signal = {}
    exceed = {}
    top_change = {}
    n_groups = 0
    for rec in records:
        sibs = rec["siblings"]
        sig = [signals(s, None, args.steps, lams) for s in sibs]
        if names is None:
            names = list(sig[0])
            per_signal = {n: {"spearman": [], "pair_hit": 0.0, "pair_tot": 0, "top1": []} for n in names}
            exceed = {n: [0, 0] for n in names}
            top_change = {n: [0, 0] for n in names}
        runnable = [i for i, s in enumerate(sibs) if s["category"] == "valid"]
        if len(runnable) < 2:
            continue
        n_groups += 1
        value = [sibling_value(sibs[i], future_only) for i in runnable]
        single_adv = advantages([sig[i]["single"] for i in range(len(sibs))])
        single_top = max(range(len(sibs)), key=lambda i: (single_adv[i], -i))
        for n in names:
            rewards = [sig[i][n] for i in range(len(sibs))]
            adv = advantages(rewards)  # GSPO normalises over the whole group, invalid siblings included
            adv_r = [adv[i] for i in runnable]
            per_signal[n]["spearman"].append(spearman([rewards[i] for i in runnable], value))
            h, t = pair_accuracy(adv_r, value, args.pair_margin)
            per_signal[n]["pair_hit"] += h
            per_signal[n]["pair_tot"] += t
            per_signal[n]["top1"].append(top1_value(adv_r, value, rng))
            for i in runnable:
                exceed[n][0] += rewards[i] > sig[i]["single"] + 1e-9
                exceed[n][1] += 1
            top = max(range(len(sibs)), key=lambda i: (adv[i], -i))
            top_change[n][0] += top != single_top
            top_change[n][1] += 1
    summary = {"v_future_only": future_only, "groups_used": n_groups, "groups_total": len(records), "signals": {}}
    base_top1 = per_signal["single"]["top1"] if per_signal else []
    for n in names or []:
        p = per_signal[n]
        # Compare on the same groups: those where both this signal and `single` have a defined rank correlation.
        both = [i for i, (a, b) in enumerate(zip(p["spearman"], per_signal["single"]["spearman"]))
                if a is not None and b is not None]
        sp = bootstrap_ci([p["spearman"][i] for i in both], rng)
        diff = [a - b for a, b in zip(p["top1"], base_top1)]
        sp_diff = [p["spearman"][i] - per_signal["single"]["spearman"][i] for i in both]
        summary["signals"][n] = {
            "spearman_mean": sp[0], "spearman_ci": [sp[1], sp[2]],
            "spearman_minus_single": bootstrap_ci(sp_diff, rng),
            "spearman_groups": len(both),
            "pair_accuracy": p["pair_hit"] / p["pair_tot"] if p["pair_tot"] else None,
            "pairs": p["pair_tot"],
            "top1_value": statistics.fmean(p["top1"]) if p["top1"] else None,
            "top1_minus_single": bootstrap_ci(diff, rng),
            "frac_reward_gt_single": exceed[n][0] / exceed[n][1] if exceed[n][1] else None,
            "frac_top1_changed": top_change[n][0] / top_change[n][1] if top_change[n][1] else None,
        }
    # Descriptive: how often the lookahead found a strictly better program than the root.
    improved = [branch_best(s["discovery"]) > s["utility"] + 1e-9
                for r in records for s in r["siblings"] if s["category"] == "valid"]
    valid = [s["category"] == "valid" for r in records for s in r["siblings"]]
    scored = [s["eligible"] for r in records for s in r["siblings"]]
    summary["root_runnable_rate"] = statistics.fmean(valid) if valid else None
    summary["root_scored_rate"] = statistics.fmean(scored) if scored else None
    summary["lookahead_improved_rate"] = statistics.fmean(improved) if improved else None
    return summary


def late_bloomers(records, ops=("improve",)):
    """Delayed-reward cases: a sibling that is NOT the group's single-step best, whose own branch later beats
    the group's best single-step utility.

    pure: every node on the branch (root + continuations) ran and scored, i.e. Improve on top of Improve,
    not a crash repaired by Debug. confirmed: in the independent continuations the late bloomer's future value
    is higher than that of the single-step best sibling (it was not just a lucky discovery chain)."""
    def u(n):
        return n["static_value"] if n["eligible"] else 0.0
    out = {"groups": 0, "candidates": 0, "late_any": 0, "late_pure": 0, "confirmed_any": 0, "confirmed_pure": 0,
           "pure_chains": 0, "examples": []}
    for rec in records:
        if rec["op"] not in ops:
            continue
        sibs = [s for s in rec["siblings"] if s["category"] == "valid"]
        if len(sibs) < 2:
            continue
        out["groups"] += 1
        u0 = [s["utility"] if s["eligible"] else 0.0 for s in sibs]
        best0 = max(u0)
        top = u0.index(best0)
        for i, s in enumerate(sibs):
            if u0[i] >= best0 - 1e-9:
                continue
            out["candidates"] += 1
            chain = s["discovery"]
            pure = all(n["eligible"] for n in chain) and len(chain) > 1
            out["pure_chains"] += pure
            if max(u(n) for n in chain) <= best0 + 1e-9:
                continue
            ok = sibling_value(s, True) > sibling_value(sibs[top], True) + 1e-9
            out["late_any"] += 1
            out["confirmed_any"] += ok
            if pure:
                out["late_pure"] += 1
                out["confirmed_pure"] += ok
            if len(out["examples"]) < 8:
                out["examples"].append({"task": rec["task"], "parent": rec["parent_utility"],
                                        "chain": [round(u(n), 3) for n in chain], "group_single_best": best0,
                                        "pure": pure, "confirmed": ok})
    return out


def _fmt(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def _table(summary):
    lines = ["| signal | Spearman(reward, V) [95% CI] | minus single [95% CI] | pairwise acc | top-1 V | "
             "top-1 V minus single [95% CI] | reward > single | top-1 changed |",
             "|---|---|---|---|---|---|---|---|"]
    for n, s in summary["signals"].items():
        d = s["spearman_minus_single"]
        t = s["top1_minus_single"]
        lines.append(
            f"| {n} | {_fmt(s['spearman_mean'])} [{_fmt(s['spearman_ci'][0])}, {_fmt(s['spearman_ci'][1])}] | "
            f"{_fmt(d[0])} [{_fmt(d[1])}, {_fmt(d[2])}] | {_fmt(s['pair_accuracy'])} ({s['pairs']}) | "
            f"{_fmt(s['top1_value'])} | {_fmt(t[0])} [{_fmt(t[1])}, {_fmt(t[2])}] | "
            f"{_fmt(s['frac_reward_gt_single'], 2)} | {_fmt(s['frac_top1_changed'], 2)} |")
    return lines


def failure_counts(records, top=10):
    import collections
    c = collections.Counter()
    for r in records:
        for s in r["siblings"]:
            tail = (s.get("log_tail") or "").strip().splitlines()
            if tail:
                c[tail[-1][:100]] += 1
    return c.most_common(top)


def report(summary, records, out, args):
    with_root = summary["with_root"]
    summary = summary["future"]
    lines = ["# Multi-step vs single-step reward (inference only)", "",
             f"States: {summary['groups_total']} (groups with >= 2 runnable siblings: {summary['groups_used']}), "
             f"G={args.group_size}, K={args.steps}, independent repeats R={args.repeats}.",
             f"Root runnable rate {_fmt(summary['root_runnable_rate'], 2)}, scored rate "
             f"{_fmt(summary['root_scored_rate'], 2)}; lookahead found a better program than the root for "
             f"{_fmt(summary['lookahead_improved_rate'], 2)} of runnable roots.", "",
             "## Main: V = future potential (root excluded)", "",
             "V = mean over R independent K-step continuations of the best utility reached *after* the sibling. "
             "This is how good a search starting point the sibling is, which is what the multi-step reward claims "
             "to measure.", ""]
    lines += _table(summary)
    lines += ["", "## Secondary: V includes the root's own utility", "",
              "Shares the root score with the single-step reward, so it mechanically favours `single`.", ""]
    lines += _table(with_root)
    lb = summary.get("late_bloomers")
    if lb and lb["groups"]:
        c = max(1, lb["candidates"])
        lines += ["", "## Delayed reward in Improve states (late bloomers)", "",
                  f"Improve groups: {lb['groups']}; siblings that were not the single-step best: {lb['candidates']} "
                  f"(of which {lb['pure_chains']} have an all-scored continuation chain).",
                  f"- Overtook the group's single-step best later (any chain): {lb['late_any']} ({lb['late_any'] / c:.1%}); "
                  f"confirmed by independent continuations: {lb['confirmed_any']}",
                  f"- Same, pure Improve-on-Improve chains only: {lb['late_pure']} ({lb['late_pure'] / c:.1%}); "
                  f"confirmed: {lb['confirmed_pure']}", ""]
        for e in lb["examples"]:
            lines.append(f"  - {e['task']} parent={_fmt(e['parent'], 2)} chain={e['chain']} "
                         f"single-best={_fmt(e['group_single_best'], 2)} pure={e['pure']} confirmed={e['confirmed']}")
    fails = failure_counts(records)
    if fails:
        lines += ["", "## Most common failures of root programs", ""] + [f"- {n} x `{m}`" for m, n in fails]
    lines += ["", "How to read it:",
              "- A multi-step signal is a better GSPO training signal if its Spearman and pairwise accuracy "
              "against V are higher than `single`, with the 'minus single' CI above 0.",
              "- `credit_*` rewards can never be below `single` (gains are clipped at 0), so 'reward > single' "
              "only says how often lookahead found a new record; it is not evidence on its own.",
              "- The frozen model is both the operator and the continuation policy; this measures signal quality "
              "at the start of training, not the effect of training."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({"future": summary, "with_root": with_root}, indent=1))
    try:
        plot(summary, records, out, args)
    except Exception as e:  # plotting must never lose the numbers
        print(f"plot failed: {e}", flush=True)


def plot(summary, records, out, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigs = summary["signals"]
    names = list(sigs)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    y = [sigs[n]["spearman_mean"] or 0.0 for n in names]
    lo = [max(0.0, (sigs[n]["spearman_mean"] or 0) - (sigs[n]["spearman_ci"][0] or 0)) for n in names]
    hi = [max(0.0, (sigs[n]["spearman_ci"][1] or 0) - (sigs[n]["spearman_mean"] or 0)) for n in names]
    colors = ["#8a8f98" if n == "single" else "#2a7de1" for n in names]
    ax.bar(range(len(names)), y, yerr=[lo, hi], color=colors, capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spearman(reward, V) within group")
    ax.set_title("(a) GSPO signal vs independent future value")
    ax.axhline(y[0], color="#8a8f98", ls="--", lw=1)

    ax = axes[1]
    t = [sigs[n]["top1_minus_single"] for n in names[1:]]
    ax.bar(range(len(t)), [x[0] or 0 for x in t],
           yerr=[[max(0, (x[0] or 0) - (x[1] or 0)) for x in t], [max(0, (x[2] or 0) - (x[0] or 0)) for x in t]],
           color="#2a7de1", capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(names[1:], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("V(top-1 by signal) - V(top-1 by single)")
    ax.set_title("(b) Is the sibling GSPO favours a better start?")

    ax = axes[2]
    key = f"best_K{args.steps}"
    xs, ys = [], []
    for rec in records:
        for s in rec["siblings"]:
            if s["category"] != "valid":
                continue
            v = sibling_value(s, True)
            xs.append(s["r0"])
            ys.append(v)
    ax.scatter(xs, ys, s=10, alpha=0.5, color="#8a8f98", label="single-step r0")
    xs2 = [signals(s, None, args.steps, [])[key] for rec in records for s in rec["siblings"]
           if s["category"] == "valid"]
    ax.scatter(xs2, ys, s=10, alpha=0.5, color="#2a7de1", label=f"{key}")
    ax.set_xlabel("reward used by GSPO")
    ax.set_ylabel("V (best utility reached after the sibling)")
    ax.set_title("(c) Per-sibling reward vs future value")
    ax.legend(fontsize=8)
    fig.suptitle("Frozen-model inference: does lookahead reward rank siblings closer to their real search value?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "credit_signal.png", dpi=150)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Inference-only comparison of single-step vs multi-step GSPO reward")
    ap.add_argument("--tasks", default=None, help="synthetic task dir built by tasks.py (built here if missing)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--fake-policy", action="store_true", help="scripted policy, no GPU (pipeline test)")
    ap.add_argument("--num-states", type=int, default=24)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2, help="lookahead depth K")
    ap.add_argument("--repeats", type=int, default=4, help="independent continuations for ground truth V")
    ap.add_argument("--lams", default="0.5,0.9,1.0", help="lambda values for credit mode")
    ap.add_argument("--seed-drafts", type=int, default=4, help="drafts per task to seed Improve/Debug parents")
    ap.add_argument("--op-probs", default="draft=0.34,improve=0.33,debug=0.33")
    ap.add_argument("--improve-only", action="store_true",
                    help="every state is Improve on a scored parent whose utility lies in --parent-range")
    ap.add_argument("--parent-range", default="0.2,0.8")
    ap.add_argument("--seed-scripted", default="",
                    help="also seed these fixed programs as candidate parents, e.g. 'linear,constant'")
    ap.add_argument("--chunk", type=int, default=4, help="states per generation batch (resume granularity)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--mem-gb", type=float, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--pair-margin", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-only", action="store_true")
    g = ap.add_argument_group("real OpenMLE task packages (paper prompts, sandbox scoring and reward)")
    g.add_argument("--openmle-tasks", nargs="+", default=None,
                   help="task package dir(s) (data/public, data/private, utils/metric.py, info/task_metadata.json)")
    g.add_argument("--task-names", default="", help="comma list to restrict the packages")
    g.add_argument("--max-tasks", type=int, default=0)
    g.add_argument("--allow-gpu-tasks", action="store_true")
    g.add_argument("--allow-missing-bounds", action="store_true",
                   help="keep tasks without leaderboard/theoretical bounds (static utility then uses +-100)")
    g.add_argument("--model-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible server (vLLM)")
    g.add_argument("--model-name", default="Frontis-MA1-30B", help="served model name")
    g.add_argument("--concurrency", type=int, default=32, help="parallel model requests")
    g.add_argument("--judge", choices=["model", "none"], default="model",
                   help="hack check with the released judge prompt, sent to the policy server")
    g.add_argument("--apptainer-image", default=None, help="run generated programs inside this .sif")
    g.add_argument("--program-python", default=sys.executable)
    g.add_argument("--program-gpus", default="", help="CUDA_VISIBLE_DEVICES for generated programs")
    g.add_argument("--resources", default="- CPU: 6 cores\n - System Memory (RAM): 32 GB\n - GPU: none",
                   help="resource paragraph written into the public system prompt")
    g.add_argument("--keep-original-resources", action="store_true")
    g.add_argument("--job-workdir", default=None, help="scratch dir for program runs (default: $TMPDIR)")
    args = ap.parse_args(argv)
    if not args.tasks and not args.openmle_tasks:
        ap.error("give --tasks (synthetic) or --openmle-tasks (real packages)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    records = [json.loads(line) for line in rec_path.read_text().splitlines()] if rec_path.exists() else []
    if not args.report_only:
        if args.openmle_tasks:
            pass
        elif not (Path(args.tasks) / "splits.json").exists():
            import tasks as tasks_mod
            tasks_mod.main(["--out", args.tasks, "--train-per-kind", "3", "--eval-per-kind", "2"])
        (out / "config.json").write_text(json.dumps(vars(args), indent=1))
        rng = random.Random(args.seed)
        if args.openmle_tasks:
            import openmle_backend as ob
            import train as train_mod
            tasks = ob.load_openmle_tasks(
                args.openmle_tasks, [x for x in args.task_names.split(",") if x] or None,
                cpu_only=not args.allow_gpu_tasks, require_bounds=not args.allow_missing_bounds,
                resources=None if args.keep_original_resources else args.resources.replace("\\n", "\n"),
                limit=args.max_tasks)
            if not tasks:
                raise SystemExit("no usable OpenMLE task package")
            print(f"{len(tasks)} OpenMLE tasks: {', '.join(t.name for t in tasks[:20])}", flush=True)
            global extract_code
            extract_code = ob.extract
            train_mod.extract_code = ob.extract
            rng.shuffle(tasks)
            if args.fake_policy:
                policy = ob.FakeOpenMLEPolicy(args.seed)
                judge = None
            else:
                policy = ob.OpenAIPolicy(args.model_url, args.model_name, args.concurrency, args.seed)
                judge = ob.ModelJudge(args.model_url, args.model_name) if args.judge == "model" else None
            sandbox = ob.LocalSandbox(job_timeout=args.timeout, workdir=args.job_workdir, python=args.program_python,
                                      apptainer_image=args.apptainer_image, program_gpus=args.program_gpus)
            executor = ob.OpenMLEExecutor(sandbox, judge, args.workers)
        else:
            tasks = load_tasks(Path(args.tasks), args.split)
            rng.shuffle(tasks)
        if args.openmle_tasks:
            pass
        elif args.fake_policy:
            from policy import FakePolicy
            policy = FakePolicy(args.seed)
        else:
            import torch
            from policy import HFPolicy
            torch.manual_seed(args.seed)
            policy = HFPolicy(args.model, device=args.device, gen_batch=args.gen_batch, train=False,
                              grad_checkpointing=False)
        if not args.openmle_tasks:
            executor = Executor(args.workers, args.timeout, args.mem_gb)
        states = build_states(policy, executor, tasks, args, rng, out)
        with open(rec_path, "a") as f:
            for s in range(len(records), len(states), args.chunk):
                t0 = time.time()
                chunk = collect_chunk(policy, executor, states[s:s + args.chunk], args)
                for r in chunk:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                records += chunk
                print(f"[{len(records)}/{len(states)} states] executions={executor.calls} "
                      f"t={time.time() - t0:.0f}s", flush=True)
    summary = {"future": analyse(records, args, True), "with_root": analyse(records, args, False)}
    summary["future"]["late_bloomers"] = late_bloomers(records)
    report(summary, records, out, args)
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
