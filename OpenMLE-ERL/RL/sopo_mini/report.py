"""Compare arms on the held-out search curves and state a conclusion.

Usage: python report.py --baseline control --out results/ results/eval/*.jsonl
(arm name = jsonl file stem). Differences are paired on identical (task, seed)
pairs; 95% CIs use a cluster bootstrap over tasks (seeds of one task are correlated).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def load(path):
    return {(r["task"], r["seed"]): r for r in map(json.loads, Path(path).read_text().splitlines()) if r}


def at(curve, b):
    return curve[min(b, len(curve)) - 1]


def boot_ci(per_task: dict[str, list[float]], iters=5000, seed=0):
    rng = random.Random(seed)
    tasks = list(per_task)
    means = []
    for _ in range(iters):
        sample = [x for t in (rng.choice(tasks) for _ in tasks) for x in per_task[t]]
        means.append(statistics.fmean(sample))
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def summarise(arms: dict[str, dict], baseline: str, budgets: list[int]):
    rows = {}
    for name, runs in arms.items():
        curves = [r["curve"] for r in runs.values()]
        drafts = [u for r in runs.values() for u in r["draft_utils"]]
        rows[name] = {
            **{f"best@{b}": statistics.fmean(at(c, b) for c in curves) for b in budgets},
            "draft_mean_utility": statistics.fmean(drafts) if drafts else 0.0,
            "valid_rate": statistics.fmean(v for r in runs.values() for v in r["valid"]),
            "n": len(curves),
        }
    comps = {}
    base = arms[baseline]
    final = budgets[-1]
    for name, runs in arms.items():
        if name == baseline:
            continue
        keys = sorted(set(runs) & set(base))
        per_task: dict[str, list[float]] = {}
        late: dict[str, list[float]] = {}
        for k in keys:
            d = at(runs[k]["curve"], final) - at(base[k]["curve"], final)
            per_task.setdefault(k[0], []).append(d)
            n_d = next((i for i, op in enumerate(runs[k]["ops"]) if op != "draft"), len(runs[k]["ops"])) or 1
            gain_a = at(runs[k]["curve"], final) - at(runs[k]["curve"], n_d)
            gain_b = at(base[k]["curve"], final) - at(base[k]["curve"], n_d)
            late.setdefault(k[0], []).append(gain_a - gain_b)
        diffs = [x for v in per_task.values() for x in v]
        lo, hi = boot_ci(per_task)
        llo, lhi = boot_ci(late)
        comps[name] = {
            "pairs": len(diffs), "tasks": len(per_task),
            f"delta_best@{final}": statistics.fmean(diffs), "ci95": [lo, hi],
            "wins": sum(d > 1e-9 for d in diffs), "ties": sum(abs(d) <= 1e-9 for d in diffs),
            "losses": sum(d < -1e-9 for d in diffs),
            "delta_search_gain_after_drafts": statistics.fmean(x for v in late.values() for x in v),
            "search_gain_ci95": [llo, lhi],
        }
    return rows, comps


def verdict(c, final):
    lo, hi = c["ci95"]
    if lo > 0:
        return f"better than baseline at budget {final} (95% CI excludes 0)"
    if hi < 0:
        return f"worse than baseline at budget {final} (95% CI excludes 0)"
    return "no significant difference at this scale (CI includes 0)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="1,4,8,16")
    args = ap.parse_args(argv)
    budgets = sorted({int(b) for b in args.budgets.split(",")})
    arms = {Path(f).stem: load(f) for f in args.files}
    if args.baseline not in arms:
        raise SystemExit(f"baseline {args.baseline} not in {list(arms)}")
    rows, comps = summarise(arms, args.baseline, budgets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"arms": rows, "vs_baseline": comps}, indent=1))
    final = budgets[-1]
    lines = ["# SOPO small-scale result", "",
             "Utility: 0 = constant predictor, 1 = default gradient-boosting reference (held-out tasks).", "",
             "| arm | " + " | ".join(f"best@{b}" for b in budgets) + " | draft utility | valid rate | runs |",
             "|---" * (len(budgets) + 4) + "|"]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[f'best@{b}']:.3f}" for b in budgets)
                     + f" | {r['draft_mean_utility']:.3f} | {r['valid_rate']:.2f} | {r['n']} |")
    lines += ["", f"## Paired difference vs `{args.baseline}` (same task and seed)", ""]
    for name, c in comps.items():
        lines.append(
            f"- **{name}**: Δbest@{final} = {c[f'delta_best@{final}']:+.3f} "
            f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], W/T/L {c['wins']}/{c['ties']}/{c['losses']}; "
            f"Δ(gain after drafts) = {c['delta_search_gain_after_drafts']:+.3f} "
            f"[{c['search_gain_ci95'][0]:+.3f}, {c['search_gain_ci95'][1]:+.3f}] → {verdict(c, final)}")
    lines += ["", "If SOPO only shifts the curve left (higher early, same best@final) it saves search time; "
              "a higher best@final and a larger gain after drafts means search reaches programs the baseline "
              "does not."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, runs in arms.items():
            curves = [r["curve"] for r in runs.values()]
            n = min(map(len, curves))
            ax.plot(range(1, n + 1), [statistics.fmean(c[i] for c in curves) for i in range(n)], label=name)
        ax.set_xlabel("programs executed in search")
        ax.set_ylabel("best utility so far (held-out)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "search_curves.png", dpi=150)
    except ImportError:
        pass
    print("\n".join(lines))


if __name__ == "__main__":
    main()
