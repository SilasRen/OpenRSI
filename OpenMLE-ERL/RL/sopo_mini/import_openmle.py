"""Convert OpenMLE task packages (the paper's real Kaggle tasks) into the sopo_mini task format.

An OpenMLE package looks like

    <pkg>/data/public/{train.csv,test.csv,description.txt}
    <pkg>/data/private/test_answer.csv
    <pkg>/utils/metric.py

This script reads the id column, target column and metric from utils/metric.py, keeps every
original column (including the original id, which can carry information such as the group in
Spaceship Titanic), adds an integer ``id`` column that the local executor scores against, and
writes test labels plus dumb/reference scores to ``_private``. The original competition
description is shown to the model, followed by the submission rules of this environment.

    python import_openmle.py --packages .../task_package --out tasks_openmle
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tasks import METRICS, _describe, _reference_scores

# metric.py docstring keyword -> sopo_mini metric name
_METRIC_WORDS = [("rmsle", "rmsle"), ("root mean squared log", "rmsle"), ("rmse", "rmse"),
                 ("auc", "roc_auc"), ("f1", "macro_f1"), ("accuracy", "accuracy")]


def parse_metric_file(path: Path) -> dict:
    src = path.read_text()
    target = re.search(r'value:\s*str\s*=\s*"([^"]+)"', src)
    ident = re.search(r'id_column\s*=\s*"([^"]+)"', src)
    doc = " ".join(re.findall(r'"""(.*?)"""', src, flags=re.S)).lower()
    metric = next((m for w, m in _METRIC_WORDS if w in doc), None)
    if not (target and metric):
        raise ValueError(f"cannot read target/metric from {path}; add it to --override")
    out = dict(target=target.group(1), metric=metric)
    if ident:
        out["id_col"] = ident.group(1)
    return out


def _as_label(s: pd.Series) -> pd.Series:
    if s.dtype == bool or s.astype(str).str.lower().isin(["true", "false"]).all():
        return s.astype(str).str.lower().map({"true": 1, "false": 0}).astype(int)
    return s


def convert(pkg: Path, root: Path, override: dict | None = None, cap: float = 3.0) -> str:
    name = pkg.name
    info = parse_metric_file(pkg / "utils" / "metric.py")
    info.update(override or {})
    pub = pkg / "data" / "public"
    train = pd.read_csv(pub / "train.csv")
    test = pd.read_csv(pub / "test.csv")
    ans = pd.read_csv(pkg / "data" / "private" / "test_answer.csv")
    if "id_col" not in info:  # the answer file is <id>,<target>
        info["id_col"] = next(c for c in ans.columns if c != info["target"])
    target, id_col, metric = info["target"], info["id_col"], info["metric"]
    ans = test[[id_col]].merge(ans[[id_col, target]], on=id_col, how="left")
    if ans[target].isna().any():
        raise ValueError(f"{name}: test_answer.csv does not cover every test row")
    y_tr = _as_label(train.pop(target))
    y_te = _as_label(ans[target]).to_numpy()
    if metric not in ("rmse", "rmsle") and pd.api.types.is_float_dtype(y_tr):
        y_tr = y_tr.astype(int)
        y_te = y_te.astype(int)

    feats_tr, feats_te = train.copy(), test.copy()
    for df in (feats_tr, feats_te):  # high-cardinality text would make the reference model slow
        for c in df.columns:
            if df[c].dtype == object and df[c].nunique() > 200:
                df.drop(columns=c, inplace=True)
    dumb, ref = _reference_scores(feats_tr, feats_te[feats_tr.columns], y_tr.to_numpy(), y_te, metric)

    train.insert(0, "id", np.arange(len(train)))
    test.insert(0, "id", np.arange(len(test)))
    train[target] = y_tr.to_numpy()
    out = root / name / "public"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    task_type = "regression" if metric in ("rmse", "rmsle") else "classification"
    rules, data = _describe(name, metric, train, target, len(test), task_type)
    original = (pub / "description.txt").read_text() if (pub / "description.txt").exists() else ""
    if (pub / "data_description.txt").exists():
        data += "\n\nColumn notes:\n" + (pub / "data_description.txt").read_text()[:4000]
    if set(y_tr.unique()) <= {0, 1} and metric == "accuracy":
        rules += f" Write `{target}` as 0/1 integers."
    desc = (f"{original[:6000]}\n\n=== Rules of this environment (these override the text above) ===\n{rules}\n"
            f"The original id column `{id_col}` is kept as an ordinary column; the submission must use the "
            f"integer `id` column instead.")
    (root / name / "task.json").write_text(json.dumps(dict(
        name=name, metric=metric, target=target, description=desc, data_description=data), indent=1))
    priv = root / "_private"
    priv.mkdir(exist_ok=True)
    (priv / f"{name}.json").write_text(json.dumps(dict(
        metric=metric, y_test=np.asarray(y_te).tolist(), dumb_score=dumb, reference_score=ref,
        cap=cap)))
    better = "higher" if METRICS[metric] else "lower"
    print(f"{name}: metric={metric} ({better} better) dumb={dumb:.4f} reference(HGB)={ref:.4f} "
          f"train={len(train)} test={len(test)}")
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packages", required=True, nargs="+",
                    help="task package dirs, or a parent dir that contains several")
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", default="{}",
                    help='JSON {pkg: {"target":..,"id_col":..,"metric":..}} for packages that cannot be parsed')
    ap.add_argument("--cap", type=float, default=3.0,
                    help="utility cap; real tasks leave room above the default-GBM reference (=1)")
    args = ap.parse_args(argv)
    override = json.loads(args.override)
    pkgs = []
    for p in map(Path, args.packages):
        pkgs += [p] if (p / "data" / "public").exists() else sorted(q for q in p.iterdir() if (q / "data" / "public").exists())
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for pkg in pkgs:
        try:
            names.append(convert(pkg, root, override.get(pkg.name), args.cap))
        except Exception as e:  # noqa: BLE001
            print(f"skip {pkg.name}: {e}")
    (root / "splits.json").write_text(json.dumps({"train": names, "eval": names}, indent=1))
    print(f"imported {len(names)} tasks into {root}")


if __name__ == "__main__":
    main()
