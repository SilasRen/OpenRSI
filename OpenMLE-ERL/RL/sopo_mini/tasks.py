"""Offline tabular ML task pool for the small-scale SOPO experiment.

Every task is built from scikit-learn generators or bundled datasets, so nothing is
downloaded. Each task directory contains only what generated code may read:

    <task>/public/train.csv   features + target
    <task>/public/test.csv    features + id (no target)
    <task>/task.json          description shown to the model

Test labels and the score normalisers live in <root>/_private/<task>.json, which is
never copied into the execution directory.

Utility (the reward scale) is

    utility = clip((score - dumb) / (reference - dumb), 0, 1.5)

for higher-is-better metrics (sign flipped for RMSE), where ``dumb`` is a constant
predictor and ``reference`` is a default HistGradientBoosting model trained on the
same split. 0 = no better than a constant, 1 = matches the reference, >1 = beats it.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {"accuracy": True, "roc_auc": True, "macro_f1": True, "rmse": False}


@dataclass
class Task:
    name: str
    root: Path
    metric: str
    target: str
    description: str
    data_description: str

    @property
    def public_dir(self) -> Path:
        return self.root / self.name / "public"

    def private(self) -> dict:
        return json.loads((self.root / "_private" / f"{self.name}.json").read_text())


# ---------------------------------------------------------------- metrics

def compute_metric(metric: str, y_true, pred) -> float:
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score

    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    if metric == "accuracy":
        return float(accuracy_score(y_true, pred.astype(y_true.dtype, copy=False)))
    if metric == "macro_f1":
        return float(f1_score(y_true, pred.astype(y_true.dtype, copy=False), average="macro"))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, pred.astype(float)))
    if metric == "rmse":
        return float(math.sqrt(mean_squared_error(y_true, pred.astype(float))))
    raise ValueError(metric)


def to_utility(score: float, private: dict) -> float:
    dumb, ref = private["dumb_score"], private["reference_score"]
    if not METRICS[private["metric"]]:
        score, dumb, ref = -score, -dumb, -ref
    span = ref - dumb
    if not math.isfinite(score) or abs(span) < 1e-12:
        return 0.0
    return float(min(1.5, max(0.0, (score - dumb) / span)))


# ---------------------------------------------------------------- generators

def _messy(df: pd.DataFrame, rng: np.random.Generator, cat_cols: int, missing: float) -> pd.DataFrame:
    """Add categorical columns and missing values so the pipeline itself matters."""
    df = df.copy()
    num_cols = [c for c in df.columns]
    for j in range(cat_cols):
        src = df[num_cols[j % len(num_cols)]]
        bins = rng.integers(3, 8)
        labels = [f"c{k}" for k in rng.permutation(bins)]
        df[f"cat_{j}"] = pd.qcut(src.rank(method="first"), bins, labels=labels).astype(str)
        if rng.random() < 0.5:
            df = df.drop(columns=[num_cols[j % len(num_cols)]])
    if missing > 0:
        for c in df.columns:
            if rng.random() < 0.5:
                mask = rng.random(len(df)) < missing
                df.loc[mask, c] = np.nan
    return df


def _synthetic(kind: str, seed: int):
    from sklearn import datasets

    rng = np.random.default_rng(seed)
    n = int(rng.integers(600, 3000))
    if kind == "clf_linear":
        X, y = datasets.make_classification(n_samples=n, n_features=int(rng.integers(8, 30)),
                                            n_informative=int(rng.integers(3, 8)), flip_y=float(rng.uniform(0, .1)),
                                            class_sep=float(rng.uniform(.5, 1.5)), random_state=seed)
        metric = "roc_auc"
    elif kind == "clf_multiclass":
        k = int(rng.integers(3, 6))
        X, y = datasets.make_classification(n_samples=n, n_features=int(rng.integers(10, 30)), n_classes=k,
                                            n_informative=int(rng.integers(4, 9)), n_clusters_per_class=1,
                                            flip_y=float(rng.uniform(0, .08)), random_state=seed)
        metric = "macro_f1"
    elif kind == "clf_nonlinear":
        X, y = datasets.make_moons(n_samples=n, noise=float(rng.uniform(.1, .35)), random_state=seed)
        X = np.hstack([X, rng.normal(size=(n, int(rng.integers(4, 15))))])
        metric = "accuracy"
    elif kind == "clf_interaction":
        X = rng.normal(size=(n, int(rng.integers(6, 20))))
        logits = 2 * X[:, 0] * X[:, 1] - 1.5 * np.sin(2 * X[:, 2]) + X[:, 3] ** 2 - 1
        y = (logits + rng.normal(scale=.5, size=n) > 0).astype(int)
        metric = "roc_auc"
    elif kind == "reg_friedman":
        f = [datasets.make_friedman1, datasets.make_friedman2, datasets.make_friedman3][seed % 3]
        X, y = f(n_samples=n, noise=float(rng.uniform(.1, 1.0)), random_state=seed)
        if X.shape[1] < 6:
            X = np.hstack([X, rng.normal(size=(n, 6))])
        metric = "rmse"
    elif kind == "reg_linear":
        X, y = datasets.make_regression(n_samples=n, n_features=int(rng.integers(8, 40)),
                                        n_informative=int(rng.integers(3, 10)), noise=float(rng.uniform(5, 40)),
                                        random_state=seed)
        y = y + 0.02 * y ** 2 * rng.choice([0, 1])
        metric = "rmse"
    else:
        raise ValueError(kind)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df = _messy(df, rng, cat_cols=int(rng.integers(0, 3)), missing=float(rng.choice([0, 0, .05, .1])))
    return df, pd.Series(y), metric


def _bundled(name: str, seed: int):
    from sklearn import datasets

    rng = np.random.default_rng(seed)
    loader = {"breast_cancer": (datasets.load_breast_cancer, "roc_auc"),
              "wine": (datasets.load_wine, "macro_f1"),
              "digits": (datasets.load_digits, "accuracy"),
              "diabetes": (datasets.load_diabetes, "rmse"),
              "iris": (datasets.load_iris, "accuracy")}[name]
    d = loader[0]()
    df = pd.DataFrame(d.data, columns=[f"x{i}" for i in range(d.data.shape[1])])
    df = _messy(df, rng, cat_cols=int(rng.integers(0, 2)), missing=float(rng.choice([0, .05])))
    return df, pd.Series(d.target), loader[1]


TRAIN_KINDS = ["clf_linear", "clf_multiclass", "clf_nonlinear", "reg_friedman", "reg_linear"]
# Held-out evaluation adds a family never seen in training to measure transfer.
EVAL_KINDS = TRAIN_KINDS + ["clf_interaction"]
BUNDLED = ["breast_cancer", "wine", "digits", "diabetes", "iris"]


def _reference_scores(train: pd.DataFrame, test: pd.DataFrame, y_tr, y_te, metric: str):
    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder

    enc = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), make_column_selector(dtype_exclude="number")),
        remainder="passthrough")
    if metric == "rmse":
        model = make_pipeline(enc, HistGradientBoostingRegressor(random_state=0))
        model.fit(train, y_tr)
        ref = compute_metric(metric, y_te, model.predict(test))
        dumb = compute_metric(metric, y_te, np.full(len(y_te), float(np.mean(y_tr))))
    else:
        model = make_pipeline(enc, HistGradientBoostingClassifier(random_state=0))
        model.fit(train, y_tr)
        if metric == "roc_auc":
            ref = compute_metric(metric, y_te, model.predict_proba(test)[:, 1])
            dumb = 0.5
        else:
            ref = compute_metric(metric, y_te, model.predict(test))
            majority = pd.Series(y_tr).mode().iloc[0]
            dumb = compute_metric(metric, y_te, np.full(len(y_te), majority))
    return float(dumb), float(ref)


def _describe(name: str, metric: str, train: pd.DataFrame, target: str, n_test: int, task_type: str):
    pred_desc = {
        "roc_auc": f"the probability that `{target}` is 1",
        "accuracy": f"the predicted class label of `{target}`",
        "macro_f1": f"the predicted class label of `{target}`",
        "rmse": f"the predicted numeric value of `{target}`",
    }[metric]
    desc = (f"Tabular {task_type} task `{name}`. Predict `{target}` for every row of test.csv.\n"
            f"Evaluation metric: {metric} ({'higher' if METRICS[metric] else 'lower'} is better).\n"
            f"Write `submission.csv` in the current working directory with columns `id,{target}`, one row per "
            f"test row, where `{target}` is {pred_desc}.")
    feats = [c for c in train.columns if c not in ("id", target)]
    dtypes = ", ".join(f"{c}:{'numeric' if pd.api.types.is_numeric_dtype(train[c]) else 'categorical'}" for c in feats[:40])
    missing = int(train[feats].isna().any(axis=0).sum())
    data = (f"Files in the current working directory: train.csv ({len(train)} rows, columns id, {len(feats)} "
            f"features and the target `{target}`) and test.csv ({n_test} rows, same columns without `{target}`).\n"
            f"Features: {dtypes}{' ...' if len(feats) > 40 else ''}.\n"
            f"{missing} feature columns contain missing values. Only pandas, numpy and scikit-learn are installed; "
            f"there is no GPU and no internet. The program must finish within the time limit.")
    return desc, data


def build_task(root: Path, name: str, df: pd.DataFrame, y: pd.Series, metric: str, seed: int) -> bool:
    from sklearn.model_selection import train_test_split

    target = "target"
    strat = y if metric != "rmse" else None
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed, stratify=strat)
    train = df.iloc[tr_idx].reset_index(drop=True)
    test = df.iloc[te_idx].reset_index(drop=True)
    y_tr, y_te = y.iloc[tr_idx].to_numpy(), y.iloc[te_idx].to_numpy()
    dumb, ref = _reference_scores(train, test, y_tr, y_te, metric)
    gain = (ref - dumb) if METRICS[metric] else (dumb - ref)
    if gain <= 0.05 * max(abs(dumb), 1e-9) and gain <= 0.02:
        return False  # reference barely beats a constant: utility would be noise
    train.insert(0, "id", np.arange(len(train)))
    test.insert(0, "id", np.arange(len(test)))
    train[target] = y_tr
    pub = root / name / "public"
    pub.mkdir(parents=True, exist_ok=True)
    train.to_csv(pub / "train.csv", index=False)
    test.to_csv(pub / "test.csv", index=False)
    task_type = "regression" if metric == "rmse" else "classification"
    desc, data = _describe(name, metric, train, target, len(test), task_type)
    (root / name / "task.json").write_text(json.dumps(dict(
        name=name, metric=metric, target=target, description=desc, data_description=data), indent=1))
    priv = root / "_private"
    priv.mkdir(exist_ok=True)
    (priv / f"{name}.json").write_text(json.dumps(dict(
        metric=metric, y_test=y_te.tolist(), dumb_score=dumb, reference_score=ref)))
    return True


def _build_synthetic(root: Path, name: str, kind: str, seed: int) -> None:
    for attempt in range(20):  # resample degenerate tasks deterministically
        if build_task(root, name, *_synthetic(kind, seed + 7919 * attempt), seed):
            return
    raise RuntimeError(f"could not build a non-degenerate {kind} task")


def load_tasks(root: Path, split: str) -> list[Task]:
    names = json.loads((Path(root) / "splits.json").read_text())[split]
    out = []
    for name in names:
        meta = json.loads((Path(root) / name / "task.json").read_text())
        out.append(Task(name=name, root=Path(root), metric=meta["metric"], target=meta["target"],
                        description=meta["description"], data_description=meta["data_description"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-kind", type=int, default=10)
    ap.add_argument("--eval-per-kind", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "eval": []}
    for i, kind in enumerate(TRAIN_KINDS):
        for j in range(args.train_per_kind):
            seed = args.seed * 100000 + i * 1000 + j
            name = f"train_{kind}_{j}"
            _build_synthetic(root, name, kind, seed)
            splits["train"].append(name)
    for i, kind in enumerate(EVAL_KINDS):
        for j in range(args.eval_per_kind):
            seed = args.seed * 100000 + 50000 + i * 1000 + j  # disjoint seeds from training
            name = f"eval_{kind}_{j}"
            _build_synthetic(root, name, kind, seed)
            splits["eval"].append(name)
    for j, b in enumerate(BUNDLED):
        name = f"eval_real_{b}"
        if build_task(root, name, *_bundled(b, args.seed + j), args.seed + j):
            splits["eval"].append(name)
    (root / "splits.json").write_text(json.dumps(splits, indent=1))
    print(f"built {len(splits['train'])} train and {len(splits['eval'])} eval tasks in {root}")


if __name__ == "__main__":
    main()
