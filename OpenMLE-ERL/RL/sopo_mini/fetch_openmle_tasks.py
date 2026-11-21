"""Download ready-to-use OpenMLE task packages (FrontisAI/OpenMLE-Tasks, release_type=built_task).

Reads task_index.jsonl + package_manifest.jsonl (downloaded once from the dataset repo), picks
single-part packages by size / file count (small tabular tasks by default), downloads each
tasks/<task_id>/task.tar.zst through HF_ENDPOINT (e.g. https://hf-mirror.com), checks its SHA-256
and extracts it to <out>/<task_id>. Already extracted tasks are skipped, so it can be re-run.

    export HF_ENDPOINT=https://hf-mirror.com
    python fetch_openmle_tasks.py --meta $W/openmle_meta --out $W/openmle_tasks --max-tasks 40
    python fetch_openmle_tasks.py --meta ... --out ... --task-ids pokemon@1,bakery@1
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

REPO = "FrontisAI/OpenMLE-Tasks"


def _url(path: str) -> str:
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{base}/datasets/{REPO}/resolve/main/{path}"


def _download(url: str, dest: Path, sha256: str) -> None:
    tmp = dest.with_suffix(".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {url}")
    tmp.replace(dest)


def _safe_extract(archive: Path, out: Path) -> None:
    import zstandard
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    with open(archive, "rb") as fh:
        stream = zstandard.ZstdDecompressor().stream_reader(fh)
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for m in tar:
                target = (out / m.name).resolve()
                if not str(target).startswith(str(root)) or m.issym() or m.islnk():
                    raise ValueError(f"unsafe member {m.name}")
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif m.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)


def _package_root(d: Path) -> Path:
    """Archives may contain the package directly or inside one top-level folder."""
    if (d / "data" / "public").is_dir():
        return d
    subs = [p for p in d.iterdir() if p.is_dir()]
    if len(subs) == 1 and (subs[0] / "data" / "public").is_dir():
        return subs[0]
    return d


def select(meta: Path, args) -> list[dict]:
    index = {r["task_id"]: r for r in map(json.loads, open(meta / "task_index.jsonl")) if r["release_type"] == "built_task"}
    manifest = {r["task_id"]: r for r in map(json.loads, open(meta / "package_manifest.jsonl"))}
    if args.task_ids:
        ids = [t for t in args.task_ids.split(",") if t]
        missing = [t for t in ids if t not in index or t not in manifest]
        if missing:
            raise SystemExit(f"not ready-to-use packages: {missing}")
        return [manifest[t] for t in ids]
    pool = [manifest[t] for t, r in index.items() if t in manifest and len(manifest[t]["parts"]) == 1
            and args.min_bytes <= r["package_bytes"] <= args.max_bytes and r["package_file_count"] <= args.max_files]
    pool.sort(key=lambda r: r["task_id"])
    random.Random(args.seed).shuffle(pool)
    return pool[:args.max_tasks] if args.max_tasks else pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta", required=True, help="dir with task_index.jsonl and package_manifest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-ids", default="")
    ap.add_argument("--max-tasks", type=int, default=40)
    ap.add_argument("--min-bytes", type=float, default=200e3, help="uncompressed package size lower bound")
    ap.add_argument("--max-bytes", type=float, default=30e6)
    ap.add_argument("--max-files", type=int, default=20, help="few files = tabular csv tasks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picked = select(Path(args.meta), args)
    print(f"{len(picked)} packages selected", flush=True)
    ok = 0
    for i, rec in enumerate(picked):
        tid = rec["task_id"]
        dest = out / tid
        if (dest / "data" / "public").is_dir():
            ok += 1
            continue
        part = rec["parts"][0]
        arc = out / f".{tid}.tar.zst"
        try:
            _download(_url(f"{rec['artifact_path']}/{part['path']}"), arc, part["sha256"])
            tmp = out / f".{tid}.extract"
            shutil.rmtree(tmp, ignore_errors=True)
            _safe_extract(arc, tmp)
            _package_root(tmp).rename(dest)
            shutil.rmtree(tmp, ignore_errors=True)
            ok += 1
            print(f"[{i + 1}/{len(picked)}] {tid}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(picked)}] {tid} FAILED: {e}", flush=True)
        finally:
            arc.unlink(missing_ok=True)
    print(f"{ok}/{len(picked)} packages ready in {out}")


if __name__ == "__main__":
    main()
