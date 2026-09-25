#!/usr/bin/env python3
"""Reorder a preprocessed mmap directory to match a baseline mmap order.

The exiD/highD preprocessors collect per-recording buffers from
ProcessPoolExecutor.as_completed(), so sample order can differ across runs.
Existing split indices are index-based, therefore preprocessing-time ablations
must be aligned back to the baseline metadata order before training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.lib.format import open_memmap


META_FILES = ("meta_recordingId.npy", "meta_trackId.npy", "meta_frame.npy")
ARRAY_FILES = (
    "x_ego.npy",
    "y.npy",
    "y_vel.npy",
    "y_acc.npy",
    "x_nb.npy",
    "nb_mask.npy",
    "x_last_abs.npy",
    "meta_recordingId.npy",
    "meta_trackId.npy",
    "meta_frame.npy",
)


def _load_meta(mmap_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [name for name in META_FILES if not (mmap_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{mmap_dir} missing metadata files: {', '.join(missing)}")
    return tuple(np.load(mmap_dir / name, mmap_mode="r") for name in META_FILES)  # type: ignore[return-value]


def _keys(meta: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Iterable[Tuple[int, int, int]]:
    rec, track, frame = meta
    for r, t, f in zip(rec, track, frame):
        yield int(r), int(t), int(f)


def build_permutation(base_dir: Path, target_dir: Path) -> np.ndarray | None:
    base_meta = _load_meta(base_dir)
    target_meta = _load_meta(target_dir)

    n = len(base_meta[0])
    if any(len(arr) != n for arr in (*base_meta, *target_meta)):
        raise ValueError(
            f"metadata length mismatch: base={len(base_meta[0])}, target={len(target_meta[0])}"
        )

    if all(np.array_equal(b, t) for b, t in zip(base_meta, target_meta)):
        return None

    target_lookup: Dict[Tuple[int, int, int], int] = {}
    for i, key in enumerate(_keys(target_meta)):
        if key in target_lookup:
            raise ValueError(f"duplicate target sample key: {key}")
        target_lookup[key] = i

    perm = np.empty(n, dtype=np.int64)
    missing = []
    for i, key in enumerate(_keys(base_meta)):
        j = target_lookup.get(key)
        if j is None:
            missing.append(key)
            if len(missing) >= 5:
                break
        else:
            perm[i] = j

    if missing:
        preview = ", ".join(map(str, missing))
        raise ValueError(f"target is missing baseline sample keys, e.g. {preview}")

    return perm


def rewrite_array(path: Path, perm: np.ndarray, chunk_size: int) -> None:
    arr = np.load(path, mmap_mode="r")
    n = len(perm)
    if arr.shape[0] != n:
        print(f"[SKIP] {path.name}: first dimension {arr.shape[0]} != {n}")
        return

    tmp = path.with_name(f"{path.stem}.aligned.npy")
    tmp.unlink(missing_ok=True)
    out = open_memmap(tmp, mode="w+", dtype=arr.dtype, shape=arr.shape)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = arr[perm[start:end]]
    out.flush()
    del out
    del arr
    os.replace(tmp, path)
    print(f"[OK] aligned {path.name}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Align a variant mmap directory to a baseline mmap metadata order."
    )
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--target-dir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=32768)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    perm = build_permutation(args.base_dir, args.target_dir)
    if perm is None:
        print(f"[OK] already aligned: {args.target_dir}")
        return

    print(f"[INFO] aligning {args.target_dir} to {args.base_dir}")
    for name in ARRAY_FILES:
        path = args.target_dir / name
        if path.exists():
            rewrite_array(path, perm, args.chunk_size)
    print("[DONE] mmap order aligned")


if __name__ == "__main__":
    main()
