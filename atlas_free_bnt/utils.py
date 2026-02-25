from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for Python / NumPy / PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_txt_list(path: str | Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(str(path))


def _first_ndarray_in_mat(d: Dict[str, Any]) -> np.ndarray:
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            return v
    raise KeyError("No ndarray found in MAT file. Provide an explicit key.")


def load_mat_array(path: str | Path, key: Optional[str] = None) -> np.ndarray:
    """Load an array from a .mat file.

    Supports MATLAB v7.2 and earlier via scipy.io.loadmat, and tries mat73 for v7.3.

    Args:
        path: .mat file path.
        key: Optional variable name inside MAT.

    Returns:
        numpy array.
    """
    path = Path(path)
    # Try scipy first (v7.2 or earlier)
    try:
        from scipy.io import loadmat

        d = loadmat(str(path))
        if key is not None:
            if key not in d:
                raise KeyError(f"Key '{key}' not found in {path.name}. Keys: {list(d.keys())}")
            arr = d[key]
        else:
            arr = _first_ndarray_in_mat(d)
        return np.asarray(arr)
    except NotImplementedError:
        # likely v7.3
        pass
    except Exception as e:
        # if scipy fails for any reason, try mat73 below
        scipy_err = e
    else:
        scipy_err = None

    # Try mat73 (MATLAB v7.3)
    try:
        import mat73  # type: ignore

        d = mat73.loadmat(str(path))
        if key is not None:
            if key not in d:
                raise KeyError(f"Key '{key}' not found in {path.name}. Keys: {list(d.keys())}")
            arr = d[key]
        else:
            # mat73 returns nested python lists sometimes; find first ndarray-like
            for k, v in d.items():
                if k.startswith("__"):
                    continue
                arr = np.asarray(v)
                if arr.size > 0:
                    break
            else:
                raise KeyError("No array-like variable found in MAT file.")
        return np.asarray(arr)
    except Exception as e:
        if scipy_err is not None:
            raise RuntimeError(f"Failed to load MAT file {path} via scipy and mat73.") from e
        raise


@dataclass(frozen=True)
class BlockSpec:
    block_size: int = 3
    stride: int = 1


def generate_block_coords_from_mask(mask: np.ndarray, spec: BlockSpec, block_thd: int) -> np.ndarray:
    """Generate overlapping block start coordinates that intersect the brain mask.

    mask: 3D boolean or {0,1} array in MNI space, shape (X,Y,Z).
    Returns: int array of shape (num_blocks, 3) with (x,y,z) starts.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {mask.shape}")
    B = int(spec.block_size)
    s = int(spec.stride)
    X, Y, Z = mask.shape
    m = mask.astype(bool)

    coords: List[Tuple[int, int, int]] = []
    # Note: simple loops for clarity; you can precompute once and reuse the saved coords.
    for x in range(0, X - B + 1, s):
        xs = slice(x, x + B)
        for y in range(0, Y - B + 1, s):
            ys = slice(y, y + B)
            for z in range(0, Z - B + 1, s):
                zs = slice(z, z + B)
                if (np.sum(m[xs, ys, zs])>=block_thd):
                    coords.append((x, y, z))

    return np.asarray(coords, dtype=np.int32)


def build_block_index_matrix(C: np.ndarray, block_coords: np.ndarray, spec: BlockSpec) -> np.ndarray:
    """Build D: [num_blocks, block_size^3] cluster indices per block.

    C: 3D cluster id volume, 0 = background, positive = cluster index.
       Values are assumed to be in [0, a] where a matches the per-subject feature matrix rows.
    block_coords: [num_blocks, 3] start coordinates
    Returns:
      D: int64 array [num_blocks, B^3] with flattened C values for each block.
    """
    if C.ndim != 3:
        raise ValueError(f"C must be 3D, got shape {C.shape}")
    B = int(spec.block_size)
    D = np.empty((block_coords.shape[0], B * B * B), dtype=np.int64)
    for i, (x, y, z) in enumerate(block_coords):
        blk = C[x : x + B, y : y + B, z : z + B]
        if blk.shape != (B, B, B):
            raise ValueError(
                f"Block out of bounds at coord {(x,y,z)} with C shape {C.shape} and B={B}"
            )
        D[i] = blk.reshape(-1)
    return D


def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == y_true).mean())


class SimpleLogger:
    """Minimal logger that writes to both stdout and a file."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        ensure_dir(self.log_path.parent)

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
