from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import BlockSpec, build_block_index_matrix, load_mat_array, load_txt_list


@dataclass
class DatasetConfig:
    # MAT keys (optional). If not provided, first ndarray-like variable is used.
    features_key: Optional[str] = None
    clusters_key: Optional[str] = None

    # Block spec
    # Use default_factory to avoid sharing the same instance across configs.
    block_spec: BlockSpec = field(default_factory=lambda: BlockSpec(block_size=3, stride=1))

    # Performance options
    cache_block_index: bool = True  # cache D per subject in RAM (useful if epochs>1)

    # Label dtype
    # - 'int64'  : classification labels
    # - 'float32': regression targets
    label_dtype: str = "int64"


class AtlasFreeDataset(Dataset):
    """Dataset for atlas-free BNT.

    Each subject has:
      - G: [a, b] features matrix (clusters x reduced-dim features), where a varies per subject.
      - C: [X, Y, Z] cluster index volume (0 background, 1..a cluster id).

    We build D (block index matrix) on the fly from C and shared block coordinates.

    Returns a tuple:
      (features, D, label/target)
      - features: float32 [a+1, d] with row 0 = zeros (background).
      - D: int64 [num_blocks, block_size^3]
      - label/target: scalar (dtype controlled by DatasetConfig.label_dtype)
    """

    def __init__(
        self,
        features_list_path: str | Path,
        clusters_list_path: str | Path,
        labels: np.ndarray,
        block_coords: np.ndarray,
        cfg: DatasetConfig,
    ):
        self.feature_paths = load_txt_list(features_list_path)
        self.cluster_paths = load_txt_list(clusters_list_path)
        if len(self.feature_paths) != len(self.cluster_paths):
            raise ValueError("features_list and clusters_list must have same length")
        if len(labels) != len(self.feature_paths):
            raise ValueError("labels length must match number of subjects")
        if cfg.label_dtype not in {"int64", "float32"}:
            raise ValueError(f"Unknown label_dtype: {cfg.label_dtype}. Use 'int64' or 'float32'.")
        self.labels = labels.astype(np.int64) if cfg.label_dtype == "int64" else labels.astype(np.float32)
        self.block_coords = block_coords.astype(np.int32)
        self.cfg = cfg

        self._D_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.feature_paths)

    def _load_features(self, idx: int) -> np.ndarray:
        G = load_mat_array(self.feature_paths[idx], key=self.cfg.features_key)
        if G.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {G.shape} at {self.feature_paths[idx]}")
        G = np.asarray(G, dtype=np.float32)
        # Add background row 0
        out = np.zeros((G.shape[0] + 1, G.shape[1]), dtype=np.float32)
        out[1 :, :] = G
        return out

    def _load_block_index(self, idx: int) -> np.ndarray:
        if self.cfg.cache_block_index and idx in self._D_cache:
            return self._D_cache[idx]

        C = load_mat_array(self.cluster_paths[idx], key=self.cfg.clusters_key)
        C = np.asarray(C, dtype=np.int64)
        if C.ndim != 3:
            raise ValueError(f"Expected 3D cluster volume, got shape {C.shape} at {self.cluster_paths[idx]}")
        D = build_block_index_matrix(C, self.block_coords, self.cfg.block_spec)

        if self.cfg.cache_block_index:
            self._D_cache[idx] = D
        return D

    def __getitem__(self, idx: int):
        feats = self._load_features(idx)  # [a+1, b]
        D = self._load_block_index(idx)   # [num_blocks, 27]
        y = self.labels[idx]
        return feats, D, y


def collate_variable_clusters(batch):
    """Pad per-subject feature tables to the max #clusters in the batch.

    batch: list of (feats [a+1, init_dim], D [U,27], y)
    Returns:
      feats_padded: [B, max_a+1, init_dim]
      D_tensor: [B, U, 27]
      y_tensor: [B]
    """
    feats_list, D_list, y_list = zip(*batch)
    max_rows = max(f.shape[0] for f in feats_list)
    init_dim = feats_list[0].shape[1]

    B = len(batch)
    feats_padded = np.zeros((B, max_rows, init_dim), dtype=np.float32)
    for i, f in enumerate(feats_list):
        feats_padded[i, : f.shape[0], :] = f

    # D has fixed shape across subjects if block_coords are shared
    D_arr = np.stack(D_list, axis=0).astype(np.int64)
    # Keep dtype consistent with task
    y_arr = np.asarray(y_list)
    if np.issubdtype(y_arr.dtype, np.floating):
        y_arr = y_arr.astype(np.float32)
    else:
        y_arr = y_arr.astype(np.int64)

    return (
        torch.from_numpy(feats_padded),
        torch.from_numpy(D_arr),
        torch.from_numpy(y_arr),
    )


def make_dataloaders(
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from torch.utils.data import Subset

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_variable_clusters,
        drop_last=False,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_variable_clusters,
        drop_last=False,
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_variable_clusters,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
