#!/usr/bin/env python3
"""Train atlas-free BNT for classification or regression.

This script is designed for public release:
- Cross-validation is done inside the script (StratifiedKFold), controlled by --fold.
- Per-subject 3D cluster index volumes C are used to construct block index matrices D on the fly.
- Clean logging, reproducibility, best-checkpoint saving, and optional prediction dumping.

Expected conventions:
- Cluster volume C stores indices in [0, a], with 0 = background.
- Feature matrix G is [a, b], we need to add an all-zero feature vector at the beginning of G, which corresponds to the background.
- DEC(Z) returns [batch, dec_clusters, hidden_dim].

Tasks:
  - Sex classification:        --task cls  (labels must be int)
  - Brain-age regression:      --task reg  (labels must be float)

Usage:
  python train.py --task cls --features-list ... --clusters-list ... --labels-npy ... --mask-npy ... --n-splits 5 --fold 0
  python train.py --task reg --features-list ... --clusters-list ... --labels-npy ... --mask-npy ... --n-splits 5 --fold 0

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import StepLR

from atlas_free_bnt.af_bnt import AtlasFreeBNT, ModelConfig
from atlas_free_bnt.dataset import AtlasFreeDataset, DatasetConfig, make_dataloaders
from atlas_free_bnt.utils import (
    BlockSpec,
    SimpleLogger,
    accuracy_from_logits,
    ensure_dir,
    generate_block_coords_from_mask,
    load_npy,
    load_txt_list,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Task
    p.add_argument("--task", type=str, default="cls", choices=["cls", "reg"],
                   help="cls: classification (sex); reg: regression (brain age)")
    p.add_argument("--loss", type=str, default=None, choices=[None, "ce", "mse", "smoothl1"],
                   help="Override loss. Defaults: ce for cls, smoothl1 for reg")
    p.add_argument("--reg-bins", type=int, default=0,
                   help="If >0 for regression, quantile-bin targets and use StratifiedKFold.")
    p.add_argument("--standardize-target", action="store_true",
                   help="(reg only) Standardize targets using train-split mean/std (leakage-safe).")
    # Data
    p.add_argument("--features-list", type=str, required=True,
                   help="Text file: one MAT path per line for per-subject feature matrices G.")
    p.add_argument("--clusters-list", type=str, required=True,
                   help="Text file: one MAT path per line for per-subject 3D cluster volumes C.")
    p.add_argument("--labels-npy", type=str, required=True, help=".npy labels array (N,).")

    # Block coordinates
    p.add_argument("--mask-npy", type=str, default=None,
                   help="Optional brain mask .npy (X,Y,Z). Used to generate block coordinates.")
    p.add_argument("--block-coords", type=str, default=None,
                   help="Optional precomputed block_coords.npy (num_blocks,3). Overrides --mask-npy.")
    p.add_argument("--block-size", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--block-thd", type=int, default=4,
                   help="the minimum number of valid voxels in every block.")

    # Feature slicing
    p.add_argument("--init-dim", type=int, default=1080,
                   help="the feature dimensionality.")

    # MAT keys (optional)
    p.add_argument("--features-key", type=str, default=None)
    p.add_argument("--clusters-key", type=str, default=None)

    # CV
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)

    # Model
    p.add_argument("--hidden-dim", type=int, default=500)
    p.add_argument("--num-heads", type=int, default=10)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dec-clusters", type=int, default=25)
    p.add_argument("--dec-encoder-hidden-size", type=int, default=32)
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "masked_mean"],
                   help="Block pooling mode: mean includes background id=0; masked_mean excludes it.")

    p.add_argument("--num-classes", type=int, default=None,
                   help="Override output dim. Default: 2 for cls, 1 for reg.")

    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0)
    p.add_argument("--lr-step-size", type=int, default=20,
                   help="learning rate scheduler step size, reduce the learning rate every couple of epoches")
    p.add_argument("--lr-gamma", type=float, default=0.5,
                   help="learning rate reduction rate")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-dp", action="store_true",
                   help="Use torch.nn.DataParallel if multiple GPUs are available.")

    # Output
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--save-preds", action="store_true",
                   help="Save per-subject logits/preds for val/test at the best checkpoint.")

    return p.parse_args()


def build_splits(
    y: np.ndarray,
    n_splits: int,
    fold: int,
    val_ratio: float,
    seed: int,
    task: str,
    reg_bins: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/val/test indices for a given fold.

    - cls: StratifiedKFold on class labels.
    - reg: KFold by default; optionally StratifiedKFold on quantile bins if reg_bins>0.
    """
    if task == "cls":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros_like(y), y))
        strat = y
    else:
        if reg_bins and reg_bins > 1:
            # Quantile bins for stable distribution across folds
            qs = np.linspace(0, 1, reg_bins + 1)
            edges = np.quantile(y, qs)
            edges[0] -= 1e-6
            edges[-1] += 1e-6
            y_bins = np.digitize(y, edges[1:-1], right=False)
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = list(splitter.split(np.zeros_like(y_bins), y_bins))
            strat = y_bins
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = list(splitter.split(np.zeros_like(y)))
            strat = None

    if fold < 0 or fold >= len(folds):
        raise ValueError(f"fold must be in [0,{len(folds)-1}], got {fold}")
    train_val_idx, test_idx = folds[fold]

    # Split train/val from the train_val pool
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=(strat[train_val_idx] if strat is not None else None),
    )

    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    task: str,
    standardize_target: bool = False,
    y_mean: float | None = None,
    y_std: float | None = None,
) -> Dict[str, np.ndarray | float]:
    """Evaluate model on a loader.

    - cls: returns acc/logits/pred.
    - reg: returns mae/rmse/r2/pred.
    """
    model.eval()
    all_out = []
    all_y = []
    for feats, D, y in loader:
        feats = feats.to(device, non_blocking=True)
        D = D.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(feats, D)
        all_out.append(out.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    out_np = np.concatenate(all_out, axis=0)
    y_np = np.concatenate(all_y, axis=0)

    if task == "cls":
        acc = accuracy_from_logits(out_np, y_np)
        return {"acc": acc, "logits": out_np, "y": y_np, "pred": np.argmax(out_np, axis=1)}

    # regression
    pred = out_np.squeeze(-1)
    if standardize_target:
        if y_mean is None or y_std is None:
            raise ValueError("standardize_target=True requires y_mean and y_std")
        pred = pred * y_std + y_mean

    mae = float(mean_absolute_error(y_np, pred))
    rmse = float(np.sqrt(mean_squared_error(y_np, pred)))
    r2 = float(r2_score(y_np, pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "pred": pred, "y": y_np, "raw_out": out_np}


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    logger = SimpleLogger(outdir / "run.log")

    set_seed(args.seed)
    logger.log(f"Args: {vars(args)}")

    # Load labels/targets
    y_raw = load_npy(args.labels_npy)
    if args.task == "cls":
        y = np.asarray(y_raw).astype(np.int64)
        # Common convention: {1,2} -> {0,1}
        if y.min() == 1 and y.max() == 2:
            y = y - 1
        if y.min() < 0:
            raise ValueError(f"Classification labels must be non-negative, got min={y.min()}")
    else:
        y = np.asarray(y_raw).astype(np.float32)
    N = int(len(y))

    # Basic input sanity check
    feats_paths = load_txt_list(args.features_list)
    clus_paths = load_txt_list(args.clusters_list)
    if len(feats_paths) != N or len(clus_paths) != N:
        raise ValueError(
            f"Mismatch: labels N={N}, features-list={len(feats_paths)}, clusters-list={len(clus_paths)}"
        )

    # Prepare block coordinates
    spec = BlockSpec(block_size=args.block_size, stride=args.stride)
    if args.block_coords is not None:
        block_coords = load_npy(args.block_coords).astype(np.int32)
        logger.log(f"Loaded block coords from {args.block_coords}: {block_coords.shape}")
    else:
        if args.mask_npy is None:
            raise ValueError("Provide either --block-coords or --mask-npy to generate block coordinates.")
        mask = load_npy(args.mask_npy)
        block_coords = generate_block_coords_from_mask(mask, spec, block_thd=args.block_thd)
        np.save(outdir / "block_coords.npy", block_coords)
        logger.log(f"Generated block coords from mask: {block_coords.shape}. Saved to {outdir/'block_coords.npy'}")

    # CV splits
    train_idx, val_idx, test_idx = build_splits(
        y,
        n_splits=args.n_splits,
        fold=args.fold,
        val_ratio=args.val_ratio,
        seed=args.seed,
        task=args.task,
        reg_bins=args.reg_bins,
    )
    logger.log(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Dataset
    dcfg = DatasetConfig(
        features_key=args.features_key,
        clusters_key=args.clusters_key,
        block_spec=spec,
        cache_block_index=True,
        label_dtype=("int64" if args.task == "cls" else "float32"),
    )
    dataset = AtlasFreeDataset(
        features_list_path=args.features_list,
        clusters_list_path=args.clusters_list,
        labels=y,
        block_coords=block_coords,
        cfg=dcfg,
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # (reg only) target standardization stats (fit on train split only)
    y_mean: float | None = None
    y_std: float | None = None
    if args.task == "reg" and args.standardize_target:
        y_mean = float(np.mean(y[train_idx]))
        y_std = float(np.std(y[train_idx]))
        if not np.isfinite(y_std) or y_std < 1e-8:
            y_std = 1.0
        logger.log(f"Standardize targets: mean={y_mean:.6f}, std={y_std:.6f} (train split)")

    # Model
    out_dim = args.num_classes
    if out_dim is None:
        out_dim = 2 if args.task == "cls" else 1
    mcfg = ModelConfig(
        init_dim=args.init_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_blocks=int(block_coords.shape[0]),
        block_size=args.block_size,
        pooling=args.pooling,
        dec_clusters=args.dec_clusters,
        dec_encoder_hidden_size=args.dec_encoder_hidden_size,
        num_classes=int(out_dim),
    )
    model = AtlasFreeBNT(mcfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.use_dp and torch.cuda.device_count() > 1:
        logger.log(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    def _unwrap(m: nn.Module) -> nn.Module:
        return m.module if hasattr(m, "module") else m

    # Loss
    if args.loss is None:
        args.loss = "ce" if args.task == "cls" else "smoothl1"
    if args.task == "cls":
        if args.loss != "ce":
            raise ValueError("For --task cls, use --loss ce")
        criterion: nn.Module = nn.CrossEntropyLoss(reduction="mean")
    else:
        if args.loss == "mse":
            criterion = nn.MSELoss(reduction="mean")
        else:
            # smooth L1 is a strong default for age regression
            criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    if args.task == "cls":
        best_val_metric = -1.0
        metric_name = "acc"
        better = lambda new, best: new > best
    else:
        best_val_metric = 1e9
        metric_name = "mae"
        better = lambda new, best: new < best
    best_path = outdir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0

        for feats, D, labels in train_loader:
            feats = feats.to(device, non_blocking=True)
            D = D.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(feats, D)
            if args.task == "cls":
                loss = criterion(out, labels.long())
            else:
                pred = out.squeeze(-1)
                target = labels.float()
                if args.standardize_target:
                    target = (target - float(y_mean)) / float(y_std)
                loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            bs = int(labels.shape[0])
            # Keep train_loss comparable across tasks: average per sample
            if args.task == "reg":
                running_loss += float(loss.item()) * bs       # loss is sum over batch
            else:
                running_loss += float(loss.item()) * bs       # loss is sum over batch
            n_seen += bs
        
        scheduler.step()

        train_loss = running_loss / max(n_seen, 1)
        
        val_res = evaluate(model, val_loader, device, task=args.task,
                           standardize_target=args.standardize_target,
                           y_mean=y_mean, y_std=y_std)
        test_res = evaluate(model, test_loader, device, task=args.task,
                            standardize_target=args.standardize_target,
                            y_mean=y_mean, y_std=y_std)

        val_metric = float(val_res[metric_name])
        test_metric = float(test_res[metric_name])

        dt = time.time() - t0

        record = {"epoch": epoch, "train_loss": train_loss, "val": val_metric, "test": test_metric, "seconds": dt,
                  "metric": metric_name}
        if args.task == "reg":
            record.update({"val_rmse": float(val_res["rmse"]), "val_r2": float(val_res["r2"]),
                           "test_rmse": float(test_res["rmse"]), "test_r2": float(test_res["r2"])})
        history.append(record)
        if args.task == "cls":
            logger.log(f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_acc={val_metric:.4f} | test_acc={test_metric:.4f} | {dt:.1f}s")
        else:
            logger.log(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_mae={val_metric:.4f} | test_mae={test_metric:.4f} | "
                f"val_rmse={float(val_res['rmse']):.4f} | val_r2={float(val_res['r2']):.4f} | {dt:.1f}s"
            )

        if better(val_metric, best_val_metric):
            best_val_metric = val_metric
            torch.save({"model_state": _unwrap(model).state_dict(), "config": vars(mcfg), "epoch": epoch}, best_path)
            logger.log(f"  New best val_{metric_name}={best_val_metric:.4f}. Saved: {best_path}")

            if args.save_preds:
                # Save validation predictions at best checkpoint
                if args.task == "cls":
                    np.save(outdir / "val_logits_best.npy", val_res["logits"])
                    np.save(outdir / "val_pred_best.npy", val_res["pred"])
                else:
                    np.save(outdir / "val_pred_best.npy", val_res["pred"])
                np.save(outdir / "val_y.npy", val_res["y"])

    # Load best and evaluate on test
    ckpt = torch.load(best_path, map_location=device)
    _unwrap(model).load_state_dict(ckpt["model_state"])
    test_res = evaluate(
        model,
        test_loader,
        device,
        task=args.task,
        standardize_target=args.standardize_target,
        y_mean=y_mean,
        y_std=y_std,
    )

    if args.task == "cls":
        logger.log(f"Best checkpoint epoch={ckpt['epoch']} | test_acc={float(test_res['acc']):.4f}")
    else:
        logger.log(
            f"Best checkpoint epoch={ckpt['epoch']} | test_mae={float(test_res['mae']):.4f} | "
            f"test_rmse={float(test_res['rmse']):.4f} | test_r2={float(test_res['r2']):.4f}"
        )

    # Save artifacts
    summary: Dict[str, float | int | str | None] = {
        "task": args.task,
        "best_epoch": int(ckpt["epoch"]),
        "best_val_metric_name": metric_name,
        "best_val_metric": float(best_val_metric),
    }
    if args.task == "cls":
        summary.update({"test_acc": float(test_res["acc"])})
    else:
        summary.update({
            "test_mae": float(test_res["mae"]),
            "test_rmse": float(test_res["rmse"]),
            "test_r2": float(test_res["r2"]),
            "standardize_target": bool(args.standardize_target),
            "target_mean": y_mean,
            "target_std": y_std,
        })
    save_json(summary, outdir / "summary.json")
    save_json(history, outdir / "history.json")

    if args.save_preds:
        if args.task == "cls":
            np.save(outdir / "test_logits_best.npy", test_res["logits"])
            np.save(outdir / "test_pred_best.npy", test_res["pred"])
        else:
            np.save(outdir / "test_pred_best.npy", test_res["pred"])
        np.save(outdir / "test_y.npy", test_res["y"])

    logger.log("Done.")


if __name__ == "__main__":
    main()
