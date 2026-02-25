from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ptdec import DEC  # type: ignore

@dataclass
class ModelConfig:
    # Input cluster feature (after skipping any positional columns)
    init_dim: int = 1080
    # Transformer
    hidden_dim: int = 500
    num_heads: int = 10
    dim_feedforward: int = 1024
    num_layers: int = 1
    dropout: float = 0.1
    # Block pooling
    num_blocks: int = 4653  # will be inferred by data_index at runtime; kept for sanity checks
    block_size: int = 3
    pooling: str = "mean"  # "mean" or "masked_mean" (exclude background id=0)
    # DEC / OCR
    dec_encoder_hidden_size: int = 32
    dec_clusters: int = 25
    # Task head
    num_classes: int = 2


class AtlasFreeBNT(nn.Module):
    """Atlas-free Brain Network Transformer for sex classification.

    Expected inputs:
      - X_dict: float tensor [B, max_clusters_in_batch+1, init_dim]
          Row 0 is reserved for background (zero vector). Cluster IDs in C/D use:
            0 = background, 1..a = real clusters.
      - data_index: long tensor [B, num_blocks, block_size^3]
          Each row provides the cluster IDs inside each 3x3x3 block.
          Pooling is controlled by cfg.pooling:
            - "mean": mean over all entries (background included)
            - "masked_mean": mean over non-background entries only

    Note:
      "mean" pooling with background IDs will dilute activations in boundary/background-heavy blocks.
      This is intentional in your original setup; "masked_mean" is provided as an optional alternative.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.fnn = nn.Sequential(
            nn.Linear(cfg.init_dim, cfg.init_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.init_dim, cfg.hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # DEC acts as OCR readout and should return [batch, dec_clusters, hidden_dim]
        self.dec_encoder = nn.Sequential(
            nn.Linear(cfg.hidden_dim * cfg.num_blocks, cfg.dec_encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cfg.dec_encoder_hidden_size, cfg.dec_encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cfg.dec_encoder_hidden_size, cfg.hidden_dim * cfg.num_blocks),
        )
        self.dec = DEC(
            cluster_number=cfg.dec_clusters,
            hidden_dimension=cfg.hidden_dim,
            encoder=self.dec_encoder,
            orthogonal=True,
            freeze_center=True,
            project_assignment=True,
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * cfg.dec_clusters, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, X_dict: torch.Tensor, data_index: torch.Tensor) -> torch.Tensor:
        # X_dict: [B, C, init_dim]
        # data_index: [B, U, 27]
        B, U, K = data_index.shape
        if U != self.cfg.num_blocks:
            raise ValueError(
                f"data_index has U={U} blocks, but cfg.num_blocks={self.cfg.num_blocks}. "
                "Set cfg.num_blocks to match your generated block_coords."
            )
        if K != self.cfg.block_size ** 3:
            raise ValueError(
                f"Expected last dim {self.cfg.block_size ** 3} but got {K}"
            )

        # Encode cluster features -> [B, C, hidden_dim]
        X_dict = self.fnn(X_dict)

        # Gather cluster embeddings for each block and mean pool:

        ## first attempt out-of-momory
        ## X_tmp: [B, U, 27, hidden_dim] -> mean over 27 -> [B, U, hidden_dim]
        ## Batched gather:
        ## X_dict: [B, C, H] -> expand to [B, U, C, H]
        #X_exp = X_dict.unsqueeze(1).expand(-1, U, -1, -1)
        #idx_exp = data_index.unsqueeze(-1).expand(-1, -1, -1, self.cfg.hidden_dim)
        #X_tmp = torch.take_along_dim(X_exp, idx_exp, dim=2)  # [B, U, 27, H]
        #if self.cfg.pooling == "mean":
        #    # Mean over all entries (including background id=0).
        #    X_nodes = torch.mean(X_tmp, dim=2)  # [B, U, H]
        #elif self.cfg.pooling == "masked_mean":
        #    # Mean over non-background indices only.
        #    mask = (data_index != 0).unsqueeze(-1).to(X_tmp.dtype)  # [B, U, 27, 1]
        #    denom = mask.sum(dim=2).clamp_min(1.0)  # [B, U, 1]
        #    X_nodes = (X_tmp * mask).sum(dim=2) / denom
        #else:
        #    raise ValueError(f"Unknown pooling mode: {self.cfg.pooling}. Use 'mean' or 'masked_mean'.")

        # second attempt handle pooling one subject at a time
        # X_dict: [B, C, H]
        # data_index: [B, U, 27]  (cluster ids in [0..C-1] or [0..C] depending on your padding)
        B, U, K = data_index.shape  # K should be 27
        H = self.cfg.hidden_dim

        X_nodes = torch.empty((B, U, H), device=X_dict.device, dtype=X_dict.dtype)
        
        for b in range(B):
            # X_b: [C, H], idx_b: [U, 27]
            X_b = X_dict[b]
            idx_b = data_index[b].long()
        
            # Gather only for this subject: [U, 27, H]
            X_tmp_b = X_b[idx_b]
        
            if self.cfg.pooling == "mean":
                X_nodes[b] = X_tmp_b.mean(dim=1)  # [U, H]
        
            elif self.cfg.pooling == "masked_mean":
                # exclude background (idx==0)
                mask = (idx_b != 0).unsqueeze(-1).to(X_tmp_b.dtype)  # [U, 27, 1]
                denom = mask.sum(dim=1).clamp_min(1.0)               # [U, 1]
                X_nodes[b] = (X_tmp_b * mask).sum(dim=1) / denom     # [U, H]
            else:
                raise ValueError(f"Unknown pooling mode: {self.cfg.pooling}")

        # Transformer encoder over block nodes
        Z = self.encoder(X_nodes)  # [B, U, hidden_dim]

        # DEC readout
        Z_dec, assignment = self.dec(Z)  # Z_dec: [B, dec_clusters, hidden_dim]
        logits = self.mlp_head(Z_dec.reshape(Z_dec.shape[0], -1))
        return logits
