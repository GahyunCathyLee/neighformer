#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — EncDecFormer model definition + builder + scheduler.

Architecture
────────────
  1. Ego + Neighbor tokens → projected to d_model
  2. Time PE + Slot embedding → Transformer Encoder → memory
  3. M learnable queries → Transformer Decoder (cross-attn to memory)
  4. Trajectory head  : (B, M, Tf, 2)
     Score head       : (B, M)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEncoding(nn.Module):
    """
    Sinusoidal positional encoding indexed by an arbitrary integer sequence.

    Parameters
    ----------
    d_model : int
    max_len : int   upper bound on sequence length
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)          # (max_len, d_model)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        indices : (L,) long tensor — time indices

        Returns
        -------
        (L, d_model)
        """
        return self.pe[indices]                 # type: ignore[index]


# ──────────────────────────────────────────────────────────────────────────────
# EncDecFormer
# ──────────────────────────────────────────────────────────────────────────────

class EncDecFormer(nn.Module):
    """
    Encoder-Decoder Transformer for multi-modal trajectory prediction.

    Parameters
    ----------
    T            : int    history timesteps
    Tf           : int    future timesteps
    K            : int    neighbor slots
    ego_dim      : int    ego feature dimension
    nb_dim       : int    neighbor feature dimension
    d_model      : int    transformer hidden dimension  (default 128)
    nhead        : int    attention heads               (default 4)
    enc_layers   : int    encoder depth                 (default 2)
    dec_layers   : int    decoder depth                 (default 2)
    dropout      : float                                (default 0.1)
    M            : int    number of predicted modes     (default 6)
    use_neighbors : bool  toggle neighbor tokens        (default True)
    use_slot_emb  : bool  learned slot embedding        (default True)
    return_scores : bool  return (traj, scores) tuple   (default True)

    Forward
    -------
    Input  : x_ego (B,T,ego_dim), x_nb (B,T,K,nb_dim), nb_mask (B,T,K)
    Output : (traj, scores)  if return_scores else traj
               traj   : (B, M, Tf, 2)
               scores : (B, M)
    """

    def __init__(
        self,
        T: int,
        Tf: int,
        K: int,
        ego_dim: int,
        nb_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dropout: float = 0.1,
        M: int = 6,
        use_neighbors: bool = True,
        use_slot_emb: bool = True,
        return_scores: bool = True,
    ) -> None:
        super().__init__()

        self.T  = T
        self.Tf = Tf
        self.K  = K
        self.M  = M
        self.use_neighbors = use_neighbors
        self.use_slot_emb  = use_slot_emb
        self.return_scores = return_scores

        # ── input projections ─────────────────────────────────────────────────
        self.ego_proj = nn.Linear(ego_dim, d_model)
        self.nb_proj  = nn.Linear(nb_dim,  d_model)

        # ── positional / slot encodings ───────────────────────────────────────
        self.time_enc = SinusoidalTimeEncoding(d_model, max_len=T)
        self.slot_emb = nn.Embedding(1 + K, d_model) if use_slot_emb else None

        # ── encoder ───────────────────────────────────────────────────────────
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=enc_layers,
        )

        # ── learnable mode queries ────────────────────────────────────────────
        self.query_emb = nn.Embedding(M, d_model)

        # ── decoder ───────────────────────────────────────────────────────────
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=dec_layers,
        )

        # ── output heads ──────────────────────────────────────────────────────
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, Tf * 2),
        )
        self.score_head = nn.Linear(d_model, 1)

    def forward(
        self,
        x_ego: torch.Tensor,
        x_nb: torch.Tensor,
        nb_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Parameters
        ----------
        x_ego   : (B, T, ego_dim)
        x_nb    : (B, T, K, nb_dim)
        nb_mask : (B, T, K)  True = neighbor exists

        Returns
        -------
        (traj, scores)  if return_scores
          traj   : (B, M, Tf, 2)
          scores : (B, M)
        traj only       otherwise
        """
        B, T, _ = x_ego.shape

        ego_tok = self.ego_proj(x_ego)          # (B, T, d)

        if self.use_neighbors:
            nb_tok = self.nb_proj(x_nb)         # (B, T, K, d)

            # zero-out absent neighbors
            nb_tok = nb_tok * nb_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1)

            # interleave ego + neighbor tokens: (B, T*(1+K), d)
            tok = torch.cat([ego_tok.unsqueeze(2), nb_tok], dim=2)
            tok = tok.reshape(B, T * (1 + self.K), -1)

            # key padding mask: True = pad (ignored by attention)
            valid = torch.ones(
                (B, T, 1 + self.K), device=tok.device, dtype=torch.bool
            )
            valid[:, :, 1:] = nb_mask
            key_pad = ~valid.reshape(B, -1)
            key_pad[:, 0] = False               # always attend to first token

            t_idx   = torch.arange(T, device=tok.device).repeat_interleave(1 + self.K)
            slot_ids = torch.arange(1 + self.K, device=tok.device).repeat(T)
        else:
            tok     = ego_tok
            key_pad = None
            t_idx   = torch.arange(T, device=tok.device)

        # time PE
        tok = tok + self.time_enc(t_idx).unsqueeze(0)

        # slot embedding
        if self.use_neighbors and self.slot_emb is not None:
            tok = tok + self.slot_emb(slot_ids).unsqueeze(0)

        # encoder
        memory = self.encoder(tok, src_key_padding_mask=key_pad)   # (B, L, d)

        # decoder: M learnable queries cross-attend to memory
        queries  = self.query_emb.weight.unsqueeze(0).expand(B, -1, -1)  # (B, M, d)
        dec_out  = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=key_pad,
        )                                                           # (B, M, d)

        traj   = self.traj_head(dec_out).view(B, self.M, self.Tf, 2)  # (B, M, Tf, 2)
        scores = self.score_head(dec_out).squeeze(-1)                  # (B, M)

        return (traj, scores) if self.return_scores else traj


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler builder
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    sched_type: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    LambdaLR scheduler with optional linear warmup.

    sched_type
    ──────────
    "cosine"  : cosine annealing after warmup
    "none"    : constant LR after warmup
    """
    sched_type = (sched_type or "none").lower()

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if sched_type == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Dict[str, Any], ego_dim: int, nb_dim: int) -> EncDecFormer:
    """
    Config-driven factory for EncDecFormer.

    ego_dim / nb_dim 은 dataset.ego_feature_dim / nb_feature_dim 에서 전달받습니다.

    Expected cfg structure
    ──────────────────────
    model:
      name      : "encdecformer"
      T         : 6
      Tf        : 15
      K         : 8
      d_model   : 128
      nhead     : 4
      enc_layers: 2
      dec_layers: 2
      dropout   : 0.1
      M         : 6
      use_neighbors : true
      return_scores : true
    """
    mcfg = cfg.get("model", {})
    name = str(mcfg.get("name", "encdecformer")).lower().replace("_", "").replace("-", "")

    if name != "encdecformer":
        raise ValueError(
            f"Unknown model name: '{mcfg.get('name')}'. Expected 'encdecformer'."
        )

    return EncDecFormer(
        T            = int(mcfg["T"]),
        Tf           = int(mcfg["Tf"]),
        K            = int(mcfg.get("K", 8)),
        ego_dim      = ego_dim,
        nb_dim       = nb_dim,
        d_model      = int(mcfg.get("d_model",    128)),
        nhead        = int(mcfg.get("nhead",         4)),
        enc_layers   = int(mcfg.get("enc_layers",    2)),
        dec_layers   = int(mcfg.get("dec_layers",    2)),
        dropout      = float(mcfg.get("dropout",   0.1)),
        M            = int(mcfg.get("M",             6)),
        use_neighbors = bool(mcfg.get("use_neighbors", True)),
        return_scores = bool(mcfg.get("return_scores",  True)),
    )