"""
Neural Network Architecture for 2D Tile Placement

This module contains the neural network components for solving the 2D tile placement problem.
The architecture consists of:

1. TileCNN: Encodes individual tiles into embeddings
2. RasterCNN: Encodes the current layout state (occupancy grid)
3. SetEncoder: Processes the set of remaining tiles  
4. NeuralSolver: Main policy-value network that combines all components

The model takes as input:
- Visual state: Occupancy grid showing current tile placements
- Remaining tiles: Embeddings of tiles not yet placed
- Candidate actions: Features describing each possible next move

And outputs:
- Policy: Probability distribution over candidate actions
- Value: Estimated quality of the current state
"""

# neural_solver.py
# Minimal neural solver scaffold for the 2D tile-canvas problem.
# PyTorch-only (no external libs). Python 3.10+, torch 2.x recommended.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utility Functions
# ----------------------------

def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply log softmax with masking for invalid actions.
    
    Args:
        logits: Raw model outputs (B, A)  
        mask: Boolean mask where True = valid action (B, A)
        dim: Dimension to apply softmax over
        
    Returns:
        Log probabilities with invalid actions set to very negative values
    """
    very_neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask.bool(), very_neg)
    return F.log_softmax(logits, dim=dim)

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply softmax with masking for invalid actions.
    
    Args:
        logits: Raw model outputs (B, A)
        mask: Boolean mask where True = valid action (B, A)  
        dim: Dimension to apply softmax over
        
    Returns:
        Probabilities with invalid actions set to zero
    """
    very_neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask.bool(), very_neg)
    return F.softmax(logits, dim=dim)

def positional_encoding_2d(h: int, w: int, d: int, device) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional encodings.
    
    Args:
        h, w: Height and width of the 2D grid
        d: Embedding dimension (must be divisible by 4)
        device: Device to create tensor on
        
    Returns:
        Positional encoding tensor of shape (1, d, h, w)
    """
    assert d % 4 == 0, "Embedding dimension must be divisible by 4"
    pe = torch.zeros(1, d, h, w, device=device)
    d2 = d // 2
    d4 = d // 4

    y_pos = torch.arange(h, device=device).unsqueeze(1)
    x_pos = torch.arange(w, device=device).unsqueeze(0)

    div_y = torch.exp(torch.arange(0, d4, 2, device=device) * (-math.log(10000.0) / d4))
    div_x = torch.exp(torch.arange(0, d4, 2, device=device) * (-math.log(10000.0) / d4))

    pe[:, 0:d4:2, :, :] = torch.sin(y_pos * div_y).unsqueeze(2).repeat(1, 1, w)
    pe[:, 1:d4:2, :, :] = torch.cos(y_pos * div_y).unsqueeze(2).repeat(1, 1, w)
    pe[:, d4:d2:2, :, :] = torch.sin(x_pos * div_x).unsqueeze(1).repeat(1, h, 1)
    pe[:, d4+1:d2:2, :, :] = torch.cos(x_pos * div_x).unsqueeze(1).repeat(1, h, 1)

    return pe


# ----------------------------
# Encoders
# ----------------------------

class RasterCNN(nn.Module):
    """Encodes cropped occupancy raster (B, C_occ, H, W) into a vector."""
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, d_model),
            nn.ReLU(),
        )

    def forward(self, occ: torch.Tensor) -> torch.Tensor:
        # occ: (B, C_occ, H, W)
        x = self.conv(occ)
        x = self.proj(x)  # (B, d_model)
        return x


class TileCNN(nn.Module):
    """Encodes an n×n tile (symbol channels) into an embedding."""
    def __init__(self, in_ch: int, d_tile: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, d_tile), nn.ReLU(),
        )

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        # tiles: (B, M, C_sym, n, n) or (B*C, C_sym, n, n)
        if tiles.dim() == 5:
            B, M = tiles.size(0), tiles.size(1)
            x = tiles.view(B*M, *tiles.shape[2:])
            x = self.enc(x).view(B, M, -1)
        else:
            x = self.enc(tiles)
        return x


class SetEncoder(nn.Module):
    """Encodes the set of remaining tile embeddings with a Transformer encoder."""
    def __init__(self, d_in: int, d_model: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=dropout)
        self.inp = nn.Linear(d_in, d_model)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = nn.Linear(d_model, d_model)

    def forward(self, tiles_left: torch.Tensor, tiles_left_mask: torch.Tensor) -> torch.Tensor:
        # tiles_left: (B, M, d_in), mask: (B, M) with True for AVAILABLE (we’ll make pad mask=True means keep)
        x = self.inp(tiles_left)  # (B, M, d_model)
        # transformer expects mask where True=PAD (to ignore); invert:
        pad_mask = ~tiles_left_mask.bool()
        x = self.enc(x, src_key_padding_mask=pad_mask)  # (B, M, d_model)
        # masked mean pool
        mask = tiles_left_mask.unsqueeze(-1)  # (B, M, 1)
        x = x * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = x.sum(dim=1) / denom
        return self.pool(pooled)


# ----------------------------
# Policy-Value Model
# ----------------------------

class NeuralSolver(nn.Module):
    """
    Inputs (per step):
      occ: (B, C_occ, H, W)        - occupancy crop (symbol one-hots + empty)
      tiles_left: (B, M, d_tile)   - embeddings for remaining tiles (precomputed or from TileCNN)
      tiles_left_mask: (B, M)      - which entries in tiles_left are valid
      cand_feats: (B, A, F)        - per-candidate numeric features
      cand_mask: (B, A)            - which candidates are feasible
      cand_tile_idx: (B, A)        - which tile each candidate places (index into tiles_left row)
    Outputs:
      policy_logits: (B, A) masked
      value: (B,)
    """
    def __init__(self, c_occ: int, d_tile: int, d_model: int, cand_feat_dim: int):
        super().__init__()
        self.raster = RasterCNN(c_occ, d_model)
        self.setenc = SetEncoder(d_tile, d_model, n_heads=4, n_layers=2)
        self.fuse = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.cand_proj = nn.Sequential(
            nn.Linear(cand_feat_dim + d_tile, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        # attention-like scoring from state to each candidate token
        self.score = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        occ: torch.Tensor,
        tiles_left: torch.Tensor,
        tiles_left_mask: torch.Tensor,
        cand_feats: torch.Tensor,
        cand_mask: torch.Tensor,
        cand_tile_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, A, F = cand_feats.shape
        # Encode raster & set
        e_r = self.raster(occ)                           # (B, d_model)
        e_s = self.setenc(tiles_left, tiles_left_mask)   # (B, d_model)
        e_state = self.fuse(torch.cat([e_r, e_s], dim=-1))  # (B, d_model)

        # Gather tile embeddings for each candidate
        # tiles_left: (B, M, d_tile), cand_tile_idx: (B, A)
        b_idx = torch.arange(B, device=tiles_left.device).unsqueeze(-1).expand(B, A)
        cand_tile_emb = tiles_left[b_idx, cand_tile_idx]  # (B, A, d_tile)

        # Candidate tokens
        cand_tok = self.cand_proj(torch.cat([cand_feats, cand_tile_emb], dim=-1))  # (B, A, d_model)

        # Score with a simple multiplicative interaction: concat(e_state, cand_token)
        e_state_exp = e_state.unsqueeze(1).expand(B, A, -1)                        # (B, A, d_model)
        pair = torch.cat([e_state_exp, cand_tok], dim=-1)                          # (B, A, 2*d_model)
        logits = self.score(pair).squeeze(-1)                                      # (B, A)

        # Masked logits for policy, scalar value head
        value = self.value(e_state).squeeze(-1)                                    # (B,)
        return logits.masked_fill(~cand_mask.bool(), torch.finfo(logits.dtype).min), value

