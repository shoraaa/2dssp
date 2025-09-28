"""
Neural solver model for 2D-SSP with maskable candidate head.

Inputs (match env.build_step_batch_from_env):
  - occ_crop: (B, C, H, W) raster (C = A+1)
  - placed_mask: (B, T) bool
  - offsets_xy: (B, T, 2) int  (unused here but available)
  - bbox_min/max: (B, 2) int
  - candidates: CandidateBatch with concatenated rows across B
Optionally:
  - tile_embs: (B, T, E_tile) precomputed embeddings per tile (e.g., Tiny CNN upstream)

Outputs:
  - policy_logits: (K,) scores for each candidate row (to be masked/softmax per batch)
  - value: (B,) state value estimate
  - aux (dict): optional intermediates useful for losses/diagnostics

Notes
-----
* This module keeps inference GPU-friendly: raster CNN, lightweight Transformer for tile set,
  simple graph summary, and candidate MLP + cross-attention scoring.
* Grouped (per-batch) softmax helpers are provided; many training loops prefer raw logits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Small building blocks
# ----------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d_prev = d_in
        for i in range(n_layers - 1):
            layers += [nn.Linear(d_prev, d_hidden), nn.GELU(), nn.Dropout(dropout)]
            d_prev = d_hidden
        layers += [nn.Linear(d_prev, d_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------------------------------
# Encoders: Raster, Set (remaining tiles), Graph (placed layout)
# ----------------------------------------------------------------------------

class RasterCNN(nn.Module):
    def __init__(self, in_ch: int, d_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Linear(96, d_out)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        h = self.conv(x)
        # Global mean pool
        h = h.mean(dim=(2,3))  # (B,96)
        return self.head(h)

class TileSetEncoder(nn.Module):
    """Transformer encoder over remaining tile embeddings.

    Inputs:
      - tile_embs: (B, T, E)
      - remain_mask: (B, T) bool (True if tile remains)
    Output: (B, D) pooled embedding
    """
    def __init__(self, e_in: int, d_model: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True, dropout=dropout, activation="gelu")
        self.proj_in = nn.Linear(e_in, d_model)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = nn.Linear(d_model, d_model)
    def forward(self, tile_embs: torch.Tensor, remain_mask: torch.Tensor) -> torch.Tensor:
        if tile_embs is None:
            # If not provided, use zeros of matching batch/length
            B, T = remain_mask.shape
            E = self.pool.in_features
            tile_embs = torch.zeros((B, T, E), device=remain_mask.device)
        h = self.proj_in(tile_embs)  # (B,T,D)
        # Transformer expects True = pad; we invert remain_mask (remain=False => pad=True)
        key_padding_mask = ~remain_mask
        h = self.enc(h, src_key_padding_mask=key_padding_mask)
        # masked mean pool over remaining positions
        mask = remain_mask.unsqueeze(-1)  # (B,T,1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = (h * mask).sum(dim=1) / denom
        return self.pool(pooled)

class GraphSummary(nn.Module):
    """Lightweight graph summary: sum & max over placed tile embeddings + degrees.

    We avoid explicit message passing to keep it simple/fast; in practice you can
    replace with a proper GNN if you maintain node features.
    """
    def __init__(self, e_tile: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(e_tile + 1, d_out)  # + degree
    def forward(self, tile_embs: Optional[torch.Tensor], placed_mask: torch.Tensor, degrees: Optional[torch.Tensor]=None) -> torch.Tensor:
        B, T = placed_mask.shape
        if tile_embs is None:
            E = self.proj.in_features - 1
            tile_embs = torch.zeros((B, T, E), device=placed_mask.device)
        if degrees is None:
            deg = placed_mask.to(tile_embs.dtype)  # proxy degree=1 for placed, 0 otherwise
        else:
            deg = degrees.to(tile_embs.dtype)
        feats = torch.cat([tile_embs, deg.unsqueeze(-1)], dim=-1)
        # masked mean + max
        mask = placed_mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        mean = (feats * mask).sum(dim=1) / denom
        maxv, _ = feats.masked_fill(~mask, float('-inf')).max(dim=1)
        maxv[maxv == float('-inf')] = 0.0
        h = mean + maxv
        return self.proj(h)

# ----------------------------------------------------------------------------
# Candidate encoder + scorer
# ----------------------------------------------------------------------------

class CandidateEncoder(nn.Module):
    def __init__(self, d_in_scalar: int, d_kind: int, e_tile: int, d_token: int):
        super().__init__()
        self.kind_emb = nn.Embedding(3, d_kind)
        self.proj = MLP(d_in_scalar + d_kind + e_tile, d_token, d_token, n_layers=2)
    def forward(self, scalars: torch.Tensor, kind: torch.Tensor, tile_vec: torch.Tensor) -> torch.Tensor:
        # scalars: (K, S) normalized/scaled; kind: (K,), tile_vec: (K, E)
        k = self.kind_emb(kind.clamp(min=0, max=2))
        x = torch.cat([scalars, k, tile_vec], dim=-1)
        return self.proj(x)  # (K, d_token)

class DotProductScorer(nn.Module):
    def __init__(self, d_state: int, d_token: int):
        super().__init__()
        self.q = nn.Linear(d_state, d_token)
        self.k = nn.Linear(d_token, d_token, bias=False)  # identity-like; keep simple
    def forward(self, state_vec: torch.Tensor, cand_tokens: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        # state_vec: (B, D), cand_tokens: (K, d_token), starts: (B+1,)
        q = self.q(state_vec)  # (B,d)
        # For each batch b, dot q[b] with its slice of candidates
        K = cand_tokens.size(0)
        logits = cand_tokens.new_zeros((K,))
        for b in range(starts.numel() - 1):
            s, e = int(starts[b].item()), int(starts[b+1].item())
            if e <= s:
                continue
            logits[s:e] = (cand_tokens[s:e] @ self.k(q[b]).unsqueeze(-1)).squeeze(-1)
        return logits

# ----------------------------------------------------------------------------
# Full Model
# ----------------------------------------------------------------------------

class NeuralSolverNet(nn.Module):
    def __init__(
        self,
        raster_channels: int,
        e_tile: int = 64,
        d_model: int = 256,
        d_token: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.e_tile = e_tile
        # Encoders
        self.raster = RasterCNN(raster_channels, d_out=d_model)
        self.set_enc = TileSetEncoder(e_in=e_tile, d_model=d_model, nhead=4, num_layers=2, dropout=dropout)
        self.graph_enc = GraphSummary(e_tile=e_tile, d_out=d_model)
        # Fuse state
        self.fuse = MLP(d_in=3*d_model, d_hidden=d_model, d_out=d_model, n_layers=2, dropout=dropout)
        # Candidates
        self.cand_enc = CandidateEncoder(d_in_scalar=5, d_kind=16, e_tile=e_tile, d_token=d_token)
        self.scorer = DotProductScorer(d_state=d_model, d_token=d_token)
        # Heads
        self.value_head = MLP(d_model, d_model, 1, n_layers=2, dropout=dropout)

    # ------------- helpers -------------
    @staticmethod
    def _normalize_features(x: torch.Tensor, denom: float = 1.0) -> torch.Tensor:
        return x.to(torch.float32) / max(denom, 1.0)

    @staticmethod
    def grouped_log_softmax(logits: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        """Compute per-batch log-softmax over candidate logits.
        Returns a tensor of shape (K,) aligned with logits.
        """
        out = logits.new_empty(logits.shape)
        for b in range(starts.numel() - 1):
            s, e = int(starts[b].item()), int(starts[b+1].item())
            if e <= s:
                continue
            out[s:e] = F.log_softmax(logits[s:e], dim=0)
        return out

    # ------------- forward -------------
    def forward(
        self,
        step_batch: Dict[str, Any],
        tile_embs: Optional[torch.Tensor] = None,  # (B,T,E_tile)
        extra_cand_scalars: Optional[torch.Tensor] = None,  # (K, S_extra)
    ) -> Dict[str, torch.Tensor]:
        """Compute policy logits for candidates and value for states.
        
        step_batch expects keys: 'occ_crop', 'placed_mask', 'bbox_min', 'bbox_max', 'candidates'
        """
        occ = step_batch["occ_crop"]              # (B,C,H,W)
        placed_mask = step_batch["placed_mask"]    # (B,T)
        bbox_min = step_batch["bbox_min"]          # (B,2)
        bbox_max = step_batch["bbox_max"]          # (B,2)
        cands = step_batch["candidates"]           # CandidateBatch

        B, T = placed_mask.shape
        dev = occ.device

        # ---- State encoders ----
        e_r = self.raster(occ)  # (B,D)
        # For set encoder, remaining mask is ~placed
        remain_mask = ~placed_mask
        if tile_embs is None:
            # If no embeddings provided, use zeros (can be swapped later)
            tile_vecs = torch.zeros((B, T, self.e_tile), device=dev)
        else:
            assert tile_embs.shape[-1] == self.e_tile, "tile_embs last dim must match e_tile"
            tile_vecs = tile_embs
        e_s = self.set_enc(tile_vecs, remain_mask)  # (B,D)
        # Graph summary over placed
        e_g = self.graph_enc(tile_vecs, placed_mask)  # (B,D)
        # Fuse
        e_state = self.fuse(torch.cat([e_r, e_s, e_g], dim=-1))  # (B,D)

        # ---- Candidate tokens ----
        K = cands.b.size(0)
        if K == 0:
            # No candidates anywhere; return minimal tensors
            value = self.value_head(e_state).squeeze(-1)
            return {"policy_logits": torch.empty(0, device=dev), "value": value, "logits_grouped": torch.empty(0, device=dev)}

        # Gather per-candidate tile vectors v
        v_vec = tile_vecs[cands.b, cands.v]  # (K,E)

        # Scalar features per candidate (normalize for stability)
        # Use: ov_size, d_m, touch_edges, cur_m (per-batch), remaining_count (per-batch)
        cur_m = (bbox_max - bbox_min).max(dim=1).values  # (B,)
        rem_count = (~placed_mask).sum(dim=1)            # (B,)
        scalars = torch.stack([
            cands.ov_size.to(torch.float32),
            cands.d_m.to(torch.float32),
            cands.touch_edges.to(torch.float32),
            cur_m[cands.b].to(torch.float32),
            rem_count[cands.b].to(torch.float32),
        ], dim=-1)
        # Simple scaling
        scalars = self._normalize_features(scalars, denom=max(1.0, float(rem_count.max().item())))
        if extra_cand_scalars is not None:
            scalars = torch.cat([scalars, extra_cand_scalars.to(scalars.dtype)], dim=-1)
        cand_tokens = self.cand_enc(scalars, cands.kind, v_vec)  # (K,d_token)

        # ---- Score candidates with state vector (per-batch) ----
        logits = self.scorer(e_state, cand_tokens, cands.starts)  # (K,)

        # ---- Value head ----
        value = self.value_head(e_state).squeeze(-1)  # (B,)

        return {
            "policy_logits": logits,
            "value": value,
        }


# ----------------------------------------------------------------------------
# Optional tile CNN (for precomputing tile_embs). Not used inside the model.
# ----------------------------------------------------------------------------

class TinyTileCNN(nn.Module):
    def __init__(self, alphabet_size: int, e_tile: int = 64):
        super().__init__()
        # Input is (A+1, n, n) one-hot per tile; if tiles are ints, prepare upstream.
        self.conv = nn.Sequential(
            nn.Conv2d(alphabet_size + 1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(64, e_tile)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).mean(dim=(2,3))
        return self.head(h)


# ----------------------------------------------------------------------------
# Convenience: factory
# ----------------------------------------------------------------------------

def build_model(raster_channels: int, e_tile: int = 64, d_model: int = 256, d_token: int = 256, dropout: float = 0.0) -> NeuralSolverNet:
    return NeuralSolverNet(raster_channels=raster_channels, e_tile=e_tile, d_model=d_model, d_token=d_token, dropout=dropout)
