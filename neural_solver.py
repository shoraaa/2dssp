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
# Utilities
# ----------------------------

def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply -inf to masked positions, then log_softmax."""
    very_neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask.bool(), very_neg)
    return F.log_softmax(logits, dim=dim)

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    very_neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask.bool(), very_neg)
    return F.softmax(logits, dim=dim)

def positional_encoding_2d(h: int, w: int, d: int, device) -> torch.Tensor:
    """Simple fixed 2D sinusoidal PE; returns (1, d, h, w). d must be divisible by 4."""
    assert d % 4 == 0
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

    # duplicate to fill remaining channels if any (or keep zeros)
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


# ----------------------------
# Example feature builder (plug in your own)
# ----------------------------

@dataclass
class StepBatch:
    """
    One construction step (batched).
    You must populate these from your environment/generator.
    """
    occ: torch.Tensor              # (B, C_occ, H, W)
    tiles_left: torch.Tensor       # (B, M, d_tile)  -- or produce via TileCNN
    tiles_left_mask: torch.Tensor  # (B, M)
    cand_feats: torch.Tensor       # (B, A, F)
    cand_mask: torch.Tensor        # (B, A)
    cand_tile_idx: torch.Tensor    # (B, A), integer
    expert_action: Optional[torch.Tensor] = None  # (B,) index into A (for IL)


# ----------------------------
# Losses
# ----------------------------

def imitation_loss(policy_logits: torch.Tensor, cand_mask: torch.Tensor, expert_action: torch.Tensor) -> torch.Tensor:
    # log prob under masked softmax
    logp = masked_log_softmax(policy_logits, cand_mask, dim=-1)
    nll = F.nll_loss(logp, expert_action, reduction='mean')
    return nll

def value_loss(pred_value: torch.Tensor, target_value: torch.Tensor, clip_value: Optional[float] = None) -> torch.Tensor:
    if clip_value is None:
        return F.mse_loss(pred_value, target_value, reduction='mean')
    # Huber loss
    return F.smooth_l1_loss(pred_value, target_value, reduction='mean')

def ppo_loss(
    old_logp: torch.Tensor, new_logits: torch.Tensor, cand_mask: torch.Tensor,
    actions: torch.Tensor, advantages: torch.Tensor, clip_eps: float = 0.2, entropy_coef: float = 0.01
) -> torch.Tensor:
    new_logp_all = masked_log_softmax(new_logits, cand_mask, dim=-1)
    new_logp_a = new_logp_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    ratio = (new_logp_a - old_logp).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy = -torch.mean(torch.min(unclipped, clipped))

    # entropy term encourages exploration over feasible actions
    entropy = -(new_logp_all.exp() * new_logp_all).sum(dim=-1)
    entropy = (entropy * (cand_mask.any(dim=-1).float())).mean()

    return policy - entropy_coef * entropy


# ----------------------------
# Training loops
# ----------------------------

def train_imitation(
    model: NeuralSolver,
    data_iter,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_nll, total_v = 0.0, 0.0
    n_steps = 0

    for batch in data_iter:
        # batch: StepBatch
        occ = batch.occ.to(device)
        tiles_left = batch.tiles_left.to(device)
        tiles_left_mask = batch.tiles_left_mask.to(device)
        cand_feats = batch.cand_feats.to(device)
        cand_mask = batch.cand_mask.to(device)
        cand_tile_idx = batch.cand_tile_idx.to(device)
        expert_action = batch.expert_action.to(device)

        logits, value = model(occ, tiles_left, tiles_left_mask, cand_feats, cand_mask, cand_tile_idx)
        loss_pol = imitation_loss(logits, cand_mask, expert_action)
        # optional: bootstrapped value target; for IL you can skip or supply oracle cost-to-go
        loss = loss_pol

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_nll += loss_pol.item()
        total_v += value.detach().abs().mean().item()
        n_steps += 1

    return {"nll": total_nll / max(1, n_steps), "value_abs": total_v / max(1, n_steps)}


# ----------------------------
# Inference (beam search)
# ----------------------------

@torch.no_grad()
def beam_search_step(
    model: NeuralSolver,
    batch: StepBatch,
    beam_width: int = 8,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
):
    """
    Single decision step with beam scoring over candidates.
    For full-episode beam search, call this iteratively while updating your environment.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    occ = batch.occ.to(device)
    tiles_left = batch.tiles_left.to(device)
    tiles_left_mask = batch.tiles_left_mask.to(device)
    cand_feats = batch.cand_feats.to(device)
    cand_mask = batch.cand_mask.to(device)
    cand_tile_idx = batch.cand_tile_idx.to(device)

    logits, _ = model(occ, tiles_left, tiles_left_mask, cand_feats, cand_mask, cand_tile_idx)
    logits = logits / max(1e-6, temperature)
    probs = masked_softmax(logits, cand_mask, dim=-1)
    # pick top-k per instance
    topk = min(beam_width, probs.size(-1))
    values, indices = probs.topk(topk, dim=-1)
    return indices, values  # (B, K), (B, K)


# ----------------------------
# Example: putting it together
# ----------------------------

def build_model_example(
    c_occ: int,
    d_tile: int = 64,
    d_model: int = 128,
    cand_feat_dim: int = 16,
) -> NeuralSolver:
    return NeuralSolver(c_occ=c_occ, d_tile=d_tile, d_model=d_model, cand_feat_dim=cand_feat_dim)


# ----------------------------
# HOW TO INTEGRATE WITH YOUR PIPELINE
# ----------------------------
#
# 1) Precompute tile embeddings (optional but recommended)
#    - Use TileCNN on your (C_sym, n, n) tiles; cache per tile id.
#    - During a step, assemble tiles_left (B, M, d_tile) by indexing into cache.
#
# 2) Build StepBatch from your candidate generator:
#    - occ: crop around bbox; channels = symbol one-hots + empty (C_occ = |Σ|+1)
#    - tiles_left: from your cache; tiles_left_mask indicates valid slots
#    - cand_feats: per-candidate features, e.g.:
#        [sum_overlap_H, pheromone_Tsum, delta_m_if_place, dx_norm, dy_norm,
#         touches_left, touches_right, touches_up, touches_down, is_adjacency, ...]
#    - cand_mask: feasibility mask (True where candidate is legal)
#    - cand_tile_idx: for each candidate, which tile id it places (index into tiles_left row)
#    - expert_action (for IL): the chosen candidate index from your oracle/ACO/MILP
#
# 3) Train with train_imitation() first. Later, switch to PPO:
#    - Roll out your environment using the model policy (sample or top-1).
#    - Compute advantages from episode rewards (e.g., -Δm per step, final compactness).
#    - Use ppo_loss() with stored old_logp, actions, advantages; plus value_loss().
#
# 4) Inference:
#    - At each step, call model to get masked logits; pick argmax or sample.
#    - For higher quality, use beam_search_step() and commit the best branch
#      (maintain multiple env clones if you want full beam through the episode).
#
# 5) Optional: add a tiny GNN over placed tiles
#    - Create node feats (tile emb, degree, boundary exposure), edge feats (dx,dy,ov).
#    - Pool to a layout embedding and fuse into e_state (concat with raster/set encodings).
#
# That’s it — plug your generator & environment into StepBatch, and this model will act as
# the neural policy/value for your solver.
