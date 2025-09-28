#!/usr/bin/env python3
"""
Batched REINFORCE (with value baseline & entropy) for the 2D-SSP neural solver.

This script wires together env.py and net.py to run policy-gradient training on
synthetic data. It supports curriculum over T and n via CLI flags.

Key ideas
---------
- Discrete, maskable action space from env.candidates(); we sample exactly one
  candidate per active environment in the batch each step.
- Reward shaping: r_t = -Δm + alpha_overlap * ov_used.
- Baseline: value head V(s_t). Loss = -E[(R_t - V) * log π(a|s)] + λ_v * MSE + λ_H * H.
- Episodes end when all tiles are placed. We seed the first tile at (0,0).

Note
----
This is a lean trainer focused on correctness and GPU efficiency. Plug your own
expert data for IL pretraining if desired; this script implements on-policy RL.
"""
from __future__ import annotations

import os
import math
import argparse
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm, trange

# Local imports
from env2 import (
    TilePlacementEnv,
    make_synthetic_tiles,
    build_step_batch_from_env,
)
from net2 import build_model

# ------------------------------- utils ---------------------------------------

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('train2.log')
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_log_softmax_grouped(logits: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    """Compute per-group log-softmax over concatenated candidate logits.
    Returns (K,) tensor aligned with logits. Groups are contiguous slices [s:e).
    """
    out = torch.empty_like(logits)
    for b in range(starts.numel() - 1):
        s, e = int(starts[b].item()), int(starts[b+1].item())
        if e <= s:
            continue
        out[s:e] = F.log_softmax(logits[s:e], dim=0)
    return out


@dataclass
class TrainConfig:
    # Data
    batch_size: int = 16
    T: int = 30
    n: int = 4
    alphabet: int = 2
    hole_prob: float = 0.0
    # Model
    e_tile: int = 64
    d_model: int = 256
    d_token: int = 256
    dropout: float = 0.0
    # Optim
    lr: float = 3e-4
    weight_decay: float = 1e-2
    # RL
    alpha_overlap: float = 0.01
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    # Run
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    steps_per_epoch: int = 256  # episodes per epoch (since each instance is one episode)
    epochs: int = 50
    save_every: int = 10
    out_dir: str = "./checkpoints"
    log_level: str = "INFO"


# --------------------------- sampling helpers --------------------------------

def sample_actions_per_env(logits: torch.Tensor, cands, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each environment b that has candidates, sample exactly one action.

    Returns: (sel_idx, b, v, x, y) where each has shape (B_active,)
             sel_idx indexes into the concatenated candidate arrays.
    """
    sel_idx = []
    b_list = []
    v_list = []
    x_list = []
    y_list = []

    for b in range(cands.starts.numel() - 1):
        s, e = int(cands.starts[b].item()), int(cands.starts[b+1].item())
        if e <= s:
            continue
        slice_logits = logits[s:e]
        if greedy:
            j = int(torch.argmax(slice_logits).item())
        else:
            probs = F.softmax(slice_logits, dim=0)
            j = int(torch.multinomial(probs, num_samples=1).item())
        k = s + j
        sel_idx.append(k)
        b_list.append(int(cands.b[k].item()))
        v_list.append(int(cands.v[k].item()))
        x_list.append(int(cands.x[k].item()))
        y_list.append(int(cands.y[k].item()))

    if len(sel_idx) == 0:
        # No actions (e.g., episodes all done)
        z = torch.empty(0, dtype=torch.long, device=logits.device)
        return z, z, z, z, z

    dev = logits.device
    return (
        torch.tensor(sel_idx, dtype=torch.long, device=dev),
        torch.tensor(b_list, dtype=torch.long, device=dev),
        torch.tensor(v_list, dtype=torch.long, device=dev),
        torch.tensor(x_list, dtype=torch.long, device=dev),
        torch.tensor(y_list, dtype=torch.long, device=dev),
    )


# ------------------------------- training ------------------------------------

def run_episode(env: TilePlacementEnv, model: nn.Module, alpha_overlap: float) -> Dict[str, Any]:
    """Roll out one batched episode and collect REINFORCE terms.

    Returns dict with keys: logps (list of tensors), values (list), rewards (list),
    done_steps, final_m (B,), total_reward (B,).
    """
    model.train()
    B = env.B
    device = next(model.parameters()).device

    logps = []
    values = []
    rewards = []

    while True:
        # Build step batch and run model
        step = build_step_batch_from_env(env)
        occ = step["occ_crop"].to(device)
        # Move everything else to device as needed
        step["occ_crop"] = occ
        step["placed_mask"] = step["placed_mask"].to(device)
        step["bbox_min"] = step["bbox_min"].to(device)
        step["bbox_max"] = step["bbox_max"].to(device)
        # Candidates live as Tensors already; ensure device
        cands = step["candidates"]
        for name in ("b","v","x","y","from_u","kind","ov_size","feas","d_m","touch_edges","starts"):
            setattr(cands, name, getattr(cands, name).to(device))

        out = model(step, tile_embs=None)
        logits = out["policy_logits"]  # (K,)
        value = out["value"]            # (B,)

        if logits.numel() == 0:
            break  # No candidates anywhere (should coincide with done)

        # Sample one action per active env
        sel_idx, ab, av, ax, ay = sample_actions_per_env(logits, cands, greedy=False)
        if sel_idx.numel() == 0:
            break
        # Log-prob of chosen actions
        logps_step = []
        for i, b in enumerate(ab.tolist()):
            s, e = int(cands.starts[b].item()), int(cands.starts[b+1].item())
            lp = F.log_softmax(logits[s:e], dim=0)
            j = int(sel_idx[i].item()) - s
            logps_step.append(lp[j])
        logps.append(torch.stack(logps_step, dim=0))          # (B_active,)
        values.append(value[ab])                               # (B_active,)

        # Apply actions to env
        stats = env.step(ab, av, ax, ay)
        # Reward shaping: -Δm + alpha*overlap_used
        r = -stats["delta_m"].to(device).to(torch.float32) + alpha_overlap * stats["ov_used"].to(torch.float32)
        rewards.append(r)  # (B_active,)

        # Stop if all done
        if env.done().all():
            break

    # Episode summary per batch env
    bmin, bmax, m = env.bbox_min.to(device), env.bbox_max.to(device), env.m.to(device)
    total_reward = torch.stack(rewards).sum(dim=0) if len(rewards) > 0 else torch.zeros((B,), device=device)

    return dict(
        logps=logps,
        values=values,
        rewards=rewards,
        final_m=m,
        total_reward=total_reward,
    )


def reinforce_loss(traj: Dict[str, Any], gamma: float = 1.0, value_coef: float = 0.5, entropy_coef: float = 0.01) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute REINFORCE with a learned value baseline (advantage actor-critic style).

    Expects variable B_active per step; we treat each step's selections independently.
    """
    logps = traj["logps"]
    values = traj["values"]
    rewards = traj["rewards"]

    # Compute per-step returns (since gamma=1, returns are future sum from t)
    # We aggregate per step independently; shapes vary per step.
    actor_losses = []
    critic_losses = []
    entropies = []

    # For entropy, recompute per-step slice softmax using the cached logits via logps.
    # We don't have logits here; keep entropy term approximate by -mean(log π) over chosen actions.
    # (For exact entropy, pass logits in traj if desired.)

    # Backward accumulate returns per-batch selection sequence length; due to variable B_active, we treat per-step reward as the advantage baseline target.
    for t in range(len(logps)):
        lp = logps[t]          # (B_t,)
        v = values[t]          # (B_t,)
        r = rewards[t]         # (B_t,)
        # One-step return target (REINFORCE can use full return; with gamma=1 and stationary shaping, 1-step still provides a signal.)
        # If you prefer full-trajectory returns, accumulate over future rewards for matching b-indices.
        G = r.detach()
        adv = (G - v.detach())
        actor_losses.append(-(adv * lp).mean())
        critic_losses.append(F.mse_loss(v, G))
        entropies.append(-lp.mean())

    if len(actor_losses) == 0:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, {"actor": 0.0, "critic": 0.0, "entropy": 0.0}

    actor_loss = torch.stack(actor_losses).mean()
    critic_loss = torch.stack(critic_losses).mean()
    entropy = torch.stack(entropies).mean()

    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
    stats = {
        "actor": float(actor_loss.item()),
        "critic": float(critic_loss.item()),
        "entropy": float(entropy.item()),
    }
    return loss, stats


# ------------------------------- main ----------------------------------------

def main(cfg: TrainConfig):
    # Setup logging
    log_level = getattr(logging, cfg.log_level.upper())
    logger = setup_logging(level=log_level)
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    dev = torch.device(cfg.device)
    logger.info(f"Using device: {dev}")
    logger.info(f"Random seed: {cfg.seed}")

    # Build a model; raster_channels = A+1 (symbols + empty)
    raster_channels = cfg.alphabet + 1
    model = build_model(raster_channels, e_tile=cfg.e_tile, d_model=cfg.d_model, d_token=cfg.d_token, dropout=cfg.dropout).to(dev)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger.info(f"Optimizer: AdamW(lr={cfg.lr}, weight_decay={cfg.weight_decay})")

    # Log training configuration
    logger.info("Training Configuration:")
    logger.info(f"  Problem: T={cfg.T}, n={cfg.n}, alphabet={cfg.alphabet}, batch_size={cfg.batch_size}")
    logger.info(f"  Training: epochs={cfg.epochs}, steps_per_epoch={cfg.steps_per_epoch}")
    logger.info(f"  Learning: lr={cfg.lr}, alpha_overlap={cfg.alpha_overlap}")
    logger.info(f"  Loss coefficients: value_coef={cfg.value_coef}, entropy_coef={cfg.entropy_coef}")

    global_step = 0
    start_time = time.time()

    # Training progress bar for epochs
    epoch_pbar = trange(1, cfg.epochs + 1, desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_loss = 0.0
        avg_m = 0.0
        avg_R = 0.0
        
        # Progress bar for episodes within epoch
        episode_pbar = trange(cfg.steps_per_epoch, desc=f"Epoch {epoch}", leave=False, unit="ep")
        
        for epi in episode_pbar:
            # Generate a fresh batch of instances
            tiles = make_synthetic_tiles(
                B=cfg.batch_size, T=cfg.T, n=cfg.n, alphabet_size=cfg.alphabet, hole_prob=cfg.hole_prob, seed=cfg.seed + epoch * 997 + epi, device=dev,
            )
            env = TilePlacementEnv(tiles=tiles, alphabet_size=cfg.alphabet, device=dev)
            env.reset(seed_place_first=True)

            traj = run_episode(env, model, alpha_overlap=cfg.alpha_overlap)
            loss, stats = reinforce_loss(traj, gamma=1.0, value_coef=cfg.value_coef, entropy_coef=cfg.entropy_coef)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            epoch_loss += float(loss.item())
            avg_m += float(traj["final_m"].float().mean().item())
            avg_R += float(traj["total_reward"].float().mean().item())
            global_step += 1
            
            # Update episode progress bar
            episode_pbar.set_postfix({
                'loss': f'{float(loss.item()):.3f}',
                'R': f'{float(traj["total_reward"].float().mean().item()):.2f}',
                'm': f'{float(traj["final_m"].float().mean().item()):.1f}',
            })

        episode_pbar.close()
        
        # Calculate epoch averages
        epoch_loss /= max(1, cfg.steps_per_epoch)
        avg_m /= max(1, cfg.steps_per_epoch)
        avg_R /= max(1, cfg.steps_per_epoch)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Update main progress bar
        epoch_pbar.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'avg_m': f'{avg_m:.3f}',
            'avg_R': f'{avg_R:.3f}',
            'time': f'{elapsed_time:.1f}s'
        })
        
        logger.info(f"Epoch {epoch:03d}: loss={epoch_loss:.4f}, avg_m={avg_m:.3f}, avg_R={avg_R:.3f}, time={elapsed_time:.1f}s")

        if (epoch % cfg.save_every) == 0:
            ckpt = {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "global_step": global_step,
            }
            path = os.path.join(cfg.out_dir, f"model_epoch{epoch:03d}.pt")
            torch.save(ckpt, path)
            logger.info(f"Checkpoint saved to {path}")

    epoch_pbar.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--T", type=int, default=TrainConfig.T)
    p.add_argument("--n", type=int, default=TrainConfig.n)
    p.add_argument("--alphabet", type=int, default=TrainConfig.alphabet)
    p.add_argument("--hole_prob", type=float, default=TrainConfig.hole_prob)
    p.add_argument("--e_tile", type=int, default=TrainConfig.e_tile)
    p.add_argument("--d_model", type=int, default=TrainConfig.d_model)
    p.add_argument("--d_token", type=int, default=TrainConfig.d_token)
    p.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--alpha_overlap", type=float, default=TrainConfig.alpha_overlap)
    p.add_argument("--entropy_coef", type=float, default=TrainConfig.entropy_coef)
    p.add_argument("--value_coef", type=float, default=TrainConfig.value_coef)
    p.add_argument("--max_grad_norm", type=float, default=TrainConfig.max_grad_norm)
    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--steps_per_epoch", type=int, default=TrainConfig.steps_per_epoch)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--save_every", type=int, default=TrainConfig.save_every)
    p.add_argument("--out_dir", type=str, default=TrainConfig.out_dir)
    p.add_argument("--log_level", type=str, default=TrainConfig.log_level, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                   help='Set the logging level')
    args = p.parse_args()

    # Print welcome message
    print("=" * 60)
    print("2DSSP Batched REINFORCE Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Steps per epoch: {args.steps_per_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Problem size: T={args.T}, n={args.n}, alphabet={args.alphabet}")
    print(f"Device: {args.device}")
    print(f"Log level: {args.log_level}")
    print("=" * 60)
    print("Features:")
    print("  ✓ Progress bars with tqdm")
    print("  ✓ Detailed logging to console and train2.log")
    print("  ✓ Real-time metrics display")
    print("  ✓ Automatic checkpointing")
    print("=" * 60)

    cfg = TrainConfig(**vars(args))
    main(cfg)
