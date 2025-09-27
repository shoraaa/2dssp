# train.py
# Usage:
#   python train.py --instances 64 --steps 5000 --T 16 --n 6 --alphabet 6 --batch_size 8 --device cuda
#

from __future__ import annotations
import os
import time
import math
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Local modules from previous replies
from neural_solver import NeuralSolver, TileCNN
from env import (
    TilePlacementEnv, StepBatch,
    make_synthetic_tiles, precompute_tile_embeddings,
    build_step_batch_from_env, choose_expert_action, layout_bbox
)

# ----------------------------
# Collation utilities
# ----------------------------

def pad_step_batches(batches: List[StepBatch]) -> StepBatch:
    """
    Collate a list of StepBatch (possibly with different M/A sizes) into one batch by padding.
    Returns a (B)-batched StepBatch with:
      - occ: stacked (pad H/W to max via center padding)
      - tiles_left, tiles_left_mask: pad M to max_M
      - cand_feats, cand_mask, cand_tile_idx: pad A to max_A
      - expert_action: stacked (if any is None -> returns None)
    """
    B = len(batches)
    # 1) OCC crop: (C, H, W) vary in H/W; we center-pad to max
    C = batches[0].occ.size(1)
    maxH = max(b.occ.size(2) for b in batches)
    maxW = max(b.occ.size(3) for b in batches)
    occ_out = []
    for b in batches:
        _, Cb, Hb, Wb = b.occ.shape
        pad_h = maxH - Hb
        pad_w = maxW - Wb
        # center pad: (left, right, top, bottom)
        l = pad_w // 2
        r = pad_w - l
        t = pad_h // 2
        bt = pad_h - t
        occ_out.append(F.pad(b.occ, (l, r, t, bt)))  # (1,C,maxH,maxW)
    occ = torch.cat(occ_out, dim=0)  # (B, C, maxH, maxW)

    # 2) tiles_left
    maxM = max(b.tiles_left.size(1) for b in batches)
    d_tile = batches[0].tiles_left.size(-1)
    tiles_left = torch.zeros(B, maxM, d_tile, dtype=batches[0].tiles_left.dtype)
    tiles_left_mask = torch.zeros(B, maxM, dtype=torch.bool)
    for i, b in enumerate(batches):
        M = b.tiles_left.size(1)
        tiles_left[i, :M] = b.tiles_left
        tiles_left_mask[i, :M] = b.tiles_left_mask

    # 3) candidates
    maxA = max(b.cand_feats.size(1) for b in batches)
    Fdim = batches[0].cand_feats.size(-1)
    cand_feats = torch.zeros(B, maxA, Fdim, dtype=batches[0].cand_feats.dtype)
    cand_mask = torch.zeros(B, maxA, dtype=torch.bool)
    cand_tile_idx = torch.zeros(B, maxA, dtype=torch.long)
    for i, b in enumerate(batches):
        A = b.cand_feats.size(1)
        cand_feats[i, :A] = b.cand_feats
        cand_mask[i, :A] = b.cand_mask
        cand_tile_idx[i, :A] = b.cand_tile_idx

    # 4) expert actions (if any missing -> return None)
    if any(b.expert_action is None for b in batches):
        expert_action = None
    else:
        expert_action = torch.cat([b.expert_action for b in batches], dim=0)  # (B,)

    return StepBatch(
        occ=occ,
        tiles_left=tiles_left,
        tiles_left_mask=tiles_left_mask,
        cand_feats=cand_feats,
        cand_mask=cand_mask,
        cand_tile_idx=cand_tile_idx,
        expert_action=expert_action
    )

# ----------------------------
# Instance manager for batching
# ----------------------------

class InstancePool:
    """
    Holds multiple parallel environments (synthetic) and their tile embeddings.
    Each call to `next_batch()` returns a collated StepBatch with expert labels,
    and advances every env by applying the model (or expert) action from previous step.
    """
    def __init__(self, num_envs: int, T: int, n: int, alphabet: int, device: torch.device,
                 tile_cnn: TileCNN):
        self.num_envs = num_envs
        self.T = T
        self.n = n
        self.alphabet = alphabet
        self.device = device
        self.tile_cnn = tile_cnn

        self.envs: List[TilePlacementEnv] = []
        self.tile_embs_list: List[torch.Tensor] = []
        self.reset_all()

    def reset_all(self):
        self.envs.clear()
        self.tile_embs_list.clear()
        for _ in range(self.num_envs):
            tiles = make_synthetic_tiles(T=self.T, n=self.n, alphabet=self.alphabet, seed=None)
            env = TilePlacementEnv(tiles, alphabet=self.alphabet)
            self.envs.append(env)
            tile_embs = precompute_tile_embeddings(tiles, self.alphabet, self.tile_cnn, self.device)
            self.tile_embs_list.append(tile_embs)

    def next_batch(self) -> StepBatch:
        # If any env is done, reset it (keeps training perpetual)
        for i, env in enumerate(self.envs):
            if env.done:
                tiles = make_synthetic_tiles(T=self.T, n=self.n, alphabet=self.alphabet, seed=None)
                self.envs[i] = TilePlacementEnv(tiles, alphabet=self.alphabet)
                self.tile_embs_list[i] = precompute_tile_embeddings(tiles, self.alphabet, self.tile_cnn, self.device)

        # Build step batches and add expert labels
        step_batches: List[StepBatch] = []
        raw_cands_per_env: List[list] = []  # keep to apply actions later if needed

        for i, env in enumerate(self.envs):
            sb = build_step_batch_from_env(env, self.tile_embs_list[i])
            # If no candidates (rare), force reset and build again
            if sb.cand_mask.sum().item() == 0:
                tiles = make_synthetic_tiles(T=self.T, n=self.n, alphabet=self.alphabet, seed=None)
                self.envs[i] = TilePlacementEnv(tiles, alphabet=self.alphabet)
                self.tile_embs_list[i] = precompute_tile_embeddings(tiles, self.alphabet, self.tile_cnn, self.device)
                sb = build_step_batch_from_env(self.envs[i], self.tile_embs_list[i])

            # Assign greedy expert action (for IL)
            act = choose_expert_action(sb)
            sb.expert_action = act
            step_batches.append(sb)

        return pad_step_batches(step_batches)

    def apply_actions(self, action_indices: torch.Tensor):
        """
        Applies chosen action indices (B,) to each env.
        `action_indices` are indices in the *padded* candidate list; we map back
        using each env's current raw candidates at time of `build_step_batch_from_env`.
        For simplicity we recompute candidates now (same state) and index them.
        """
        assert action_indices.numel() == self.num_envs
        for i, env in enumerate(self.envs):
            raw_cands = env.generate_candidates()
            # Safety: clip action idx if padding made it point past actual #cands
            a = int(action_indices[i].item())
            if a >= len(raw_cands):
                a = max(0, len(raw_cands) - 1)
            env.step(raw_cands[a])


# ----------------------------
# Training loop
# ----------------------------

def train_imitation(
    model: NeuralSolver,
    pool: InstancePool,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    ckpt_dir: str,
    device: torch.device,
    use_wandb: bool = True,
    wandb_config: dict = None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    if use_wandb and wandb_config:
        wandb.init(**wandb_config)
        wandb.watch(model, log="gradients", log_freq=log_every)
    
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    model.train()
    ema_loss = None
    t0 = time.time()

    for it in range(1, steps + 1):
        # Build a mega-batch by concatenating several collated batches
        # (We reuse the pool in chunks to reach desired batch_size)
        step_batches = []
        envs_per_micro = pool.num_envs
        micro_batches = math.ceil(batch_size / envs_per_micro)

        for _ in range(micro_batches):
            sb = pool.next_batch()
            step_batches.append(sb)

        # Merge micro-batches (pad across them too)
        mega = pad_step_batches(step_batches)

        occ = mega.occ.to(device)
        tiles_left = mega.tiles_left.to(device)
        tiles_left_mask = mega.tiles_left_mask.to(device)
        cand_feats = mega.cand_feats.to(device)
        cand_mask = mega.cand_mask.to(device)
        cand_tile_idx = mega.cand_tile_idx.to(device)
        expert_action = mega.expert_action.to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits, _ = model(occ, tiles_left, tiles_left_mask, cand_feats, cand_mask, cand_tile_idx)
            # masked log-softmax
            very_neg = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(~cand_mask, very_neg)
            logp = F.log_softmax(masked_logits, dim=-1)
            loss = F.nll_loss(logp, expert_action, reduction='mean')

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()

        # Apply model actions to advance each env once (greedy argmax)
        with torch.no_grad():
            probs = F.softmax(masked_logits, dim=-1)
            actions = probs.argmax(dim=-1)
        # Split actions back to pool micros and apply (use first micro only to keep step sync)
        # Alternatively, you can apply actions after each micro call; here we just
        # apply the first micro's actions for simplicity.
        first_micro_actions = actions[:pool.num_envs]
        pool.apply_actions(first_micro_actions.cpu())

        # Logging
        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val
        
        if it % log_every == 0:
            # Probe first env for current m
            m, _ = layout_bbox(pool.envs[0].placements, pool.envs[0].n)
            dt = time.time() - t0
            print(f"[it {it:05d}] loss={loss_val:.4f} ema={ema_loss:.4f} m(env0)={m}  dt={dt:.1f}s")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    "loss": loss_val,
                    "ema_loss": ema_loss,
                    "canvas_size_env0": m,
                    "step": it,
                    "time_per_step": dt / log_every,
                    "learning_rate": optim.param_groups[0]['lr'],
                }, step=it)
            
            t0 = time.time()  # Reset timer after logging

        # Save ckpt
        if it % max(1000, log_every*10) == 0:
            path = os.path.join(ckpt_dir, f"ckpt_{it:06d}.pt")
            torch.save({
                "it": it,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            }, path)
            
            # Log checkpoint as wandb artifact
            if use_wandb:
                artifact = wandb.Artifact(f"checkpoint-step-{it}", type="model")
                artifact.add_file(path)
                wandb.log_artifact(artifact)

    # final save
    path = os.path.join(ckpt_dir, f"final.pt")
    torch.save({"model": model.state_dict()}, path)
    print(f"Saved final checkpoint to {path}")
    
    # Log final checkpoint as wandb artifact
    if use_wandb:
        artifact = wandb.Artifact("final-checkpoint", type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.finish()


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", type=int, default=8, help="parallel environments")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--T", type=int, default=12, help="tiles per instance")
    p.add_argument("--n", type=int, default=6, help="tile size n×n")
    p.add_argument("--alphabet", type=int, default=6, help="symbol classes")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Wandb arguments
    p.add_argument("--wandb_project", type=str, default="2dssp", help="wandb project name")
    p.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    p.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    args = p.parse_args()

    device = torch.device(args.device)

    # Prepare wandb configuration
    use_wandb = not args.no_wandb
    wandb_config = None
    if use_wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": {
                "instances": args.instances,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "T": args.T,
                "n": args.n,
                "alphabet": args.alphabet,
                "lr": args.lr,
                "log_every": args.log_every,
                "device": args.device,
                "c_occ": args.alphabet + 1,
                "d_tile": 64,
                "d_model": 128,
                "cand_feat_dim": 10,
            }
        }

    # Model hyperparams (must match env feature dims)
    c_occ = args.alphabet + 1
    d_tile = 64
    d_model = 128
    cand_feat_dim = 10  # matches env_generator.build_step_batch_from_env

    # Init models
    tile_cnn = TileCNN(in_ch=args.alphabet, d_tile=d_tile).to(device)
    model = NeuralSolver(c_occ=c_occ, d_tile=d_tile, d_model=d_model, cand_feat_dim=cand_feat_dim).to(device)

    # Instance pool
    pool = InstancePool(
        num_envs=args.instances,
        T=args.T,
        n=args.n,
        alphabet=args.alphabet,
        device=device,
        tile_cnn=tile_cnn,
    )

    # Train (imitation on greedy expert)
    train_imitation(
        model=model,
        pool=pool,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        ckpt_dir=args.ckpt_dir,
        device=device,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
    )

if __name__ == "__main__":
    main()
