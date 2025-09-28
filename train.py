#!/usr/bin/env python3
# train.py — Batched REINFORCE with fixed train/val datasets.

import os
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import trange
import wandb

# ---- Import your modules ----
from net import NeuralSolver, TileCNN
from env import (
    TilePlacementEnv,
    bbox_increase_if_place,
    make_synthetic_tiles,
    precompute_tile_embeddings,
    build_step_batch_from_env,
    layout_bbox,
)

def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    very_neg = torch.finfo(logits.dtype).min
    return F.log_softmax(logits.masked_fill(~mask.bool(), very_neg), dim=dim)

@torch.no_grad()
def compute_delta_m(env, x, y):
    return float(bbox_increase_if_place(x, y, env.n, env.bbox))

@torch.no_grad()
def model_pick_action(model: NeuralSolver, sb, temperature: float = 1.0, argmax: bool = True) -> int:
    logits, _ = model(
        sb.occ,
        sb.tiles_left,
        sb.tiles_left_mask,
        sb.cand_feats,
        sb.cand_mask,
        sb.cand_tile_idx,
    )
    logits = logits / max(1e-6, temperature)
    logits = logits.masked_fill(~sb.cand_mask, -1e9)
    if argmax:
        return int(logits.argmax(dim=-1).item())
    probs = F.softmax(logits, dim=-1).squeeze(0)
    probs[~sb.cand_mask.squeeze(0)] = 0.0
    probs = probs / (probs.sum() + 1e-9)
    return int(torch.distributions.Categorical(probs).sample().item())

@torch.no_grad()
def evaluate_instance(model: NeuralSolver, env: TilePlacementEnv, tile_embs: torch.Tensor, device, temperature=0.7, greedy=True):
    dev = torch.device(device)
    if tile_embs.device != dev:
        tile_embs = tile_embs.to(dev, non_blocking=True)

    env.reset()
    steps = 0
    while not env.done and steps < 10000:
        sb = build_step_batch_from_env(env, tile_embs, device=dev)
        if not bool(sb.cand_mask.any().detach().cpu()):
            break
        a = model_pick_action(model, sb, temperature=temperature, argmax=greedy)
        raw_cands = env.generate_candidates()
        env.step(raw_cands[a])
        steps += 1
    xmin, xmax, ymin, ymax = env.bbox
    m = max(xmax - xmin + 1, ymax - ymin + 1)
    return float(m), steps


# ----------------------------
# Dataset builders
# ----------------------------
def make_dataset(size: int, T: int, n: int, alphabet: int, seed: int) -> Tuple[List[TilePlacementEnv], List[torch.Tensor]]:
    envs = []
    for i in range(size):
        tiles = make_synthetic_tiles(T=T, n=n, alphabet=alphabet, seed=seed + i)
        envs.append(TilePlacementEnv(tiles, alphabet=alphabet))
    # NOTE: tile_cnn-dependent embeddings are filled later once tile_cnn is constructed.
    return envs

def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)

def rollout_episode(
    model: NeuralSolver,
    env: TilePlacementEnv,
    tile_embs: torch.Tensor,
    device,
    temperature: float,
    max_steps: int,
    overlap_bonus_coef: float = 0.0,
    amp_enabled: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, int, float]:
    """
    Reward: r_t = (- delta_m) + overlap_bonus_coef * H_sum(action_t)
    """
    model.train()  # keep dropout/etc.; fine for policy gradient
    env.reset()
    logps: List[torch.Tensor] = []
    rewards: List[float] = []
    entropy_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    steps = 0

    while not env.done and steps < max_steps:
        sb = build_step_batch_from_env(env, tile_embs, device=device)
        # avoid device sync with .item()
        if not bool(sb.cand_mask.any().detach().cpu()):
            break

        with autocast(enabled=amp_enabled):
            logits, _ = model(
                sb.occ,
                sb.tiles_left,
                sb.tiles_left_mask,
                sb.cand_feats,
                sb.cand_mask,
                sb.cand_tile_idx,
            )
        masked_logits = logits / max(1e-6, temperature)
        masked_logits = masked_logits.masked_fill(~sb.cand_mask, -1e9)
        logp_all = F.log_softmax(masked_logits, dim=-1).squeeze(0)
        p_all = logp_all.exp()

        entropy_sum_t = entropy_sum_t - (p_all * logp_all).sum()

        probs = p_all.clone()
        probs[~sb.cand_mask.squeeze(0)] = 0.0
        probs = probs / (probs.sum() + 1e-9)
        a_t = torch.distributions.Categorical(probs).sample()  # tensor on device
        logps.append(logp_all[a_t])

        raw_cands = env.generate_candidates()
        a = int(a_t)  # a single, cheap sync
        _, delta_m, H_sum = env.step_and_metrics(raw_cands[a])

        r = float(-delta_m) + overlap_bonus_coef * float(H_sum)
        rewards.append(r)
        steps += 1

    # final m directly from bbox
    xmin, xmax, ymin, ymax = env.bbox
    final_m = float(max(xmax - xmin + 1, ymax - ymin + 1))
    return (torch.stack(logps) if logps else torch.empty(0, device=device),
            torch.tensor(rewards, dtype=torch.float32),
            final_m,
            steps,
            float(entropy_sum_t.detach().cpu()))


def train_reinforce(
    model: NeuralSolver,
    tile_cnn: nn.Module,
    batch_size: int,
    T: int,
    n: int,
    alphabet: int,
    epochs: int,
    gamma: float,
    lr: float,
    device,
    log_every: int,
    eval_every: int,
    save_dir: str,
    temperature: float,
    max_steps_per_episode: int,
    overlap_bonus_coef: float,
    train_data_size: int,
    val_data_size: int,
    baseline_ema_coef: float = 0.9,
    resume_ckpt: Optional[str] = None,
    entropy_coef: float = 0.01,
    use_wandb: bool = True,
    data_seed: int = 1234,
):
    os.makedirs(save_dir, exist_ok=True)
    opt = AdamW(model.parameters(), lr=lr)

    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    if resume_ckpt and os.path.isfile(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        print(f"Resumed from {resume_ckpt}")

    # ----------------------------
    # Build datasets
    # ----------------------------
    train_envs = make_dataset(train_data_size, T, n, alphabet, seed=data_seed)
    val_envs   = make_dataset(val_data_size,   T, n, alphabet, seed=data_seed + 10_000)

    # Precompute tile embeddings once (TileCNN is not optimized here)
    keep_embeddings_on_device = device.type == "cuda"
    train_tile_embs = [
        precompute_tile_embeddings(e.tiles, e.alphabet, tile_cnn, device, keep_on_device=keep_embeddings_on_device)
        for e in train_envs
    ]
    val_tile_embs   = [
        precompute_tile_embeddings(e.tiles, e.alphabet, tile_cnn, device, keep_on_device=keep_embeddings_on_device)
        for e in val_envs
    ]

    ema_return = None
    tbar = trange(1, epochs + 1, desc="train[REINFORCE]", ncols=100)
    for ep in tbar:
        model.train()
        opt.zero_grad(set_to_none=True)

        # ---- Sample a mini-batch of train episodes (with replacement) ----
        idxs = np.random.randint(low=0, high=train_data_size, size=batch_size)
        epi_data = []
        batch_returns, batch_steps, batch_m = [], [], []
        total_entropy_sum = 0.0

        for idx in idxs:
            env = train_envs[idx]
            tile_embs = train_tile_embs[idx]
            logps, rewards, final_m, steps, ent_sum = rollout_episode(
                model=model,
                env=env,
                tile_embs=tile_embs,
                device=device,
                temperature=temperature,
                max_steps=max_steps_per_episode,
                overlap_bonus_coef=overlap_bonus_coef,
                amp_enabled=scaler.is_enabled(),
            )
            batch_steps.append(steps)
            batch_m.append(final_m)
            total_entropy_sum += ent_sum

            if logps.numel() == 0:
                continue
            G = compute_returns(rewards.tolist(), gamma).to(device)
            batch_returns.append(G[0].item())
            epi_data.append({"logps": logps, "G": G})

        if len(epi_data) == 0:
            # all dead; skip update but still proceed to possible eval/log
            continue

        batch_avg_return = float(np.mean(batch_returns)) if batch_returns else 0.0
        ema_return = batch_avg_return if ema_return is None else (
            baseline_ema_coef * ema_return + (1 - baseline_ema_coef) * batch_avg_return
        )

        policy_loss_total = torch.zeros((), device=device)
        all_advs = []
        for ed in epi_data:
            G = ed["G"]
            baseline = torch.full_like(G, float(ema_return), device=device)
            adv = G - baseline
            all_advs.append(adv.detach().cpu())
            policy_loss_total = policy_loss_total - (ed["logps"] * adv.detach()).sum()

        loss = policy_loss_total - entropy_coef * torch.tensor(total_entropy_sum, dtype=torch.float32, device=device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        mean_steps = float(np.mean(batch_steps)) if batch_steps else 0.0
        mean_m = float(np.mean(batch_m)) if batch_m else 0.0
        mean_entropy = total_entropy_sum / max(1, int(sum(batch_steps)))
        loss_scalar = loss.detach().item()
        tbar.set_postfix_str(f"R0={batch_avg_return:.3f} m={mean_m:.0f} steps={mean_steps:.1f} loss={loss_scalar:.3f}")

        if use_wandb:
            adv_cat = torch.cat(all_advs) if len(all_advs) else torch.tensor([0.0])
            total_steps = max(1, int(sum(batch_steps)))
            wandb.log({
                "train/return": batch_avg_return,
                "train/m": mean_m,
                "train/policy_loss": policy_loss_total.detach().item() / total_steps,
                "train/loss": loss_scalar / total_steps,
                "train/entropy": mean_entropy,
                "train/baseline_ema": ema_return,
                "train/advantage": float(adv_cat.mean().item()),
                "epoch": ep,
            })

        # ---- Validation ----
        if ep % eval_every == 0:
            model.eval()
            ms, stepss = [], []
            for venv, vemb in zip(val_envs, val_tile_embs):
                m, used_steps = evaluate_instance(model, venv, vemb, device, temperature=0.7, greedy=True)
                ms.append(m)
                stepss.append(used_steps)
            val_m_mean = float(np.mean(ms)) if ms else 0.0
            val_steps_mean = float(np.mean(stepss)) if stepss else 0.0
            print(f"\n[val @ epoch {ep}] m_mean={val_m_mean:.2f} steps_mean={val_steps_mean:.1f} (N={len(ms)})")
            if use_wandb:
                wandb.log({
                    "val/m": val_m_mean,
                    "epoch": ep
                })

        if ep % log_every == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_reinforce_ep{ep}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, ckpt_path)

    final_path = os.path.join(save_dir, "final_reinforce.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, final_path)
    print(f"Saved final REINFORCE checkpoint to {final_path}")

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Train neural tile solver (REINFORCE, batched over fixed datasets)")
    p.add_argument("--alphabet", type=int, default=2)
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8, help="episodes sampled per epoch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    # model dims
    p.add_argument("--d_tile", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--cand_feat_dim", type=int, default=10)
    # training schedule + datasets
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max_steps_per_episode", type=int, default=2000)
    p.add_argument("--overlap_bonus_coef", type=float, default=0.0)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--train_data_size", type=int, default=2048)
    p.add_argument("--val_data_size", type=int, default=256)
    p.add_argument("--data_seed", type=int, default=1234, help="base seed for dataset generation")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    if args.wandb:
        wandb.init(
            project="neural-2dssp",
            config=vars(args),
            name=f"reinforce_T{args.T}_n{args.n}_a{args.alphabet}_train{args.train_data_size}_val{args.val_data_size}"
        )

    tile_cnn = TileCNN(in_ch=args.alphabet, d_tile=args.d_tile).to(device)
    model = NeuralSolver(
        c_occ=args.alphabet + 1,
        d_tile=args.d_tile,
        d_model=args.d_model,
        cand_feat_dim=args.cand_feat_dim
    ).to(device)

    train_reinforce(
        model=model,
        tile_cnn=tile_cnn,
        batch_size=args.batch_size,
        T=args.T,
        n=args.n,
        alphabet=args.alphabet,
        epochs=args.epochs,
        gamma=args.gamma,
        lr=args.lr,
        device=device,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        temperature=args.temperature,
        max_steps_per_episode=args.max_steps_per_episode,
        overlap_bonus_coef=args.overlap_bonus_coef,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        resume_ckpt=args.resume,
        entropy_coef=args.entropy_coef,
        use_wandb=args.wandb,
        data_seed=args.data_seed,
    )

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
