#!/usr/bin/env python3
import argparse, os, time, glob, csv
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pathlib import Path


from aco import solve_with_aco, ACOParams

try:
    from greedy import greedy_place_once, GreedyParams
except ImportError as e:
    raise

from generator import generate_datasets

import torch
import torch.nn.functional as F

from env import (
    TilePlacementEnv,
    precompute_tile_embeddings,
    build_step_batch_from_env,
    layout_bbox,
)
from neural_solver import NeuralSolver, TileCNN

def load_dataset(npz_path: str):
    d = np.load(npz_path, allow_pickle=False)
    tiles = d["tiles"]
    bbox = d["bbox"].astype(int)
    plc_items = d["placements"].astype(int)
    placements = {int(t): (int(x), int(y)) for t,x,y in plc_items}
    n = int(d["n"][0])
    alphabet = int(d["alphabet"][0])
    gt_time = float(d.get("gt_time", [0.0])[0]) if "gt_time" in d else 0.0
    gt_source = str(d.get("gt_source", ["unknown"])[0]) if "gt_source" in d else "unknown"
    optimal_m = int(d.get("optimal_m", [0])[0]) if "optimal_m" in d else None
    return {
        "tiles": tiles, "bbox": bbox, "placements": placements, "n": n,
        "alphabet": alphabet, "gt_time": gt_time, "gt_source": gt_source, "optimal_m": optimal_m
    }


@dataclass
class NeuralBenchmarkContext:
    model: NeuralSolver
    tile_cnn: TileCNN
    device: torch.device
    temperature: float
    greedy: bool
    max_steps: int


def _maybe_extract_state_dict(artifact, candidate_keys):
    for key in candidate_keys:
        state = artifact.get(key) if isinstance(artifact, dict) else None
        if isinstance(state, dict) and state:
            return state
    return artifact if isinstance(artifact, dict) else None


def load_neural_context(
    alphabet: int,
    checkpoint: str,
    device: torch.device,
    d_tile: int,
    d_model: int,
    cand_feat_dim: int,
    temperature: float,
    greedy: bool,
    max_steps: int,
    tilecnn_checkpoint: Optional[str] = None,
) -> NeuralBenchmarkContext:
    ckpt = torch.load(checkpoint, map_location=device)

    model = NeuralSolver(
        c_occ=alphabet + 1,
        d_tile=d_tile,
        d_model=d_model,
        cand_feat_dim=cand_feat_dim,
    ).to(device)

    model_state = _maybe_extract_state_dict(ckpt, ["model", "state_dict"])
    if model_state is None:
        raise ValueError(f"Checkpoint {checkpoint} does not contain model weights")
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[warn] missing model keys: {sorted(missing)}")
    if unexpected:
        print(f"[warn] unexpected model keys: {sorted(unexpected)}")
    model.eval()

    tile_cnn = TileCNN(in_ch=alphabet, d_tile=d_tile).to(device)
    tile_state = None
    if tilecnn_checkpoint:
        tc = torch.load(tilecnn_checkpoint, map_location=device)
        tile_state = _maybe_extract_state_dict(tc, ["tile_cnn", "tilecnn", "model", "state_dict"])
    else:
        tile_state = _maybe_extract_state_dict(ckpt, ["tile_cnn", "tilecnn"])

    if isinstance(tile_state, dict):
        missing_tc, unexpected_tc = tile_cnn.load_state_dict(tile_state, strict=False)
        if missing_tc:
            print(f"[warn] missing tile_cnn keys: {sorted(missing_tc)}")
        if unexpected_tc:
            print(f"[warn] unexpected tile_cnn keys: {sorted(unexpected_tc)}")
    else:
        print("[warn] tile CNN weights not found; using random initialization")
    tile_cnn.eval()

    return NeuralBenchmarkContext(
        model=model,
        tile_cnn=tile_cnn,
        device=device,
        temperature=temperature,
        greedy=greedy,
        max_steps=max_steps,
    )


@torch.no_grad()
def run_neural_solver(neural: NeuralBenchmarkContext, tiles, alphabet: int):
    env = TilePlacementEnv(tiles, alphabet=alphabet)
    tile_embs = precompute_tile_embeddings(
        env.tiles,
        env.alphabet,
        neural.tile_cnn,
        neural.device,
        keep_on_device=neural.device.type == "cuda",
    )
    steps = 0
    status = "ok"

    while not env.done and steps < neural.max_steps:
        sb = build_step_batch_from_env(env, tile_embs, device=neural.device)
        if not sb.cand_mask.any().item():
            status = "no_moves"
            break

        logits, _ = neural.model(
            sb.occ,
            sb.tiles_left,
            sb.tiles_left_mask,
            sb.cand_feats,
            sb.cand_mask,
            sb.cand_tile_idx,
        )
        logits = logits / max(1e-6, neural.temperature)
        logits = logits.masked_fill(~sb.cand_mask, -1e9)

        if neural.greedy:
            action = int(logits.argmax(dim=-1).item())
        else:
            probs = F.softmax(logits, dim=-1)
            probs = probs.masked_fill(~sb.cand_mask, 0.0)
            norm = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            probs = probs / norm
            action = int(torch.distributions.Categorical(probs.squeeze(0)).sample().item())

        raw_cands = env.generate_candidates()
        if action >= len(raw_cands):
            status = "invalid_action"
            break
        env.step(raw_cands[action])
        steps += 1

    if not env.done and status == "ok":
        status = "max_steps"

    best_m, _ = layout_bbox(env.placements, env.n)
    return {
        "status": status,
        "best_m": int(best_m),
        "placements": dict(env.placements),
        "steps": steps,
    }

def run_once(data, ants, iters, K, compaction, seed, neural: Optional[NeuralBenchmarkContext] = None):
    tiles = data["tiles"]
    bbox = data["bbox"]
    n = data["n"]

    xmin, xmax, ymin, ymax = [int(x) for x in bbox]
    m_gt = max(xmax - xmin + 1, ymax - ymin + 1)

    aco_params = ACOParams(
        K=K, alpha=1.0, beta=3.0, gamma=1.0,
        lam=0.05, epsilon=1.0, rho=0.10,
        Q=float(n*n), n_ants=ants, iterations=iters,
        random_seed=seed, perimeter_search_limit=32,
        enable_compaction=compaction
    )
    tiles_for_aco = [tiles[i].copy() for i in range(tiles.shape[0])]

    t0 = time.time()
    aco_res = solve_with_aco(tiles_for_aco, aco_params)
    aco_time = time.time() - t0
    if aco_res["status"] != "ok":
        return None

    m_aco = int(aco_res["best_m"])
    aco_err = ((m_aco - m_gt)/m_gt)*100 if m_gt>0 else 0.0

    g_params = GreedyParams(K=K, rng_seed=seed, perimeter_search_limit=32)
    t1 = time.time()
    tiles_for_greedy = [tiles[i].copy() for i in range(tiles.shape[0])]
    g_res = greedy_place_once(tiles_for_greedy, g_params)
    g_time = time.time() - t1
    if g_res["status"] != "ok":
        return None
    m_g = int(g_res["best_m"])
    g_err = ((m_g - m_gt)/m_gt)*100 if m_gt>0 else 0.0

    neural_status = "disabled" if neural is None else "skipped"
    m_neural = None
    neural_err = None
    neural_time = None
    neural_steps = None

    if neural is not None:
        tiles_for_neural = [tiles[i].copy() for i in range(tiles.shape[0])]
        t2 = time.time()
        neural_res = run_neural_solver(neural, tiles_for_neural, data["alphabet"])
        neural_time = time.time() - t2
        neural_status = neural_res.get("status", "unknown")
        if neural_res.get("best_m") is not None:
            m_neural = int(neural_res["best_m"])
            neural_err = ((m_neural - m_gt)/m_gt)*100 if m_gt>0 else 0.0
        neural_steps = neural_res.get("steps")

    return {
        "m_gt": m_gt, "m_aco": m_aco, "m_greedy": m_g,
        "aco_err": aco_err, "greedy_err": g_err,
        "aco_time": aco_time, "greedy_time": g_time,
        "m_neural": m_neural, "neural_err": neural_err,
        "neural_time": neural_time, "neural_steps": neural_steps,
        "neural_status": neural_status,
    }

def main():
    ap = argparse.ArgumentParser(description="Benchmark ACO, Greedy, and Neural solvers over multiple datasets with identical parameters.")
    ap.add_argument("--dataset-dir", type=str, default="datasets")
    ap.add_argument("--mode", choices=["random","hidden","synthetic"], default="random")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--tiles", type=int, default=6)
    ap.add_argument("--alphabet", type=int, default=2)
    ap.add_argument("--canvas-m", type=int, default=None)
    ap.add_argument("--min-overlap", type=int, default=None)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prefix", type=str, default="ds")
    ap.add_argument("--ants", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--compaction", action="store_true")
    ap.add_argument("--neural-checkpoint", type=str, default=None, help="Path to NeuralSolver checkpoint (.pt)")
    ap.add_argument("--tilecnn-checkpoint", type=str, default=None, help="Optional TileCNN checkpoint path")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device for neural solver")
    ap.add_argument("--neural-d-tile", type=int, default=64)
    ap.add_argument("--neural-d-model", type=int, default=128)
    ap.add_argument("--neural-cand-dim", type=int, default=10)
    ap.add_argument("--neural-temperature", type=float, default=0.7)
    ap.add_argument("--neural-max-steps", type=int, default=10000)
    ap.add_argument("--neural-sample", action="store_true", help="Sample neural policy instead of greedy argmax")
    ap.add_argument("--csv-out", type=str, default=None, help="Where to write CSV results (defaults to dataset_dir)")

    args = ap.parse_args()

    Path(args.dataset_dir).mkdir(parents=True, exist_ok=True)
    pat = f"{args.prefix}_{args.mode}_n{args.n}_T{args.tiles}_A{args.alphabet}_seed*.npz"
    existing = sorted(glob.glob(str(Path(args.dataset_dir) / pat)))
    to_make = max(0, args.count - len(existing))
    if to_make > 0:
        new_paths = generate_datasets(
            out_dir=args.dataset_dir, mode=args.mode, n=args.n, num_tiles=args.tiles,
            alphabet_size=args.alphabet, canvas_m=args.canvas_m, min_overlap=args.min_overlap,
            count=to_make, base_seed=args.seed + len(existing), prefix=args.prefix
        )
        existing.extend(new_paths)

    ds_paths = sorted(existing)[:args.count]
    print(f"[run] evaluating {len(ds_paths)} datasets")

    neural_ctx = None
    if args.neural_checkpoint:
        device = torch.device(args.device)
        try:
            neural_ctx = load_neural_context(
                alphabet=args.alphabet,
                checkpoint=args.neural_checkpoint,
                device=device,
                d_tile=args.neural_d_tile,
                d_model=args.neural_d_model,
                cand_feat_dim=args.neural_cand_dim,
                temperature=args.neural_temperature,
                greedy=not args.neural_sample,
                max_steps=args.neural_max_steps,
                tilecnn_checkpoint=args.tilecnn_checkpoint,
            )
            print(f"[neural] loaded checkpoint {args.neural_checkpoint} on {device}")
        except Exception as exc:
            print(f"[neural] failed to load checkpoint: {exc}")
            return
    else:
        print("[neural] no checkpoint provided; skipping neural benchmark")

    fieldnames = [
        "dataset",
        "m_gt",
        "m_aco",
        "m_greedy",
        "m_neural",
        "aco_err_pct",
        "greedy_err_pct",
        "neural_err_pct",
        "aco_time_s",
        "greedy_time_s",
        "neural_time_s",
        "neural_steps",
        "neural_status",
    ]

    numeric_columns = [
        "m_gt",
        "m_aco",
        "m_greedy",
        "m_neural",
        "aco_err_pct",
        "greedy_err_pct",
        "neural_err_pct",
        "aco_time_s",
        "greedy_time_s",
        "neural_time_s",
        "neural_steps",
    ]

    acc = {col: 0.0 for col in numeric_columns}
    counts = {col: 0 for col in numeric_columns}
    rows = []
    ok = 0
    for p in ds_paths:
        data = load_dataset(p)
        try:
            r = run_once(
                data,
                ants=args.ants,
                iters=args.iters,
                K=args.K,
                compaction=args.compaction,
                seed=args.seed,
                neural=neural_ctx,
            )
        except Exception as exc:
            print(f"[error] {p}: {exc}")
            continue
        if r is None:
            print(f"[skip] {p}")
            continue
        ok += 1
        row = {
            "dataset": os.path.basename(p),
            "m_gt": r["m_gt"],
            "m_aco": r["m_aco"],
            "m_greedy": r["m_greedy"],
            "m_neural": r["m_neural"],
            "aco_err_pct": r["aco_err"],
            "greedy_err_pct": r["greedy_err"],
            "neural_err_pct": r["neural_err"],
            "aco_time_s": r["aco_time"],
            "greedy_time_s": r["greedy_time"],
            "neural_time_s": r["neural_time"],
            "neural_steps": r["neural_steps"],
            "neural_status": r["neural_status"],
        }
        rows.append(row)

        for col in numeric_columns:
            val = row.get(col)
            if val is not None:
                acc[col] += float(val)
                counts[col] += 1

        if neural_ctx is None:
            print(
                f"[{row['dataset']}] GT: {row['m_gt']}, ACO: {row['m_aco']} (err {row['aco_err_pct']:.2f}%), "
                f"Greedy: {row['m_greedy']} (err {row['greedy_err_pct']:.2f}%)"
            )
        else:
            if row["m_neural"] is not None:
                neural_str = f"Neural: {row['m_neural']} (err {row['neural_err_pct']:.2f}%, status {row['neural_status']})"
            else:
                neural_str = f"Neural: n/a (status {row['neural_status']})"
            print(
                f"[{row['dataset']}] GT: {row['m_gt']}, "
                f"ACO: {row['m_aco']} (err {row['aco_err_pct']:.2f}%), "
                f"Greedy: {row['m_greedy']} (err {row['greedy_err_pct']:.2f}%), "
                f"{neural_str}"
            )

    if ok == 0:
        print("[fail] no successful runs; CSV not generated")
        return

    avg_row = {"dataset": "__average__", "neural_status": ""}
    for col in numeric_columns:
        if counts[col] > 0:
            avg_row[col] = acc[col] / counts[col]
        else:
            avg_row[col] = None

    if args.csv_out:
        csv_path = Path(args.csv_out)
    else:
        default_name = f"benchmark_{args.mode}_n{args.n}_T{args.tiles}_A{args.alphabet}.csv"
        csv_path = Path(args.dataset_dir) / default_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
        writer.writerow({k: ("" if avg_row.get(k) is None else avg_row.get(k)) for k in fieldnames})

    print(f"[done] wrote benchmark results to {csv_path}")
    print(
        f"[avg] ACO m={avg_row['m_aco']:.2f}, Greedy m={avg_row['m_greedy']:.2f}" +
        (
            f", Neural m={avg_row['m_neural']:.2f} (status aggregated)" if neural_ctx and avg_row.get("m_neural") is not None else ""
        )
    )

if __name__ == "__main__":
    main()
