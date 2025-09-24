#!/usr/bin/env python3
import argparse, os, json, time, glob
import numpy as np
from pathlib import Path

try:
    from aco import solve_with_aco, ACOParams
except ImportError as e:
    from aco import solve_with_aco, ACOParams
try:
    from greedy_overlap_insertion import greedy_place_once, GreedyParams
except ImportError as e:
    raise

from tile_generators import generate_datasets

def build_gt_canvas_from_tiles(tiles, placements, bbox):
    n = tiles.shape[1]
    xmin, xmax, ymin, ymax = bbox
    W = xmax - xmin + 1; H = ymax - ymin + 1
    canvas = -np.ones((H, W), dtype=int)
    for tid, (x, y) in placements.items():
        T = tiles[tid]
        x0 = x - xmin; y0 = y - ymin
        canvas[y0:y0+n, x0:x0+n] = T
    return canvas

def validate_solution(tiles, placements, canvas):
    if not placements: return False, "No placements"
    if len(placements) != len(tiles): return False, "Missing tiles"
    n = tiles[0].shape[0]
    recon = {}
    for tid,(x,y) in placements.items():
        if tid >= len(tiles): return False, f"Bad tile id {tid}"
        T = tiles[tid]
        for i in range(n):
            for j in range(n):
                coord=(x+i,y+j); v=int(T[i,j])
                if coord in recon and recon[coord] != v: return False, f"Conflict at {coord}"
                recon[coord]=v
    return True, "ok"

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

def run_once(data, ants, iters, K, compaction, seed):
    tiles = data["tiles"]
    bbox = data["bbox"]
    placements = data["placements"]
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
    tiles_list = [tiles[i].copy() for i in range(tiles.shape[0])]

    t0 = time.time()
    aco_res = solve_with_aco(tiles_list, aco_params)
    aco_time = time.time() - t0
    if aco_res["status"] != "ok":
        return None

    m_aco = int(aco_res["best_m"])
    aco_err = ((m_aco - m_gt)/m_gt)*100 if m_gt>0 else 0.0

    g_params = GreedyParams(K=K, rng_seed=seed, perimeter_search_limit=32)
    t1 = time.time()
    g_res = greedy_place_once(tiles_list, g_params)
    g_time = time.time() - t1
    if g_res["status"] != "ok":
        return None
    m_g = int(g_res["best_m"])
    g_err = ((m_g - m_gt)/m_gt)*100 if m_gt>0 else 0.0

    return {
        "m_gt": m_gt, "m_aco": m_aco, "m_greedy": m_g,
        "aco_err": aco_err, "greedy_err": g_err,
        "aco_time": aco_time, "greedy_time": g_time,
    }

def main():
    ap = argparse.ArgumentParser(description="Benchmark ACO/Greedy over multiple datasets with identical parameters.")
    ap.add_argument("--dataset-dir", type=str, default="datasets")
    ap.add_argument("--mode", choices=["random","hidden"], default="random")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--tiles", type=int, default=6)
    ap.add_argument("--alphabet", type=int, default=2)
    ap.add_argument("--canvas-m", type=int, default=None)
    ap.add_argument("--min-overlap", type=int, default=None)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prefix", type=str, default="ds")
    ap.add_argument("--ants", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--compaction", action="store_true")

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

    acc = {"m_gt":0.0, "m_aco":0.0, "m_g":0.0, "aco_err":0.0, "g_err":0.0, "aco_time":0.0, "g_time":0.0}
    ok=0
    for p in ds_paths:
        data = load_dataset(p)
        r = run_once(data, ants=args.ants, iters=args.iters, K=args.K, compaction=args.compaction, seed=args.seed)
        if r is None:
            print(f"[skip] {p}")
            continue
        ok += 1
        acc["m_gt"] += r["m_gt"]; acc["m_aco"] += r["m_aco"]; acc["m_g"] += r["m_greedy"]
        acc["aco_err"] += r["aco_err"]; acc["g_err"] += r["greedy_err"]
        acc["aco_time"] += r["aco_time"]; acc["g_time"] += r["greedy_time"]

    if ok==0:
        print(json.dumps({"status":"fail"})); return

    avg = {k: acc[k]/ok for k in acc}
    report = {
        "datasets_used": ok,
        "params": {"mode": args.mode, "n": args.n, "tiles": args.tiles, "alphabet": args.alphabet,
                   "canvas_m": args.canvas_m, "min_overlap": args.min_overlap,
                   "ants": args.ants, "iters": args.iters, "K": args.K, "compaction": bool(args.compaction)},
        "averages": {"m_gt": avg["m_gt"], "m_aco": avg["m_aco"], "m_greedy": avg["m_g"],
                     "aco_err_%": avg["aco_err"], "greedy_err_%": avg["g_err"],
                     "aco_time_s": avg["aco_time"], "greedy_time_s": avg["g_time"]}
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
