# aco_opt.py
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import random, math
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import wandb

_G_TILES = None
_G_N = None
_G_T = None
_G_PRE_ALL = None   # renamed: all overlaps (no TopK)
_G_IDMAP = None
_G_OV = None

def _init_worker(tiles, n, T, pre_all, idmap, ov_arr):
    global _G_TILES, _G_N, _G_T, _G_PRE_ALL, _G_IDMAP, _G_OV
    _G_TILES = tiles
    _G_N = n
    _G_T = T
    _G_PRE_ALL = pre_all
    _G_IDMAP = idmap
    _G_OV = ov_arr

Coordinate = Tuple[int, int]
Placement = Dict[int, Coordinate]

@dataclass
class ACOParams:
    # K removed from usage; kept here only to preserve constructor compatibility (ignored)
    K: int = 16
    alpha: float = 1.0
    beta: float = 3.0
    gamma: float = 1.0
    lam: float = 0.02
    epsilon: float = 1.0
    rho: float = 0.10
    Q: float = 25.0
    n_ants: int = 16
    iterations: int = 100
    random_seed: Optional[int] = 42
    perimeter_search_limit: int = 64
    enable_compaction: bool = False
    n_workers: int = 8

class TopKEntry:
    __slots__ = ("dx","dy","ov")
    def __init__(self, dx:int, dy:int, ov:int):
        self.dx=dx; self.dy=dy; self.ov=ov

# ---------- NEW: compute all overlaps (no TopK) ----------
def compute_pairwise_overlaps(tiles: List[np.ndarray], n: int):
    """Return, for every ordered pair (u,v), the full list of (dx,dy,ov) where
    the two n×n tiles exactly agree on their overlapping region when v is
    placed at (ux+dx, uy+dy) relative to u.
    """
    print(f"[ACO Performance] Starting overlap precomputation for {len(tiles)} tiles...")
    start_time = time.time()
    
    pre_all: Dict[Tuple[int,int], List[Tuple[int,int,int]]] = {}
    T = len(tiles)
    
    overlap_checks = 0
    valid_overlaps = 0
    
    for u in range(T):
        A = tiles[u]
        for v in range(T):
            if u == v:
                pre_all[(u,v)] = []
                continue
            B = tiles[v]
            entries: List[Tuple[int,int,int]] = []
            for dx in range(-(n-1), n):
                ai0 = 0 if dx <= 0 else dx
                ai1 = n if dx >= 0 else n + dx
                if ai0 >= ai1:
                    continue
                for dy in range(-(n-1), n):
                    aj0 = 0 if dy <= 0 else dy
                    aj1 = n if dy >= 0 else n + dy
                    if aj0 >= aj1:
                        continue
                    bi0 = ai0 - dx; bj0 = aj0 - dy
                    bi1 = ai1 - dx; bj1 = aj1 - dy
                    Aov = A[ai0:ai1, aj0:aj1]
                    Bov = B[bi0:bi1, bj0:bj1]
                    # exact agreement on overlap
                    overlap_checks += 1
                    if np.any(Aov != Bov):
                        continue
                    ov = Aov.size
                    if ov > 0:
                        entries.append((dx, dy, int(ov)))
                        valid_overlaps += 1
            # stable order: larger overlaps first (not required, but nice)
            if entries:
                entries.sort(key=lambda e: e[2], reverse=True)
            pre_all[(u,v)] = entries
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Log overlap precomputation metrics to wandb
    wandb.log({
        "overlap_precomp/time": total_time,
        "overlap_precomp/overlap_checks": overlap_checks,
        "overlap_precomp/valid_overlaps": valid_overlaps,
        "overlap_precomp/efficiency_checks_per_sec": overlap_checks / total_time if total_time > 0 else 0,
        "overlap_precomp/validity_rate": valid_overlaps / overlap_checks if overlap_checks > 0 else 0
    })
    
    print(f"[ACO Performance] Overlap precomputation completed in {total_time:.3f}s")
    print(f"[ACO Performance] - Overlap checks performed: {overlap_checks:,}")
    print(f"[ACO Performance] - Valid overlaps found: {valid_overlaps:,}")
    print(f"[ACO Performance] - Efficiency: {overlap_checks/total_time:.0f} checks/sec")
    
    return pre_all

def _idmap_from_pre(pre_all: Dict[Tuple[int,int], List[Tuple[int,int,int]]]):
    idmap: Dict[Tuple[int,int,int,int], int] = {}
    ov_vals: List[int] = []
    for (u,v), lst in pre_all.items():
        for (dx,dy,ov) in lst:
            key = (u,v,dx,dy)
            if key not in idmap:
                idmap[key] = len(ov_vals)
                ov_vals.append(ov)
    return idmap, np.array(ov_vals, dtype=np.int32)

def feasible_on_occupancy(tile: np.ndarray, x: int, y: int, occ: Dict[Tuple[int,int],int]) -> bool:
    n = tile.shape[0]
    for i in range(n):
        xi = x + i
        row = tile[i]
        for j in range(n):
            coord = (xi, y+j)
            val = int(row[j])
            ov = occ.get(coord)
            if ov is not None and ov != val:
                return False
    return True

def write_to_occupancy(tile: np.ndarray, x: int, y: int, occ: Dict[Tuple[int,int],int]) -> None:
    n = tile.shape[0]
    for i in range(n):
        xi = x + i
        row = tile[i]
        for j in range(n):
            occ[(xi, y+j)] = int(row[j])

def update_bbox_for_tile(x:int, y:int, n:int, bbox: List[int]) -> None:
    xmin,xmax,ymin,ymax = bbox
    if x < xmin: xmin = x
    if y < ymin: ymin = y
    tx = x+n-1; ty = y+n-1
    if tx > xmax: xmax = tx
    if ty > ymax: ymax = ty
    bbox[0],bbox[1],bbox[2],bbox[3] = xmin,xmax,ymin,ymax

def bbox_increase_if_place(x:int,y:int,n:int,bbox:List[int]) -> int:
    xmin,xmax,ymin,ymax = bbox
    nxmin = min(xmin, x); nymin = min(ymin, y)
    nxmax = max(xmax, x+n-1); nymax = max(ymax, y+n-1)
    old_m = max(xmax-xmin+1, ymax-ymin+1)
    new_m = max(nxmax-nxmin+1, nymax-nymin+1)
    return max(0, new_m-old_m)

def compaction_left_up(placements: Dict[int,Tuple[int,int]], tiles: List[np.ndarray]) -> Dict[int,Tuple[int,int]]:
    ids = list(placements.keys())
    random.shuffle(ids)
    def build_occ(pl):
        occ = {}
        for tid,(x,y) in pl.items():
            write_to_occupancy(tiles[tid], x, y, occ)
        return occ
    changed = True
    while changed:
        changed = False
        for tid in ids:
            occ = build_occ({k:v for k,v in placements.items() if k!=tid})
            x,y = placements[tid]
            moved = True
            while moved and feasible_on_occupancy(tiles[tid], x-1, y, occ):
                x-=1; moved=True; changed=True
            moved=True
            while moved and feasible_on_occupancy(tiles[tid], x, y-1, occ):
                y-=1; moved=True; changed=True
            placements[tid]=(x,y)
    return placements

def layout_bbox(placements: Dict[int,Tuple[int,int]], n:int):
    xmin=ymin=10**9; xmax=ymax=-10**9
    for (x,y) in placements.values():
        if x < xmin: xmin=x
        if y < ymin: ymin=y
        tx=x+n-1; ty=y+n-1
        if tx > xmax: xmax=tx
        if ty > ymax: ymax=ty
    m = max(xmax-xmin+1, ymax-ymin+1)
    return m, [xmin,xmax,ymin,ymax]

# ---------- helper: adjacency offsets (non-overlapping) ----------
def _adjacent_offsets(n: int) -> List[Tuple[int,int]]:
    """Return 4n+4 offsets placing a second n×n tile adjacent (no overlap) to the first.
    We use ranges that produce (n+1) positions per side:
      - Left/Right:  dx = -n or +n, dy in [-n, 0]
      - Up/Down:     dy = -n or +n, dx in [-n, 0]
    """
    offs = []
    for dy in range(-n, 1):        # n+1 values
        offs.append((-n, dy))      # left
        offs.append((+n, dy))      # right
    for dx in range(-n, 1):        # n+1 values
        offs.append((dx, -n))      # up
        offs.append((dx, +n))      # down
    return offs  # length = 4*(n+1) = 4n+4

def _build_ant_solution_worker(seed:int, tau_snapshot: np.ndarray):
    tiles = _G_TILES; n=_G_N; T=_G_T
    pre_all = _G_PRE_ALL; idmap=_G_IDMAP; ov_arr=_G_OV

    random.seed(seed); np.random.seed(seed)
    placements: Dict[int,Tuple[int,int]] = {0:(0,0)}
    placed_ids=[0]
    occ: Dict[Tuple[int,int],int]={}
    write_to_occupancy(tiles[0],0,0,occ)
    bbox=[0,n-1,0,n-1]
    parent_edges: List[Tuple[int,int,int,int]] = []

    adj_offs = _adjacent_offsets(n)

    def hp_sums(v:int, x:int, y:int):
        H=0.0; Tsum=0.0
        for u in placed_ids:
            ux,uy = placements[u]
            dx=x-ux; dy=y-uy
            key = idmap.get((u,v,dx,dy))
            if key is not None:
                H += ov_arr[key]
                Tsum += tau_snapshot[key]
        return H, Tsum

    while len(placed_ids) < T:
        candidates = []

        for v in range(T):
            if v in placements:
                continue

            # (A) try all exact overlaps first
            feasible_overlap_found = False
            overlap_positions = set()
            for u in placed_ids:
                ux, uy = placements[u]
                for (dx, dy, _) in pre_all.get((u, v), []):
                    overlap_positions.add((ux + dx, uy + dy))

            for (x, y) in overlap_positions:
                if not feasible_on_occupancy(tiles[v], x, y, occ):
                    continue
                H, Tsum = hp_sums(v, x, y)
                inc = bbox_increase_if_place(x, y, n, bbox)
                # guard removed by request
                s = ((Tsum if Tsum > 0 else 1e-12)) * ((H + 1.0) ** 3.0)
                candidates.append((v, x, y, H, Tsum, s))
                feasible_overlap_found = True

            # (B) if no overlap, use only adjacency (4n+4 per placed u)
            if not feasible_overlap_found:
                adj_positions = set()
                for u in placed_ids:
                    ux, uy = placements[u]
                    for (dx, dy) in adj_offs:   # precomputed via _adjacent_offsets(n)
                        adj_positions.add((ux + dx, uy + dy))

                for (x, y) in adj_positions:
                    if not feasible_on_occupancy(tiles[v], x, y, occ):
                        continue
                    H, Tsum = hp_sums(v, x, y)
                    inc = bbox_increase_if_place(x, y, n, bbox)
                    s = ((Tsum if Tsum > 0 else 1e-12)) * ((H + 1.0) ** 3.0)
                    candidates.append((v, x, y, H, Tsum, s))

        assert candidates, "No candidates found; should not happen"

        # roulette select
        S = sum(c[-1] for c in candidates)
        r = random.random() * S
        acc = 0.0
        for c in candidates:
            acc += c[-1]
            if acc >= r:
                v, x, y, H, Tsum, _ = c
                break

        # parent with max overlap (if any)
        best_parent, best_ov = None, -1
        for u in placed_ids:
            ux, uy = placements[u]
            dx, dy = x - ux, y - uy
            key = idmap.get((u, v, dx, dy))
            if key is not None:
                ov = ov_arr[key]
                if ov > best_ov:
                    best_ov = ov
                    best_parent = (u, dx, dy)
        if best_parent is None:
            u = random.choice(placed_ids); ux, uy = placements[u]
            best_parent = (u, x - ux, y - uy)

        placements[v] = (x, y)
        placed_ids.append(v)
        write_to_occupancy(tiles[v], x, y, occ)
        update_bbox_for_tile(x, y, n, bbox)
        ustar, dx, dy = best_parent
        parent_edges.append((ustar, v, dx, dy))

    m, _ = layout_bbox(placements, n)
    return placements, m, parent_edges

def solve_with_aco(tiles: List[np.ndarray], params: ACOParams) -> Dict[str,Any]:
    # Initialize wandb run
    wandb.init(
        project="aco-2dssp",
        config={
            "iterations": params.iterations,
            "n_ants": params.n_ants,
            "n_workers": params.n_workers,
            "alpha": params.alpha,
            "beta": params.beta,
            "gamma": params.gamma,
            "lambda": params.lam,
            "epsilon": params.epsilon,
            "rho": params.rho,
            "Q": params.Q,
            "random_seed": params.random_seed,
            "n_tiles": len(tiles),
            "tile_size": tiles[0].shape[0] if tiles else None,
            "enable_compaction": params.enable_compaction
        },
        tags=["aco", "2dssp", "optimization"]
    )
    
    print(f"[ACO Performance] Starting ACO with {params.iterations} iterations, {params.n_ants} ants, {params.n_workers} workers")
    total_start_time = time.time()
    
    if params.random_seed is not None:
        random.seed(params.random_seed); np.random.seed(params.random_seed)
    T = len(tiles); assert T>=1
    n = tiles[0].shape[0]

    # PRECOMPUTE: all overlaps (no TopK)
    precomp_start = time.time()
    pre_all = compute_pairwise_overlaps(tiles, n)
    idmap, ov_arr = _idmap_from_pre(pre_all)
    M = len(ov_arr)
    tau = np.ones(M, dtype=np.float64)
    precomp_time = time.time() - precomp_start

    # Log preprocessing metrics
    wandb.log({
        "preprocessing/time": precomp_time,
        "preprocessing/pheromone_matrix_size": M,
        "preprocessing/total_overlaps": len([ov for overlaps in pre_all.values() for ov in overlaps]),
        "preprocessing/avg_overlaps_per_tile_pair": len([ov for overlaps in pre_all.values() for ov in overlaps]) / (T * T) if T > 0 else 0
    })
    
    print(f"[ACO Performance] Preprocessing completed in {precomp_time:.3f}s")
    print(f"[ACO Performance] - Pheromone matrix size: {M:,} edges")

    best_layout=None; best_m=10**9
    
    # Performance tracking variables
    iteration_times = []
    ant_solution_times = []
    pheromone_update_times = []
    best_improvements = []

    with ProcessPoolExecutor(max_workers=params.n_workers, initializer=_init_worker,
                             initargs=(tiles, n, T, pre_all, idmap, ov_arr)) as pool:
        for it in tqdm(range(params.iterations), desc="ACO Iterations", unit="iter"):
            iter_start_time = time.time()
            
            tau_snapshot = tau.copy()
            seeds = [ (params.random_seed or 0) + it*131071 + k*104729 for k in range(params.n_ants) ]
            
            # Time ant solution building
            ant_start_time = time.time()
            results = list(pool.map(_build_ant_solution_worker, seeds, [tau_snapshot]*params.n_ants))
            ant_time = time.time() - ant_start_time
            ant_solution_times.append(ant_time)
            
            # Time pheromone update
            pheromone_start_time = time.time()
            
            tau *= (1.0 - params.rho)

            valid = [r for r in results if r is not None]
            iter_best_m = float('inf')
            iter_best_layout = None
            
            if valid:
                valid.sort(key=lambda r: r[1])
                iter_best_layout, iter_best_m, iter_best_edges = valid[0]

                for (placements, m, parent_edges) in valid:
                    depo = params.Q / max(1,m)
                    for (u,v,dx,dy) in parent_edges:
                        idx = idmap.get((u,v,dx,dy))
                        if idx is not None:
                            tau[idx] += depo
                    if m < best_m:
                        best_m=m; best_layout=dict(placements)
                        best_improvements.append((it, m, time.time() - total_start_time))

                # small all-pairs reinforcement within the iteration's best layout
                depo = (params.Q / max(1, iter_best_m)) * 0.25
                ids = list(iter_best_layout.keys())
                for i in range(len(ids)):
                    u = ids[i]
                    for j in range(len(ids)):
                        if i==j: continue
                        v = ids[j]
                        dx = iter_best_layout[v][0] - iter_best_layout[u][0]
                        dy = iter_best_layout[v][1] - iter_best_layout[u][1]
                        idx = idmap.get((u,v,dx,dy))
                        if idx is not None:
                            tau[idx] += depo
            
            pheromone_time = time.time() - pheromone_start_time
            pheromone_update_times.append(pheromone_time)
            
            iter_time = time.time() - iter_start_time
            iteration_times.append(iter_time)
            
            # Log iteration metrics to wandb
            valid_ants = len(valid)
            wandb.log({
                "iteration": it,
                "timing/iteration_time": iter_time,
                "timing/ant_solution_time": ant_time,
                "timing/pheromone_update_time": pheromone_time,
                "ants/valid_ants": valid_ants,
                "ants/valid_ant_ratio": valid_ants / params.n_ants,
                "solution/iter_best_m": iter_best_m if iter_best_m != float('inf') else None,
                "solution/global_best_m": best_m if best_m != 10**9 else None,
                "pheromone/avg_tau": float(np.mean(tau)),
                "pheromone/max_tau": float(np.max(tau)),
                "pheromone/min_tau": float(np.min(tau))
            })
            
            # Log progress every 10 iterations or if significant improvement
            if (it + 1) % 10 == 0 or (valid and iter_best_m < best_m * 1.1):
                avg_iter_time = sum(iteration_times[-10:]) / min(10, len(iteration_times))
                print(f"[ACO Performance] Iter {it+1:3d}: {valid_ants:2d}/{params.n_ants} valid, "
                      f"best_m={best_m}, iter_best={iter_best_m if valid else 'N/A'}, "
                      f"time={iter_time:.3f}s (avg={avg_iter_time:.3f}s)")

    total_time = time.time() - total_start_time
    
    # Log comprehensive performance summary to wandb
    solving_time = sum(iteration_times) if iteration_times else 0
    total_ant_time = sum(ant_solution_times) if ant_solution_times else 0
    total_pheromone_time = sum(pheromone_update_times) if pheromone_update_times else 0
    
    summary_metrics = {
        "summary/total_time": total_time,
        "summary/preprocessing_time": precomp_time,
        "summary/preprocessing_percentage": (precomp_time / total_time * 100) if total_time > 0 else 0,
        "summary/solving_time": solving_time,
        "summary/solving_percentage": (solving_time / total_time * 100) if total_time > 0 else 0,
        "summary/ant_solution_time": total_ant_time,
        "summary/ant_solution_percentage": (total_ant_time / total_time * 100) if total_time > 0 else 0,
        "summary/pheromone_update_time": total_pheromone_time,
        "summary/pheromone_update_percentage": (total_pheromone_time / total_time * 100) if total_time > 0 else 0,
        "summary/final_best_m": best_m if best_m != 10**9 else None,
        "summary/num_improvements": len(best_improvements)
    }
    
    if iteration_times:
        summary_metrics.update({
            "summary/avg_iteration_time": sum(iteration_times) / len(iteration_times),
            "summary/min_iteration_time": min(iteration_times),
            "summary/max_iteration_time": max(iteration_times)
        })
    
    if ant_solution_times:
        summary_metrics.update({
            "summary/avg_ant_time_per_iter": sum(ant_solution_times) / len(ant_solution_times)
        })
    
    # Log improvement timeline
    for i, (iter_num, m_value, timestamp) in enumerate(best_improvements):
        wandb.log({
            f"improvements/improvement_{i}_iteration": iter_num,
            f"improvements/improvement_{i}_m_value": m_value,
            f"improvements/improvement_{i}_timestamp": timestamp
        })
    
    wandb.log(summary_metrics)
    
    # Print console summary (reduced)
    print(f"\n[ACO Performance] === PERFORMANCE SUMMARY ===")
    print(f"[ACO Performance] Total solve time: {total_time:.3f}s")
    print(f"[ACO Performance] Final best solution: m={best_m}")
    print(f"[ACO Performance] Logged {len(best_improvements)} improvements to wandb")
    print(f"[ACO Performance] ========================\n")

    if best_layout is None:
        # Log failure and finish run
        wandb.log({"summary/status": "failed"})
        wandb.finish()
        return {"status":"failed","reason":"No layout constructed","best_m":None}
    
    # Log success metrics
    m,bbox = layout_bbox(best_layout, n)
    occ={}
    for tid,(x,y) in best_layout.items():
        write_to_occupancy(tiles[tid], x, y, occ)
    xmin,xmax,ymin,ymax = bbox
    W=xmax-xmin+1; H=ymax-ymin+1
    canvas = np.full((H,W), -1, dtype=int)
    for (X,Y),val in occ.items():
        canvas[Y - ymin, X - xmin] = val
    
    wandb.log({
        "summary/status": "success",
        "summary/final_canvas_width": W,
        "summary/final_canvas_height": H,
        "summary/canvas_area": W * H
    })
    
    # Finish wandb run
    wandb.finish()
    
    return {"status":"ok","best_m":m,"bbox":bbox,"placements":best_layout,"canvas":canvas}
