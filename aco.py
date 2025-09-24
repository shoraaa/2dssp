
# aco_opt.py
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import random, math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

_G_TILES = None
_G_N = None
_G_T = None
_G_PRE_TOPK = None
_G_IDMAP = None
_G_OV = None

def _init_worker(tiles, n, T, pre_topk, idmap, ov_arr):
    global _G_TILES, _G_N, _G_T, _G_PRE_TOPK, _G_IDMAP, _G_OV
    _G_TILES = tiles
    _G_N = n
    _G_T = T
    _G_PRE_TOPK = pre_topk
    _G_IDMAP = idmap
    _G_OV = ov_arr

Coordinate = Tuple[int, int]
Placement = Dict[int, Coordinate]

@dataclass
class ACOParams:
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

def compute_pairwise_topk(tiles: List[np.ndarray], n: int, K: int):
    pre_topk: Dict[Tuple[int,int], List[Tuple[int,int,int]]] = {}
    T = len(tiles)
    for u in range(T):
        for v in range(T):
            if u == v: 
                continue
            entries: List[Tuple[int,int,int]] = []
            A = tiles[u]; B = tiles[v]
            for dx in range(-(n-1), n):
                ai0 = 0 if dx<=0 else dx
                ai1 = n if dx>=0 else n+dx
                if ai0>=ai1: 
                    continue
                for dy in range(-(n-1), n):
                    aj0 = 0 if dy<=0 else dy
                    aj1 = n if dy>=0 else n+dy
                    if aj0>=aj1: 
                        continue
                    bi0 = ai0 - dx; bj0 = aj0 - dy
                    bi1 = ai1 - dx; bj1 = aj1 - dy
                    Aov = A[ai0:ai1, aj0:aj1]
                    Bov = B[bi0:bi1, bj0:bj1]
                    if np.any(Aov != Bov): 
                        continue
                    ov = Aov.size
                    if ov>0:
                        entries.append((dx,dy,int(ov)))
            if entries:
                random.shuffle(entries)
                entries.sort(key=lambda e: e[2], reverse=True)
                pre_topk[(u,v)] = entries[:K]
            else:
                pre_topk[(u,v)] = []
    return pre_topk

def _idmap_from_pre(pre_topk: Dict[Tuple[int,int], List[Tuple[int,int,int]]]):
    idmap: Dict[Tuple[int,int,int,int], int] = {}
    ov_vals: List[int] = []
    for (u,v), lst in pre_topk.items():
        for (dx,dy,ov) in lst:
            key=(u,v,dx,dy)
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

def _build_ant_solution_worker(seed:int, tau_snapshot: np.ndarray):
    tiles = _G_TILES; n=_G_N; T=_G_T
    pre_topk = _G_PRE_TOPK; idmap=_G_IDMAP; ov_arr=_G_OV

    random.seed(seed); np.random.seed(seed)
    placements: Dict[int,Tuple[int,int]] = {0:(0,0)}
    placed_ids=[0]
    occ: Dict[Tuple[int,int],int]={}
    write_to_occupancy(tiles[0],0,0,occ)
    bbox=[0,n-1,0,n-1]
    parent_edges: List[Tuple[int,int,int,int]] = []

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

    def perimeter_candidates(v:int):
        xmin,xmax,ymin,ymax = bbox
        limit=4
        cands=[]
        for pad in range(0,limit):
            y_top = ymin - n - pad
            y_bot = ymax + 1 + pad
            for x in range(xmin - n - pad, xmax + 1 + pad):
                cands.append((x,y_top)); cands.append((x,y_bot))
            x_left = xmin - n - pad
            x_right = xmax + 1 + pad
            for y in range(ymin - n - pad + 1, ymax + 1 + pad - 1):
                cands.append((x_left,y)); cands.append((x_right,y))
            if cands: break
        return cands

    while len(placed_ids) < T:
        candidates=[]
        proposed = {}
        for v in range(T):
            if v in placements: continue
            ps = proposed.setdefault(v, set())
            for u in placed_ids:
                ux,uy = placements[u]
                for (dx,dy,_) in pre_topk.get((u,v), []):
                    ps.add((ux+dx, uy+dy))
            if not ps:
                for (x,y) in perimeter_candidates(v):
                    ps.add((x,y))

        for v, posset in proposed.items():
            for (x,y) in posset:
                if not feasible_on_occupancy(tiles[v], x, y, occ):
                    continue
                H,Tsum = hp_sums(v,x,y)
                inc = bbox_increase_if_place(x,y,n,bbox)
                if H<=0 and inc>0:
                    continue
                s = ((Tsum if Tsum>0 else 1e-12)**1.0) * ((H+1.0)**3.0)
                candidates.append((v,x,y,H,Tsum,s))
        if not candidates:
            return None

        S = 0.0
        for *_, s in candidates: S += s
        r = random.random() * S
        acc=0.0; chosen=None
        for c in candidates:
            acc += c[-1]
            if acc>=r:
                chosen=c; break
        v,x,y,H,Tsum,_ = chosen

        best_parent=None; best_ov=-1
        for u in placed_ids:
            ux,uy = placements[u]
            dx=x-ux; dy=y-uy
            key = idmap.get((u,v,dx,dy))
            if key is not None:
                ov = ov_arr[key]
                if ov>best_ov:
                    best_ov=ov; best_parent=(u,dx,dy)
        if best_parent is None:
            u=random.choice(placed_ids); ux,uy=placements[u]
            best_parent=(u, x-ux, y-uy)

        placements[v]=(x,y); placed_ids.append(v)
        write_to_occupancy(tiles[v], x, y, occ)
        update_bbox_for_tile(x,y,n,bbox)
        ustar,dx,dy = best_parent
        parent_edges.append((ustar,v,dx,dy))

    m,_ = layout_bbox(placements, n)
    return placements, m, parent_edges

def solve_with_aco(tiles: List[np.ndarray], params: ACOParams) -> Dict[str,Any]:
    if params.random_seed is not None:
        random.seed(params.random_seed); np.random.seed(params.random_seed)
    T = len(tiles); assert T>=1
    n = tiles[0].shape[0]

    pre_topk = compute_pairwise_topk(tiles, n, params.K)
    idmap, ov_arr = _idmap_from_pre(pre_topk)
    M = len(ov_arr)
    tau = np.ones(M, dtype=np.float64)

    best_layout=None; best_m=10**9

    with ProcessPoolExecutor(max_workers=params.n_workers, initializer=_init_worker,
                             initargs=(tiles, n, T, pre_topk, idmap, ov_arr)) as pool:
        for it in range(params.iterations):
            tau_snapshot = tau.copy()
            seeds = [ (params.random_seed or 0) + it*131071 + k*104729 for k in range(params.n_ants) ]
            results = list(pool.map(_build_ant_solution_worker, seeds, [tau_snapshot]*params.n_ants))

            tau *= (1.0 - params.rho)

            valid = [r for r in results if r is not None]
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

    if best_layout is None:
        return {"status":"failed","reason":"No layout constructed","best_m":None}
    m,bbox = layout_bbox(best_layout, n)
    occ={}
    for tid,(x,y) in best_layout.items():
        write_to_occupancy(tiles[tid], x, y, occ)
    xmin,xmax,ymin,ymax = bbox
    W=xmax-xmin+1; H=ymax-ymin+1
    canvas = np.full((H,W), -1, dtype=int)
    for (X,Y),val in occ.items():
        canvas[Y - ymin, X - xmin] = val
    return {"status":"ok","best_m":m,"bbox":bbox,"placements":best_layout,"canvas":canvas}
