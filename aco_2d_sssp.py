# aco_2d_sssp.py  (with adjacent 3x3 candidate positions relative to placed tiles)
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import random, math
from collections import defaultdict

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
    n_ants: int = 12
    iterations: int = 200
    random_seed: Optional[int] = 42
    perimeter_search_limit: int = 64
    enable_compaction: bool = False  # off by default

class TopKEntry:
    __slots__ = ("dx","dy","ov")
    def __init__(self, dx:int, dy:int, ov:int):
        self.dx=dx; self.dy=dy; self.ov=ov

class Precompute:
    def __init__(self, n:int):
        self.topk: Dict[Tuple[int,int], List[TopKEntry]] = {}
        self.n = n

def compute_pairwise_topk(tiles: List[np.ndarray], n: int, K: int) -> Precompute:
    P = Precompute(n=n)
    T = len(tiles)
    for u in range(T):
        for v in range(T):
            if u == v: 
                continue
            entries: List[TopKEntry] = []
            A = tiles[u]; B = tiles[v]
            for dx in range(-(n-1), n):
                for dy in range(-(n-1), n):
                    ai0 = 0 if dx<=0 else dx
                    aj0 = 0 if dy<=0 else dy
                    ai1 = n if dx>=0 else n+dx
                    aj1 = n if dy>=0 else n+dy
                    if ai0>=ai1 or aj0>=aj1: 
                        continue
                    bi0 = ai0 - dx; bj0 = aj0 - dy
                    bi1 = ai1 - dx; bj1 = aj1 - dy
                    Aov = A[ai0:ai1, aj0:aj1]
                    Bov = B[bi0:bi1, bj0:bj1]
                    if np.any(Aov != Bov): 
                        continue
                    ov = Aov.size
                    if ov>0:
                        entries.append(TopKEntry(dx,dy,int(ov)))
            if entries:
                random.shuffle(entries)
                entries.sort(key=lambda e: e.ov, reverse=True)
                P.topk[(u,v)] = entries[:K]
            else:
                P.topk[(u,v)] = []
    return P

def feasible_on_occupancy(tile: np.ndarray, x: int, y: int, occ: Dict[Coordinate,int]) -> bool:
    n = tile.shape[0]
    for i in range(n):
        for j in range(n):
            val = int(tile[i,j])
            coord = (x+i, y+j)
            if coord in occ and occ[coord] != val:
                return False
    return True

def write_to_occupancy(tile: np.ndarray, x: int, y: int, occ: Dict[Coordinate,int]) -> None:
    n = tile.shape[0]
    for i in range(n):
        for j in range(n):
            occ[(x+i, y+j)] = int(tile[i,j])

def update_bbox_for_tile(x:int, y:int, n:int, bbox: List[int]) -> None:
    xmin,xmax,ymin,ymax = bbox
    xmin = min(xmin, x); ymin = min(ymin, y)
    xmax = max(xmax, x+n-1); ymax = max(ymax, y+n-1)
    bbox[0],bbox[1],bbox[2],bbox[3] = xmin,xmax,ymin,ymax

def bbox_increase_if_place(x:int,y:int,n:int,bbox:List[int]) -> int:
    xmin,xmax,ymin,ymax = bbox
    nxmin = min(xmin, x); nymin = min(ymin, y)
    nxmax = max(xmax, x+n-1); nymax = max(ymax, y+n-1)
    old_m = max(xmax-xmin+1, ymax-ymin+1)
    new_m = max(nxmax-nxmin+1, nymax-nymin+1)
    return max(0, new_m-old_m)

def compaction_left_up(placements: Placement, tiles: List[np.ndarray]) -> Placement:
    ids = list(placements.keys())
    random.shuffle(ids)
    def build_occ(pl: Placement):
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
            moved=True
            while moved:
                moved=False
                if feasible_on_occupancy(tiles[tid], x-1, y, occ):
                    x-=1; moved=True; changed=True
            moved=True
            while moved:
                moved=False
                if feasible_on_occupancy(tiles[tid], x, y-1, occ):
                    y-=1; moved=True; changed=True
            placements[tid]=(x,y)
    return placements

def layout_bbox(placements: Placement, n:int):
    xmin=ymin=10**9; xmax=ymax=-10**9
    for (x,y) in placements.values():
        xmin=min(xmin,x); ymin=min(ymin,y)
        xmax=max(xmax,x+n-1); ymax=max(ymax,y+n-1)
    m = max(xmax-xmin+1, ymax-ymin+1)
    return m, [xmin,xmax,ymin,ymax]

def solve_with_aco(tiles: List[np.ndarray], params: ACOParams) -> Dict[str,Any]:
    if params.random_seed is not None:
        random.seed(params.random_seed); np.random.seed(params.random_seed)
    T = len(tiles); assert T>=1
    n = tiles[0].shape[0]
    pre = compute_pairwise_topk(tiles, n, params.K)
    tau: Dict[Tuple[int,int,int,int], float] = {}
    for (u,v), lst in pre.topk.items():
        for e in lst:
            tau[(u,v,e.dx,e.dy)] = 1.0
    ov_lookup = {(u,v,e.dx,e.dy): e.ov for (u,v),lst in pre.topk.items() for e in lst}

    def hp_sums(v:int, x:int, y:int, placed_ids: List[int], placements: Placement):
        H=0.0; Tsum=0.0
        for u in placed_ids:
            ux,uy = placements[u]
            dx=x-ux; dy=y-uy
            ov = ov_lookup.get((u,v,dx,dy), 0)
            if ov:
                H += ov
                Tsum += tau.get((u,v,dx,dy), 0.0)
        return H, Tsum

    def perimeter_candidates(v:int, occ:Dict[Coordinate,int], bbox:List[int]):
        n = tiles[v].shape[0]
        xmin,xmax,ymin,ymax = bbox
        cands=[]; limit=params.perimeter_search_limit
        for pad in range(0,limit):
            ring=[]
            y_top = ymin - n - pad
            for x in range(xmin - n - pad, xmax + 1 + pad):
                ring.append((x,y_top))
            x_left = xmin - n - pad
            for y in range(ymin - n - pad + 1, ymax + 1 + pad):
                ring.append((x_left,y))
            y_bot = ymax + 1 + pad
            for x in range(xmin - n - pad, xmax + 1 + pad):
                ring.append((x,y_bot))
            x_right = xmax + 1 + pad
            for y in range(ymin - n - pad + 1, ymax + 1 + pad):
                ring.append((x_right,y))
            for (x,y) in ring:
                if feasible_on_occupancy(tiles[v], x, y, occ):
                    cands.append((x,y))
            if cands: break
        return cands

    best_layout=None; best_m=float('inf')

    for it in range(params.iterations):
        for ant in range(params.n_ants):
            placements: Placement = {0:(0,0)}
            placed_ids=[0]
            occ: Dict[Coordinate,int]={}
            write_to_occupancy(tiles[0],0,0,occ)
            bbox=[0,n-1,0,n-1]
            parent_edges: List[Tuple[int,int,int,int]] = []

            while len(placed_ids) < T:
                candidates=[]
                proposed = defaultdict(set)
                for v in range(T):
                    if v in placements: continue
                    for u in placed_ids:
                        ux,uy = placements[u]
                        # Top-K overlap-based proposals
                        for e in pre.topk.get((u,v), []):
                            proposed[v].add((ux+e.dx, uy+e.dy))
                        # NEW: 9 adjacent positions (touching) of v relative to u
                        N = tiles[v].shape[0]
                        for adx in (-N, 0, N):
                            for ady in (-N, 0, N):
                                if adx == 0 and ady == 0:
                                    continue
                                proposed[v].add((ux + adx, uy + ady))

                # Perimeter fallback if still empty for some v
                for v in range(T):
                    if v in placements: continue
                    if not proposed[v]:
                        for (x,y) in perimeter_candidates(v, occ, bbox):
                            proposed[v].add((x,y))

                for v, posset in proposed.items():
                    for (x,y) in posset:
                        if not feasible_on_occupancy(tiles[v], x, y, occ):
                            continue
                        H,Tsum = hp_sums(v,x,y,placed_ids,placements)
                        inc = bbox_increase_if_place(x,y,n,bbox)
                        cp = math.exp(-params.lam * inc)
                        s = ((Tsum if Tsum>0 else 1e-9)**params.alpha) * ((H+params.epsilon)**params.beta) * (cp**params.gamma)
                        candidates.append((v,x,y,H,Tsum,cp,max(s,1e-12)))

                if not candidates:
                    break

                # Roulette-wheel selection
                S = sum(c[-1] for c in candidates)
                r = random.random() * S
                acc=0.0
                chosen=None
                for c in candidates:
                    acc += c[-1]
                    if acc>=r:
                        chosen=c; break
                v,x,y,H,Tsum,cp,_ = chosen

                # parent edge for reinforcement
                best_parent=None; best_ov=-1
                for u in placed_ids:
                    ux,uy = placements[u]
                    dx=x-ux; dy=y-uy
                    ov = ov_lookup.get((u,v,dx,dy), 0)
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

            if len(placements)==T:
                m,_ = layout_bbox(placements, n)
                # evaporation
                for k in list(tau.keys()):
                    tau[k] *= (1.0 - params.rho)
                depo = params.Q / max(1,m)
                for (u,v,dx,dy) in parent_edges:
                    if (u,v,dx,dy) not in tau: tau[(u,v,dx,dy)] = 0.0
                    tau[(u,v,dx,dy)] += depo
                if m < best_m:
                    best_m=m; best_layout=dict(placements)

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
