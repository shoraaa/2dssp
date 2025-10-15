# greedy_2d_sssp.py
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

Coordinate = Tuple[int, int]
Placement = Dict[int, Tuple[int,int]]

@dataclass
class GreedyParams:
    K: int = 100
    rng_seed: Optional[int] = 0
    perimeter_search_limit: int = 64
    strategy: str = "min_bbox"  # "max_overlap" or "min_bbox"

class TopKEntry:
    __slots__ = ("dx", "dy", "ov")
    def __init__(self, dx:int, dy:int, ov:int):
        self.dx=dx; self.dy=dy; self.ov=ov

def compute_pairwise_topk(tiles: List[np.ndarray], n: int, K: int):
    topk = {}
    T = len(tiles)
    for u in range(T):
        for v in range(T):
            if u == v: continue
            entries: List[TopKEntry] = []
            A = tiles[u]; B = tiles[v]
            for dx in range(-(n-1), n):
                for dy in range(-(n-1), n):
                    ai0 = 0 if dx <= 0 else dx
                    aj0 = 0 if dy <= 0 else dy
                    ai1 = n if dx >= 0 else n + dx
                    aj1 = n if dy >= 0 else n + dy
                    if ai0 >= ai1 or aj0 >= aj1:
                        continue
                    bi0 = ai0 - dx; bj0 = aj0 - dy
                    bi1 = ai1 - dx; bj1 = aj1 - dy
                    Aov = A[ai0:ai1, aj0:aj1]
                    Bov = B[bi0:bi1, bj0:bj1]
                    if np.any(Aov != Bov): continue
                    ov = Aov.size
                    if ov>0: entries.append(TopKEntry(dx,dy,int(ov)))
            if entries:
                random.shuffle(entries)
                entries.sort(key=lambda e: e.ov, reverse=True)
                topk[(u,v)] = entries[:K]
            else:
                topk[(u,v)] = []
    ov_lookup = {(u,v,e.dx,e.dy): e.ov for (u,v), lst in topk.items() for e in lst}
    return topk, ov_lookup

def feasible_on_occupancy(tile: np.ndarray, x: int, y: int, occ: Dict[Coordinate,int]) -> bool:
    n = tile.shape[0]
    for i in range(n):
        for j in range(n):
            val = int(tile[i,j])
            coord = (x+i,y+j)
            if coord in occ and occ[coord] != val:
                return False
    return True

def write_to_occupancy(tile: np.ndarray, x: int, y: int, occ: Dict[Coordinate,int]) -> None:
    n = tile.shape[0]
    for i in range(n):
        for j in range(n):
            occ[(x+i,y+j)] = int(tile[i,j])

def bbox_after_place(x:int,y:int,n:int,bbox:List[int]):
    xmin,xmax,ymin,ymax = bbox
    nxmin=min(xmin,x); nymin=min(ymin,y)
    nxmax=max(xmax,x+n-1); nymax=max(ymax,y+n-1)
    m = max(nxmax-nxmin+1, nymax-nymin+1)
    return m, [nxmin,nxmax,nymin,ymax]

def layout_bbox(placements: Dict[int,Tuple[int,int]], n:int):
    xmin=ymin=10**9; xmax=ymax=-10**9
    for x,y in placements.values():
        xmin=min(xmin,x); ymin=min(ymin,y)
        xmax=max(xmax,x+n-1); ymax=max(ymax,y+n-1)
    m = max(xmax-xmin+1, ymax-ymin+1)
    return m, [xmin,xmax,ymin,ymax]

def greedy_place_once(tiles: List[np.ndarray], params: GreedyParams) -> Dict[str,Any]:
    if params.rng_seed is not None:
        random.seed(params.rng_seed); np.random.seed(params.rng_seed)
    T=len(tiles); assert T>=1
    n=tiles[0].shape[0]
    topk, ov_lookup = compute_pairwise_topk(tiles, n, params.K)
    order=list(range(T)); random.shuffle(order)
    placements: Dict[int,Tuple[int,int]] = {}
    occ: Dict[Coordinate,int]={}
    root=order[0]; placements[root]=(0,0)
    write_to_occupancy(tiles[root], 0, 0, occ)
    bbox=[0,n-1,0,n-1]

    def perimeter_candidates(v:int):
        n_ = tiles[v].shape[0]
        xmin,xmax,ymin,ymax = bbox
        cands=[]
        for pad in range(params.perimeter_search_limit):
            y = ymin - n_ - pad
            for x in range(xmin - n_ - pad, xmax + 1 + pad): cands.append((x,y))
            x = xmin - n_ - pad
            for y in range(ymin - n_ - pad + 1, ymax + 1 + pad): cands.append((x,y))
            y = ymax + 1 + pad
            for x in range(xmin - n_ - pad, xmax + 1 + pad): cands.append((x,y))
            x = xmax + 1 + pad
            for y in range(ymin - n_ - pad, ymax + 1 + pad): cands.append((x,y))
            feasible=[(x,y) for (x,y) in cands if feasible_on_occupancy(tiles[v], x, y, occ)]
            if feasible: return feasible
        return []

    for idx in range(1,T):
        v=order[idx]
        posset=set()
        placed_ids=list(placements.keys())
        for u in placed_ids:
            ux,uy=placements[u]
            for e in topk.get((u,v), []):
                posset.add((ux+e.dx, uy+e.dy))
        best=None
        for (x,y) in posset:
            if not feasible_on_occupancy(tiles[v], x, y, occ): continue
            total_ov=0
            for u in placed_ids:
                ux,uy=placements[u]
                dx=x-ux; dy=y-uy
                total_ov += ov_lookup.get((u,v,dx,dy), 0)
            new_m,_=bbox_after_place(x,y,n,bbox)
            key=(total_ov, -new_m, random.random())
            if best is None or key>best[0]:
                best=(key,x,y)
        if best is not None and best[0][0] > 0:
            _,x,y = best
        else:
            fallback = perimeter_candidates(v)
            if not fallback:
                x,y = bbox[0]-n, bbox[2]-n
            else:
                bestp=None
                for (px,py) in fallback:
                    new_m,_=bbox_after_place(px,py,n,bbox)
                    key=(-new_m, random.random())
                    if bestp is None or key>bestp[0]: bestp=(key,px,py)
                _,x,y = bestp
        placements[v]=(x,y)
        write_to_occupancy(tiles[v], x, y, occ)
        _,bbox = bbox_after_place(x,y,n,bbox)
    m,bbox = layout_bbox(placements, n)
    xmin,xmax,ymin,ymax = bbox
    W=xmax-xmin+1; H=ymax-ymin+1
    canvas = np.full((H,W), -1, dtype=int)
    for (X,Y),val in occ.items():
        canvas[Y - ymin, X - xmin] = val
    return {"status":"ok","order":order,"placements":placements,"best_m":m,"bbox":bbox,"canvas":canvas}

def greedy_place_min_bbox(tiles: List[np.ndarray], params: GreedyParams) -> Dict[str,Any]:
    """
    Greedy placement algorithm that prioritizes minimal bounding box size increase,
    with maximum overlap as a tie-breaker.
    """
    if params.rng_seed is not None:
        random.seed(params.rng_seed); np.random.seed(params.rng_seed)
    T=len(tiles); assert T>=1
    n=tiles[0].shape[0]
    topk, ov_lookup = compute_pairwise_topk(tiles, n, params.K)
    order=list(range(T)); random.shuffle(order)
    placements: Dict[int,Tuple[int,int]] = {}
    occ: Dict[Coordinate,int]={}
    root=order[0]; placements[root]=(0,0)
    write_to_occupancy(tiles[root], 0, 0, occ)
    bbox=[0,n-1,0,n-1]

    def perimeter_candidates(v:int):
        n_ = tiles[v].shape[0]
        xmin,xmax,ymin,ymax = bbox
        cands=[]
        for pad in range(params.perimeter_search_limit):
            y = ymin - n_ - pad
            for x in range(xmin - n_ - pad, xmax + 1 + pad): cands.append((x,y))
            x = xmin - n_ - pad
            for y in range(ymin - n_ - pad + 1, ymax + 1 + pad): cands.append((x,y))
            y = ymax + 1 + pad
            for x in range(xmin - n_ - pad, xmax + 1 + pad): cands.append((x,y))
            x = xmax + 1 + pad
            for y in range(ymin - n_ - pad, ymax + 1 + pad): cands.append((x,y))
            feasible=[(x,y) for (x,y) in cands if feasible_on_occupancy(tiles[v], x, y, occ)]
            if feasible: return feasible
        return []

    for idx in range(1,T):
        v=order[idx]
        posset=set()
        placed_ids=list(placements.keys())
        for u in placed_ids:
            ux,uy=placements[u]
            for e in topk.get((u,v), []):
                posset.add((ux+e.dx, uy+e.dy))
        best=None
        for (x,y) in posset:
            if not feasible_on_occupancy(tiles[v], x, y, occ): continue
            total_ov=0
            for u in placed_ids:
                ux,uy=placements[u]
                dx=x-ux; dy=y-uy
                total_ov += ov_lookup.get((u,v,dx,dy), 0)
            new_m,_=bbox_after_place(x,y,n,bbox)
            # Priority: minimal bbox increase, then maximum overlap
            key=(-new_m, total_ov, random.random())
            if best is None or key>best[0]:
                best=(key,x,y)
        if best is not None and best[0][1] > 0:  # check overlap (second element)
            _,x,y = best
        else:
            fallback = perimeter_candidates(v)
            if not fallback:
                x,y = bbox[0]-n, bbox[2]-n
            else:
                bestp=None
                for (px,py) in fallback:
                    new_m,_=bbox_after_place(px,py,n,bbox)
                    key=(-new_m, random.random())
                    if bestp is None or key>bestp[0]: bestp=(key,px,py)
                _,x,y = bestp
        placements[v]=(x,y)
        write_to_occupancy(tiles[v], x, y, occ)
        _,bbox = bbox_after_place(x,y,n,bbox)
    m,bbox = layout_bbox(placements, n)
    xmin,xmax,ymin,ymax = bbox
    W=xmax-xmin+1; H=ymax-ymin+1
    canvas = np.full((H,W), -1, dtype=int)
    for (X,Y),val in occ.items():
        canvas[Y - ymin, X - xmin] = val
    return {"status":"ok","order":order,"placements":placements,"best_m":m,"bbox":bbox,"canvas":canvas}

def greedy_place(tiles: List[np.ndarray], params: GreedyParams) -> Dict[str,Any]:
    """
    Main greedy placement function that selects strategy based on params.
    """
    if params.strategy == "min_bbox":
        return greedy_place_min_bbox(tiles, params)
    else:  # default to "max_overlap"
        return greedy_place_once(tiles, params)
