
# exact_sssp_fast.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time

def _dense_conflict(tile: np.ndarray, x: int, y: int, occ: Dict[Tuple[int,int], int]) -> bool:
    n = tile.shape[0]
    for i in range(n):
        xi = x + i
        row = tile[i]
        for j in range(n):
            c = (xi, y + j)
            v = int(row[j])
            ov = occ.get(c)
            if ov is not None and ov != v:
                return True
    return False

def _write_occ(tile: np.ndarray, x:int, y:int, occ: Dict[Tuple[int,int],int]):
    n = tile.shape[0]
    for i in range(n):
        xi = x + i
        row = tile[i]
        for j in range(n):
            occ[(xi, y + j)] = int(row[j])

def _remove_occ(tile: np.ndarray, x:int, y:int, occ: Dict[Tuple[int,int],int]):
    n = tile.shape[0]
    for i in range(n):
        xi = x + i
        for j in range(n):
            c=(xi,y+j)
            if c in occ:
                del occ[c]

def _bbox_after_place(x:int,y:int,n:int,bbox:Tuple[int,int,int,int]) -> Tuple[int,int,int,int,int]:
    if bbox is None:
        xmin, xmax, ymin, ymax = x, x+n-1, y, y+n-1
    else:
        xmin, xmax, ymin, ymax = bbox
        if x < xmin: xmin = x
        if y < ymin: ymin = y
        tx = x + n - 1; ty = y + n - 1
        if tx > xmax: xmax = tx
        if ty > ymax: ymax = ty
    m = max(xmax-xmin+1, ymax-ymin+1)
    return xmin,xmax,ymin,ymax,m

def _quick_greedy_upper_bound(tiles: List[np.ndarray]):
    n = tiles[0].shape[0]
    occ = {}
    placements = {0:(0,0)}
    _write_occ(tiles[0], 0, 0, occ)
    xmin=xmax=ymin=ymax=0
    for v in range(1, len(tiles)):
        best=None; best_inc=10**9; best_overlap=-1
        props=set()
        for u,(ux,uy) in placements.items():
            for dx in range(-(n-1), n):
                for dy in range(-(n-1), n):
                    props.add((ux+dx, uy+dy))
        for pad in range(0, n+2):
            for x in range(xmin - n - pad, xmax + 1 + pad):
                props.add((x, ymin - n - pad))
                props.add((x, ymax + 1 + pad))
            for y in range(ymin - n - pad, ymax + 1 + pad):
                props.add((xmin - n - pad, y))
                props.add((xmax + 1 + pad, y))
        for (x,y) in props:
            if _dense_conflict(tiles[v], x, y, occ): 
                continue
            ov=0
            rowv = tiles[v]
            for i in range(n):
                xi = x + i
                row = rowv[i]
                for j in range(n):
                    if occ.get((xi,y+j)) == int(row[j]):
                        ov+=1
            bx0,bx1,by0,by1,m = _bbox_after_place(x,y,n,(xmin,xmax,ymin,ymax))
            inc = m - max(xmax-xmin+1, ymax-ymin+1)
            if (ov > best_overlap) or (ov==best_overlap and inc<best_inc):
                best_overlap=ov; best_inc=inc; best=(x,y,bx0,bx1,by0,by1)
        if best is None:
            x = (v)*(n+1); y=0
            bx0,bx1,by0,by1,m = _bbox_after_place(x,y,n,(xmin,xmax,ymin,ymax))
            best=(x,y,bx0,bx1,by0,by1)
        x,y,xmin,xmax,ymin,ymax = best
        placements[v]=(x,y)
        _write_occ(tiles[v], x, y, occ)
    m = max(xmax-xmin+1, ymax-ymin+1)
    return m, placements

def exact_bruteforce_min_canvas(
    tiles: List[np.ndarray],
    time_limit: Optional[float] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    start = time.time()
    n = tiles[0].shape[0]
    T = len(tiles)

    ub_m, ub_pl = _quick_greedy_upper_bound(tiles)
    if verbose:
        print(f"[init] greedy upper bound m={ub_m}")

    best_m = ub_m
    best_pl = dict(ub_pl)

    occ = {}
    _write_occ(tiles[0], 0, 0, occ)
    placements = {0:(0,0)}
    bbox = (0, n-1, 0, n-1)

    # Precompute feasible offsets and weights per (u,v)
    range_d = list(range(-(n-1), n))
    rel_offsets: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
    rel_ok = set()
    w = {}
    for dx in range_d:
        for dy in range_d:
            w[(dx,dy)] = (n - abs(dx))*(n - abs(dy))

    for u in range(T):
        Au = tiles[u]
        for v in range(T):
            if u==v: continue
            Av = tiles[v]
            offs = []
            for dx in range_d:
                ai0 = 0 if dx<=0 else dx
                ai1 = n if dx>=0 else n+dx
                if ai0>=ai1: 
                    continue
                for dy in range_d:
                    aj0 = 0 if dy<=0 else dy
                    aj1 = n if dy>=0 else n+dy
                    if aj0>=aj1:
                        continue
                    bi0 = ai0 - dx; bi1 = ai1 - dx
                    bj0 = aj0 - dy; bj1 = aj1 - dy
                    Aov = Au[ai0:ai1, aj0:aj1]
                    Bov = Av[bi0:bi1, bj0:bj1]
                    if np.any(Aov != Bov):
                        continue
                    offs.append((dx,dy)); rel_ok.add((u,v,dx,dy))
            rel_offsets[(u,v)] = offs

    def dfs():
        nonlocal best_m, best_pl, occ, placements, bbox, start
        # if time_limit is not None and (time.time() - start) > time_limit:
        #     return
        if len(placements) == T:
            xmin,xmax,ymin,ymax = bbox
            m = max(xmax-xmin+1, ymax-ymin+1)
            if m < best_m:
                best_m = m
                best_pl = dict(placements)
            return

        xmin,xmax,ymin,ymax = bbox
        cur_m = max(xmax-xmin+1, ymax-ymin+1)
        if cur_m >= best_m:
            return

        remaining = [t for t in range(T) if t not in placements]
        # same ordering as baseline
        scores=[]
        for v in remaining:
            sc=0
            for u in placements.keys():
                for (dx,dy) in rel_offsets.get((u,v), []):
                    sc = max(sc, w[(dx,dy)])
            scores.append(( -sc, v))
        scores.sort()
        next_ids = [v for _,v in scores]

        for v in next_ids:
            cand=set()
            for u,(ux,uy) in placements.items():
                for (dx,dy) in rel_offsets.get((u,v), []):
                    cand.add((ux+dx, uy+dy))
            wx0 = xmin - (n-1); wx1 = xmax
            wy0 = ymin - (n-1); wy1 = ymax
            for x in range(wx0, wx1+1):
                for y in range(wy0, wy1+1):
                    cand.add((x,y))

            scored=[]
            for (x,y) in cand:
                if _dense_conflict(tiles[v], x, y, occ):
                    continue
                bx0,bx1,by0,by1,m = _bbox_after_place(x,y,n,bbox)
                if m >= best_m:
                    continue
                ov=0
                for u,(ux,uy) in placements.items():
                    dx=x-ux; dy=y-uy
                    if abs(dx) <= n-1 and abs(dy) <= n-1 and (u,v,dx,dy) in rel_ok:
                        ww = w[(dx,dy)]
                        if ww > ov: ov = ww
                scored.append((-ov, m, x, y, bx0,bx1,by0,by1))
            scored.sort()

            for _,_,x,y,bx0,bx1,by0,by1 in scored:
                _write_occ(tiles[v], x, y, occ)
                placements[v]=(x,y)
                old_bbox = bbox
                bbox = (bx0,bx1,by0,by1)
                dfs()
                del placements[v]
                _remove_occ(tiles[v], x, y, occ)
                bbox = old_bbox
                # if time_limit is not None and (time.time() - start) > time_limit:
                #     return

    dfs()

    xmin = ymin = 10**9
    xmax = ymax = -10**9
    for _,(x,y) in best_pl.items():
        xmin=min(xmin,x); ymin=min(ymin,y)
        xmax=max(xmax,x+n-1); ymax=max(ymax,y+n-1)
    W = xmax-xmin+1; H = ymax-ymin+1
    canvas = np.full((H,W), -1, dtype=int)
    for tid,(x,y) in best_pl.items():
        for i in range(n):
            for j in range(n):
                canvas[y - ymin + j, x - xmin + i] = int(tiles[tid][i,j])
    return {
        "status":"ok",
        "best_m": max(W,H),
        "bbox":[xmin,xmax,ymin,ymax],
        "placements": best_pl,
        "canvas": canvas
    }
