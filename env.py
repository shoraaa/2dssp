# env_generator.py (can live in the same file as the model)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import random
import numpy as np
import torch
import torch.nn.functional as F

# ===== Utils reused from your ACO scaffolding =====

Coordinate = Tuple[int, int]
Placement  = Dict[int, Coordinate]

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

def _adjacent_offsets(n: int) -> List[Tuple[int,int]]:
    offs=[]
    for dy in range(-n, 1):        # left/right (n+1)
        offs.append((-n, dy))
        offs.append((+n, dy))
    for dx in range(-n, 1):        # up/down (n+1)
        offs.append((dx, -n))
        offs.append((dx, +n))
    return offs  # 4n+4

# ===== Overlap precomputation =====

def compute_pairwise_overlaps(tiles: List[np.ndarray], n: int):
    pre_all: Dict[Tuple[int,int], List[Tuple[int,int,int]]] = {}
    T = len(tiles)
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
                if ai0 >= ai1: continue
                for dy in range(-(n-1), n):
                    aj0 = 0 if dy <= 0 else dy
                    aj1 = n if dy >= 0 else n + dy
                    if aj0 >= aj1: continue
                    bi0 = ai0 - dx; bj0 = aj0 - dy
                    bi1 = ai1 - dx; bj1 = aj1 - dy
                    Aov = A[ai0:ai1, aj0:aj1]
                    Bov = B[bi0:bi1, bj0:bj1]
                    if np.any(Aov != Bov):
                        continue
                    ov = Aov.size
                    if ov > 0:
                        entries.append((dx,dy,int(ov)))
            if entries:
                entries.sort(key=lambda e: e[2], reverse=True)
            pre_all[(u,v)] = entries
    return pre_all

# ===== Synthetic problem generator =====

def make_synthetic_tiles(T: int, n: int, alphabet: int, seed: Optional[int]=0) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    tiles = []
    for _ in range(T):
        tiles.append(rng.randint(0, alphabet, size=(n,n), dtype=np.int64))
    return tiles

# ===== Rasterization =====

def rasterize_occ_crop(
    occ: Dict[Tuple[int,int], int],
    bbox: List[int],
    alphabet: int,
    pad: int = 1,
    max_hw: Optional[int] = None,
):
    """
    Build a crop around bbox with 'pad' tiles of margin.
    Returns tensor (C_occ, H, W), where C_occ = alphabet + 1 (empty).
    """
    xmin,xmax,ymin,ymax = bbox
    xmin_c = xmin - pad
    ymin_c = ymin - pad
    xmax_c = xmax + pad
    ymax_c = ymax + pad
    H = ymax_c - ymin_c + 1
    W = xmax_c - xmin_c + 1

    if max_hw is not None:
        # (Optional) limit crop; here we just cap and recenter roughly
        H = min(H, max_hw)
        W = min(W, max_hw)

    C = alphabet + 1  # extra channel for empty
    arr = np.zeros((C, H, W), dtype=np.float32)
    # fill empty channel
    arr[-1, :, :] = 1.0

    for (X,Y), val in occ.items():
        if not (xmin_c <= X <= xmax_c and ymin_c <= Y <= ymax_c):
            continue
        i = X - xmin_c
        j = Y - ymin_c
        arr[-1, i, j] = 0.0
        arr[val, i, j] = 1.0

    # PyTorch is channel-first but (H,W) usually map to (row,col) = (y,x).
    # Our indexing above used i->x, j->y; to keep consistent with the rest,
    # transpose so that arr[:, H, W] corresponds to (y,x).
    arr = np.transpose(arr, (0,2,1))  # (C, H, W) with H along y
    return torch.from_numpy(arr)

# ===== Environment =====

@dataclass
class StepBatch:
    occ: torch.Tensor              # (B, C_occ, H, W)
    tiles_left: torch.Tensor       # (B, M, d_tile)
    tiles_left_mask: torch.Tensor  # (B, M)
    cand_feats: torch.Tensor       # (B, A, F)
    cand_mask: torch.Tensor        # (B, A)
    cand_tile_idx: torch.Tensor    # (B, A)
    expert_action: Optional[torch.Tensor] = None  # (B,)

class TilePlacementEnv:
    """
    Single-instance environment that constructs a layout by placing tiles.
    Overlap-first; if a tile has no feasible overlap placements, we try adjacency (4n+4 per placed tile).
    """
    def __init__(self, tiles: List[np.ndarray], alphabet: int):
        assert len(tiles) >= 1
        self.tiles = tiles
        self.n = tiles[0].shape[0]
        self.T = len(tiles)
        self.alphabet = alphabet

        self.pre_all = compute_pairwise_overlaps(tiles, self.n)
        self.adj_offs = _adjacent_offsets(self.n)

        # reset state
        self.reset()

    def reset(self, start_tile: int = 0):
        n = self.n
        self.placements: Dict[int,Tuple[int,int]] = {start_tile: (0,0)}
        self.placed_ids: List[int] = [start_tile]
        self.remaining: List[int] = [i for i in range(self.T) if i != start_tile]
        self.occ: Dict[Tuple[int,int],int] = {}
        write_to_occupancy(self.tiles[start_tile], 0, 0, self.occ)
        self.bbox: List[int] = [0, n-1, 0, n-1]
        self.done = False
        return self.state()

    def state(self):
        return {
            "placements": dict(self.placements),
            "placed_ids": list(self.placed_ids),
            "remaining": list(self.remaining),
            "bbox": list(self.bbox),
        }

    # ---- Candidate generation (overlap-first, else adjacency) ----
    def generate_candidates(self) -> List[Tuple[int,int,int,bool,int]]:
        """
        Returns a flat candidate list:
          [(v, x, y, is_adj, best_ov_among_parents), ...]
        where (x,y) are top-left coordinates of v.
        """
        cands: List[Tuple[int,int,int,bool,int]] = []
        n = self.n
        occ = self.occ
        bbox = self.bbox

        # Map v -> candidate list; ensure overlap-first per v
        for v in self.remaining:
            overlap_positions = set()
            for u in self.placed_ids:
                ux, uy = self.placements[u]
                for (dx, dy, _) in self.pre_all.get((u, v), []):
                    overlap_positions.add((ux + dx, uy + dy))

            feasible_overlap_found = False
            best_ov_cache: Dict[Tuple[int,int], int] = {}

            # Overlap candidates (filtered by feasibility)
            for (x,y) in overlap_positions:
                if not feasible_on_occupancy(self.tiles[v], x, y, occ):
                    continue
                # gather max ov among parents (for features)
                best_ov = 0
                for u in self.placed_ids:
                    ux, uy = self.placements[u]
                    dx, dy = x - ux, y - uy
                    # check if (u,v,dx,dy) is an exact-overlap edge; if yes, ov > 0
                    for (dx0, dy0, ov) in self.pre_all.get((u, v), []):
                        if dx0 == dx and dy0 == dy:
                            if ov > best_ov: best_ov = ov
                            break
                best_ov_cache[(x,y)] = best_ov
                cands.append((v, x, y, False, best_ov))
                feasible_overlap_found = True

            # If no overlap placements for this v, adjacency
            if not feasible_overlap_found:
                adj_positions = set()
                for u in self.placed_ids:
                    ux, uy = self.placements[u]
                    for (dx, dy) in self.adj_offs:
                        adj_positions.add((ux + dx, uy + dy))

                for (x,y) in adj_positions:
                    if not feasible_on_occupancy(self.tiles[v], x, y, occ):
                        continue
                    # best_ov among current parents will be 0 for adjacency (by definition)
                    cands.append((v, x, y, True, 0))

        return cands

    # ---- Step (apply chosen candidate index) ----
    def step(self, cand: Tuple[int,int,int,bool,int]):
        v, x, y, is_adj, best_ov = cand
        assert v in self.remaining
        # commit
        write_to_occupancy(self.tiles[v], x, y, self.occ)
        update_bbox_for_tile(x, y, self.n, self.bbox)
        self.placements[v] = (x, y)
        self.placed_ids.append(v)
        self.remaining.remove(v)
        if len(self.placed_ids) == self.T:
            self.done = True
        return self.state()

# ===== Candidate → StepBatch packing =====

def build_step_batch_from_env(
    env: TilePlacementEnv,
    tile_embs: torch.Tensor,          # (T, d_tile), precomputed with your TileCNN
    raster_pad: int = 1,
    max_crop_hw: Optional[int] = None,
) -> StepBatch:
    """
    Build a (B=1) StepBatch from the environment's current state.
    Candidate features (F) include:
      [ H_sum, delta_m, dx_norm, dy_norm, is_adj, best_ov, touches_left, right, up, down ]
    You can extend this easily.
    """
    assert not env.done, "Episode already finished."

    # Raster
    occ = rasterize_occ_crop(env.occ, env.bbox, alphabet=env.alphabet, pad=raster_pad, max_hw=max_crop_hw)  # (C_occ,H,W)
    occ = occ.unsqueeze(0)  # (1, C_occ, H, W)

    # Remaining tiles → tiles_left embeddings and mask
    remaining = env.remaining
    M = len(remaining)
    d_tile = tile_embs.size(-1)
    tiles_left = tile_embs[remaining].unsqueeze(0)  # (1, M, d_tile)
    tiles_left_mask = torch.ones(1, M, dtype=torch.bool)  # all valid

    # Candidates
    raw_cands = env.generate_candidates()
    if len(raw_cands) == 0:
        # No moves (should be rare); return an empty masked batch
        return StepBatch(
            occ=occ,
            tiles_left=tiles_left,
            tiles_left_mask=tiles_left_mask,
            cand_feats=torch.zeros(1,1,8),
            cand_mask=torch.zeros(1,1,dtype=torch.bool),
            cand_tile_idx=torch.zeros(1,1,dtype=torch.long),
            expert_action=None,
        )

    # Build features
    n = env.n
    xmin,xmax,ymin,ymax = env.bbox
    old_m = max(xmax-xmin+1, ymax-ymin+1)

    feats: List[List[float]] = []
    tile_idx: List[int] = []
    mask: List[bool] = []

    # For H_sum we compute sum of overlaps with all parents for this (v,x,y)
    # (We can reuse pre_all; for adjacency this is zero.)
    Hs: List[float] = []
    for (v, x, y, is_adj, best_ov) in raw_cands:
        # sum overlap over all parents where (dx,dy) exists
        H_sum = 0
        for u in env.placed_ids:
            ux, uy = env.placements[u]
            dx, dy = x - ux, y - uy
            for (dx0, dy0, ov) in env.pre_all.get((u, v), []):
                if dx0 == dx and dy0 == dy:
                    H_sum += ov
                    break
        Hs.append(float(H_sum))

    for idx, (v, x, y, is_adj, best_ov) in enumerate(raw_cands):
        # Δm
        inc = bbox_increase_if_place(x, y, n, env.bbox)
        # normalized dx, dy w.r.t. bbox origin
        dxn = (x - xmin) / max(1.0, old_m)
        dyn = (y - ymin) / max(1.0, old_m)
        # touches which sides after placement (coarse heuristic)
        left_touch  = 1.0 if x <= xmin else 0.0
        right_touch = 1.0 if (x + n - 1) >= xmax else 0.0
        up_touch    = 1.0 if y <= ymin else 0.0
        down_touch  = 1.0 if (y + n - 1) >= ymax else 0.0

        feats.append([
            Hs[idx],
            float(inc),
            float(dxn),
            float(dyn),
            float(is_adj),
            float(best_ov),
            left_touch, right_touch, up_touch, down_touch
        ])
        # map tile id v → its index within remaining
        tile_idx.append(remaining.index(v))
        mask.append(True)

    feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, A, F)
    mask_t  = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)      # (1, A)
    tilei_t = torch.tensor(tile_idx, dtype=torch.long).unsqueeze(0)  # (1, A)

    return StepBatch(
        occ=occ,
        tiles_left=tiles_left,
        tiles_left_mask=tiles_left_mask,
        cand_feats=feats_t,
        cand_mask=mask_t,
        cand_tile_idx=tilei_t,
        expert_action=None
    )

# ===== Simple greedy expert (for IL bootstrapping) =====

def choose_expert_action(step_batch: StepBatch) -> torch.Tensor:
    """
    A basic expert: prioritize larger H_sum, then smaller delta_m.
    Assumes feature order as built above.
    Returns (B,) action indices.
    """
    feats = step_batch.cand_feats  # (B, A, F)
    mask  = step_batch.cand_mask
    B, A, Fdim = feats.shape
    assert B == 1, "This simple expert handles B=1; extend as needed."

    H_sum   = feats[..., 0]    # (B, A)
    delta_m = feats[..., 1]

    # score = large H_sum, tie-broken by small delta_m
    # normalize a bit
    Hn = (H_sum - H_sum.min(dim=-1, keepdim=True).values) / (H_sum.max(dim=-1, keepdim=True).values - H_sum.min(dim=-1, keepdim=True).values + 1e-6)
    Dn = (delta_m - delta_m.min(dim=-1, keepdim=True).values) / (delta_m.max(dim=-1, keepdim=True).values - delta_m.min(dim=-1, keepdim=True).values + 1e-6)
    score = Hn - 0.25 * Dn
    score = score.masked_fill(~mask, -1e9)
    action = score.argmax(dim=-1)  # (B,)
    return action

# ===== Tile embedding helper =====

def precompute_tile_embeddings(tiles: List[np.ndarray], alphabet: int, tile_cnn: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """
    Converts integer tiles (n×n) into one-hot (alphabet) and runs tile_cnn.
    Returns (T, d_tile).
    """
    T = len(tiles)
    n = tiles[0].shape[0]
    # one-hot channels
    X = np.stack(tiles, axis=0)  # (T, n, n)
    oh = np.zeros((T, alphabet, n, n), dtype=np.float32)
    for t in range(T):
        oh[t, X[t], np.arange(n)[:,None], np.arange(n)[None,:]] = 1.0
    # torch
    x = torch.from_numpy(oh)  # (T, C_sym, n, n)
    with torch.no_grad():
        x = x.to(device)
        emb = tile_cnn(x)      # (T, d_tile)
    return emb.cpu()

# ===== Tiny demo loop (single instance, IL against greedy expert) =====

def demo_train_one_instance(
    model, tile_cnn, T=12, n=6, alphabet=4, steps=200, lr=1e-3, device=torch.device("cpu")
):
    # synth
    tiles = make_synthetic_tiles(T=T, n=n, alphabet=alphabet, seed=0)
    env = TilePlacementEnv(tiles, alphabet=alphabet)
    # embeddings
    tile_embs = precompute_tile_embeddings(tiles, alphabet, tile_cnn, device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for it in range(steps):
        # restart if finished or stuck in rare no-cand state
        if env.done:
            env.reset()

        sb = build_step_batch_from_env(env, tile_embs)
        # expert label
        act = choose_expert_action(sb)
        sb.expert_action = act

        logits, _ = model(
            sb.occ.to(device),
            sb.tiles_left.to(device),
            sb.tiles_left_mask.to(device),
            sb.cand_feats.to(device),
            sb.cand_mask.to(device),
            sb.cand_tile_idx.to(device),
        )
        loss = F.nll_loss(
            # masked log-softmax
            (logits - 1e9 * (~sb.cand_mask.to(device))).log_softmax(dim=-1),
            act.to(device),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # step in env using model argmax (on CPU to keep state in numpy)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1).cpu()
            probs[~sb.cand_mask] = 0.0
            a = probs.argmax(dim=-1).item()

        # apply chosen candidate
        # map back to concrete (v,x,y,...) to env.step
        raw_cands = env.generate_candidates()
        env.step(raw_cands[a])

        if (it+1) % 25 == 0:
            m, _ = layout_bbox(env.placements, env.n)
            print(f"[it {it+1}] loss={loss.item():.4f}  placed={len(env.placed_ids)}/{env.T}  m={m}")

    # final layout result
    m, bbox = layout_bbox(env.placements, env.n)
    return {"m": m, "bbox": bbox, "placements": dict(env.placements)}
