"""
2D Tile Placement Environment

This module provides the core environment and utilities for the 2D tile placement problem,
where tiles must be placed on a canvas such that overlapping regions match exactly.

Key Components:
- TilePlacementEnv: Main environment class for single-instance tile placement
- Synthetic tile generation utilities
- Tile embedding precomputation
- Step batch construction for neural network training
- Occupancy grid and bounding box management utilities
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import random
import numpy as np
import torch
import torch.nn.functional as F

# ===== Core Utility Functions =====

Coordinate = Tuple[int, int]
Placement  = Dict[int, Coordinate]

def feasible_on_occupancy(tile: np.ndarray, x: int, y: int, occ: Dict[Tuple[int,int],int]) -> bool:
    """
    Check if a tile can be placed at position (x, y) without conflicts.
    """
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
    """
    Write a tile to the occupancy grid at position (x, y).
    """
    n = tile.shape[0]
    for i in range(n):
        xi = x + i
        row = tile[i]
        for j in range(n):
            occ[(xi, y+j)] = int(row[j])

def update_bbox_for_tile(x: int, y: int, n: int, bbox: List[int]) -> None:
    """
    Update bounding box to include a new n×n tile at position (x, y).
    """
    xmin, xmax, ymin, ymax = bbox
    if x < xmin: xmin = x
    if y < ymin: ymin = y
    tx = x+n-1; ty = y+n-1
    if tx > xmax: xmax = tx
    if ty > ymax: ymax = ty
    bbox[0], bbox[1], bbox[2], bbox[3] = xmin, xmax, ymin, ymax

def bbox_increase_if_place(x: int, y: int, n: int, bbox: List[int]) -> int:
    """
    Calculate how much the maximum dimension would increase if placing a tile.
    """
    xmin, xmax, ymin, ymax = bbox
    nxmin = min(xmin, x); nymin = min(ymin, y)
    nxmax = max(xmax, x+n-1); nymax = max(ymax, y+n-1)
    old_m = max(xmax-xmin+1, ymax-ymin+1)
    new_m = max(nxmax-nxmin+1, nymax-nymin+1)
    return max(0, new_m-old_m)

def layout_bbox(placements: Dict[int,Tuple[int,int]], n: int):
    """
    Calculate the bounding box and maximum dimension of a layout.
    """
    xmin = ymin = 10**9
    xmax = ymax = -10**9
    for (x, y) in placements.values():
        if x < xmin: xmin = x
        if y < ymin: ymin = y
        tx = x+n-1; ty = y+n-1
        if tx > xmax: xmax = tx
        if ty > ymax: ymax = ty
    m = max(xmax-xmin+1, ymax-ymin+1)
    return m, [xmin, xmax, ymin, ymax]

def _adjacent_offsets(n: int) -> List[Tuple[int,int]]:
    """
    Generate adjacency offset positions for a tile of size n×n. (4n+4 total)
    """
    offs = []
    for dy in range(-n, 1):        # left/right (n+1)
        offs.append((-n, dy))
        offs.append((+n, dy))
    for dx in range(-n, 1):        # up/down (n+1)
        offs.append((dx, -n))
        offs.append((dx, +n))
    return offs  # 4n+4

# ===== Pairwise Overlap Precomputation =====

def compute_pairwise_overlaps(tiles: List[np.ndarray], n: int):
    """
    Precompute all possible overlaps between pairs of tiles.
    """
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

# ===== Synthetic Tile Generation =====

def make_synthetic_tiles(T: int, n: int, alphabet: int, seed: Optional[int] = 0) -> List[np.ndarray]:
    """
    Generate synthetic tiles for testing and training.
    """
    rng = np.random.RandomState(seed)
    all_tiles = rng.randint(0, alphabet, size=(T, n, n), dtype=np.int32)
    tiles = [all_tiles[i] for i in range(T)]
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
    Convert occupancy grid to a tensor representation within a cropped region.
    """
    xmin, xmax, ymin, ymax = bbox
    xmin_c = xmin - pad
    ymin_c = ymin - pad
    xmax_c = xmax + pad
    ymax_c = ymax + pad

    H = ymax_c - ymin_c + 1  # along y
    W = xmax_c - xmin_c + 1  # along x

    if max_hw is not None:
        H = min(H, max_hw)
        W = min(W, max_hw)

    C = alphabet + 1  # extra channel for empty
    arr = np.zeros((C, H, W), dtype=np.float32)
    arr[-1, :, :] = 1.0  # empty channel = 1.0 by default

    for (X, Y), val in occ.items():
        if X < xmin_c or X > xmax_c or Y < ymin_c or Y > ymax_c:
            continue
        i = X - xmin_c  # x offset
        j = Y - ymin_c  # y offset
        if 0 <= j < H and 0 <= i < W:
            arr[-1, j, i] = 0.0
            arr[val, j, i] = 1.0

    return torch.from_numpy(arr)

# ===== Main Environment Class =====

@dataclass
class StepBatch:
    occ: torch.Tensor
    tiles_left: torch.Tensor
    tiles_left_mask: torch.Tensor
    cand_feats: torch.Tensor
    cand_mask: torch.Tensor
    cand_tile_idx: torch.Tensor
    expert_action: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "StepBatch":
        dev = torch.device(device)
        return StepBatch(
            occ=self.occ.to(dev),
            tiles_left=self.tiles_left.to(dev),
            tiles_left_mask=self.tiles_left_mask.to(dev),
            cand_feats=self.cand_feats.to(dev),
            cand_mask=self.cand_mask.to(dev),
            cand_tile_idx=self.cand_tile_idx.to(dev),
            expert_action=None if self.expert_action is None else self.expert_action.to(dev),
        )

class TilePlacementEnv:
    """
    Environment for single-instance tile placement problem.
    """

    def __init__(self, tiles: List[np.ndarray], alphabet: int):
        assert len(tiles) >= 1
        self.tiles = tiles
        self.n = tiles[0].shape[0]
        self.T = len(tiles)
        self.alphabet = alphabet

        self.pre_all = compute_pairwise_overlaps(tiles, self.n)
        self.adj_offs = _adjacent_offsets(self.n)
        self.reset()

    def reset(self, start_tile: int = 0):
        n = self.n
        self.placements: Dict[int, Tuple[int, int]] = {start_tile: (0, 0)}
        self.placed_ids: List[int] = [start_tile]
        self.remaining: List[int] = [i for i in range(self.T) if i != start_tile]
        self.occ: Dict[Tuple[int, int], int] = {}
        write_to_occupancy(self.tiles[start_tile], 0, 0, self.occ)
        self.bbox: List[int] = [0, n-1, 0, n-1]
        self.done = False
        return self.state()

    def spawn_clone(self) -> "TilePlacementEnv":
        clone = TilePlacementEnv.__new__(TilePlacementEnv)
        clone.tiles = self.tiles
        clone.n = self.n
        clone.T = self.T
        clone.alphabet = self.alphabet
        clone.pre_all = self.pre_all
        clone.adj_offs = self.adj_offs
        clone.reset()
        return clone

    def state(self):
        return {
            "placements": dict(self.placements),
            "placed_ids": list(self.placed_ids),
            "remaining": list(self.remaining),
            "bbox": list(self.bbox),
        }

    def generate_candidates(self) -> List[Tuple[int, int, int, bool, int, int]]:
        """
        Generate all feasible candidate actions for the current state.

        Returns a list of tuples:
          (tile_id, x, y, is_adjacent, best_overlap_size, H_sum)
        """
        cands: List[Tuple[int, int, int, bool, int, int]] = []
        n = self.n
        occ = self.occ

        for v in self.remaining:
            # Overlap-first
            overlap_positions = set()
            for u in self.placed_ids:
                ux, uy = self.placements[u]
                for (dx, dy, _) in self.pre_all.get((u, v), []):
                    overlap_positions.add((ux + dx, uy + dy))

            for (x, y) in overlap_positions:
                if not feasible_on_occupancy(self.tiles[v], x, y, occ):
                    continue

                best_ov = 0
                H_sum = 0
                for u in self.placed_ids:
                    ux, uy = self.placements[u]
                    dx, dy = x - ux, y - uy
                    for (dx0, dy0, ov) in self.pre_all.get((u, v), []):
                        if dx0 == dx and dy0 == dy:
                            if ov > best_ov:
                                best_ov = ov
                            H_sum += ov
                            break

                cands.append((v, x, y, False, best_ov, H_sum))

            # Adjacency-based (could be use to fill gaps)
            adj_positions = set()
            for u in self.placed_ids:
                ux, uy = self.placements[u]
                for (dx, dy) in self.adj_offs:
                    adj_positions.add((ux + dx, uy + dy))

            for (x, y) in adj_positions:
                if not feasible_on_occupancy(self.tiles[v], x, y, occ):
                    continue
                cands.append((v, x, y, True, 0, 0))

        return cands

    def step(self, cand: Tuple[int, int, int, bool, int, int]):
        """
        Apply a candidate action to the environment.
        """
        v, x, y, is_adj, best_ov, H_sum = cand
        assert v in self.remaining, f"Tile {v} is not in remaining tiles"

        write_to_occupancy(self.tiles[v], x, y, self.occ)
        update_bbox_for_tile(x, y, self.n, self.bbox)

        self.placements[v] = (x, y)
        self.placed_ids.append(v)
        self.remaining.remove(v)

        if len(self.placed_ids) == self.T:
            self.done = True

        return self.state()

    def step_and_metrics(self, cand: Tuple[int, int, int, bool, int, int]) -> Tuple[dict, int, int]:
        """
        Apply action and return (state, delta_m, H_sum).
        delta_m is computed against current bbox without scanning the layout.
        """
        v, x, y, is_adj, best_ov, H_sum = cand
        delta_m = bbox_increase_if_place(x, y, self.n, self.bbox)
        st = self.step(cand)
        return st, int(delta_m), int(H_sum)

# ===== Step Batch Construction =====

@dataclass
class StepBatch:
    occ: torch.Tensor
    tiles_left: torch.Tensor
    tiles_left_mask: torch.Tensor
    cand_feats: torch.Tensor
    cand_mask: torch.Tensor
    cand_tile_idx: torch.Tensor
    expert_action: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "StepBatch":
        dev = torch.device(device)
        return StepBatch(
            occ=self.occ.to(dev),
            tiles_left=self.tiles_left.to(dev),
            tiles_left_mask=self.tiles_left_mask.to(dev),
            cand_feats=self.cand_feats.to(dev),
            cand_mask=self.cand_mask.to(dev),
            cand_tile_idx=self.cand_tile_idx.to(dev),
            expert_action=None if self.expert_action is None else self.expert_action.to(dev),
        )

def build_step_batch_from_env(
    env: TilePlacementEnv,
    tile_embs: torch.Tensor,
    raster_pad: int = 1,
    max_crop_hw: Optional[int] = None,
    device: Optional[torch.device | str] = None,
    return_cands=False
) -> StepBatch:
    """
    Convert environment state into a StepBatch for neural network processing.
    """
    assert not env.done, "Episode already finished."

    occ = rasterize_occ_crop(env.occ, env.bbox, alphabet=env.alphabet, pad=raster_pad, max_hw=max_crop_hw)
    occ = occ.unsqueeze(0)  # (1, C_occ, H, W)

    remaining = env.remaining
    M = len(remaining)
    d_tile = tile_embs.size(-1)
    tiles_left = tile_embs[remaining].unsqueeze(0)  # (1, M, d_tile)
    tiles_left_mask = torch.ones(1, M, dtype=torch.bool)

    raw_cands = env.generate_candidates()
    if len(raw_cands) == 0:
        sb = StepBatch(
            occ=occ,
            tiles_left=tiles_left,
            tiles_left_mask=tiles_left_mask,
            cand_feats=torch.zeros(1, 1, 10),
            cand_mask=torch.zeros(1, 1, dtype=torch.bool),
            cand_tile_idx=torch.zeros(1, 1, dtype=torch.long),
            expert_action=None,
        )
        if return_cands:
            return sb if device is None else sb.to(device), []
        return sb if device is None else sb.to(device)

    n = env.n
    xmin, xmax, ymin, ymax = env.bbox
    old_m = max(xmax-xmin+1, ymax-ymin+1)

    feats: List[List[float]] = []
    tile_idx: List[int] = []
    mask: List[bool] = []

    for (v, x, y, is_adj, best_ov, H_sum) in raw_cands:
        inc = bbox_increase_if_place(x, y, n, env.bbox)
        dxn = (x - xmin) / max(1.0, old_m)
        dyn = (y - ymin) / max(1.0, old_m)

        left_touch  = 1.0 if x <= xmin else 0.0
        right_touch = 1.0 if (x + n - 1) >= xmax else 0.0
        up_touch    = 1.0 if y <= ymin else 0.0
        down_touch  = 1.0 if (y + n - 1) >= ymax else 0.0

        feats.append([
            float(H_sum),        # Total overlap size
            float(inc),          # Canvas size increase
            float(dxn),          # Normalized x
            float(dyn),          # Normalized y
            float(is_adj),       # adjacency flag
            float(best_ov),      # best overlap with any parent
            left_touch,          # touches left edge
            right_touch,         # touches right edge
            up_touch,            # touches top edge
            down_touch           # touches bottom edge
        ])

        tile_idx.append(remaining.index(v))
        mask.append(True)

    feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, A, 10)
    mask_t  = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)      # (1, A)
    tilei_t = torch.tensor(tile_idx, dtype=torch.long).unsqueeze(0)  # (1, A)

    sb = StepBatch(
        occ=occ,
        tiles_left=tiles_left,
        tiles_left_mask=tiles_left_mask,
        cand_feats=feats_t,
        cand_mask=mask_t,
        cand_tile_idx=tilei_t,
        expert_action=None
    )
    if return_cands:
        return sb if device is None else sb.to(device), raw_cands
    return sb if device is None else sb.to(device)

# ===== Tile Embedding Utilities =====

def precompute_tile_embeddings(
    tiles: List[np.ndarray],
    alphabet: int,
    tile_cnn: torch.nn.Module,
    device: torch.device,
    keep_on_device: bool = False,
) -> torch.Tensor:
    """
    Convert tiles to one-hot representation and compute embeddings using a CNN.
    """
    T = len(tiles)
    if T == 0:
        return torch.empty(0, 0, device='cpu')

    n = tiles[0].shape[0]
    X = np.stack(tiles, axis=0)  # (T, n, n)
    oh = np.zeros((T, alphabet, n, n), dtype=np.float32)
    for t in range(T):
        oh[t, X[t], np.arange(n)[:,None], np.arange(n)[None,:]] = 1.0

    x = torch.from_numpy(oh)  # (T, C_sym, n, n)
    with torch.no_grad():
        x = x.to(device)
        emb = tile_cnn(x)      # (T, d_tile)
    return emb if keep_on_device else emb.cpu()
