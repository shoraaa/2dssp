# env.py
# 2D-SSP Tile Placement Environment (discrete, maskable actions)
# --------------------------------------------------------------
# - Tiles are n x n integer grids: 0 = empty, 1..K = symbol ids.
# - A placement (x, y) is the TOP-LEFT of a tile in absolute canvas coords.
# - Overlap is feasible iff any overlapping non-zero cells are EQUAL.
# - Adjacency candidates (when no feasible overlap exists for a tile) are the
#   4*(n+1) positions that touch the 4 sides of each placed tile (no overlap).
# - Action space per step is the union over remaining tiles:
#       actions = concat_v [ (v, candidate_index) ]
#   where candidate_index enumerates that tile's candidates.
#
# Exposed API (what training code needs):
#   - TilePlacementEnv(tiles, tile_size, alphabet, seed=0)
#   - reset(initial_tiles=None)
#   - enumerate_candidates() -> CandidateBatch
#   - step(v, cand) -> info dict (including delta_bbox_area, overlap_used, done)
#   - rasterize_occ_crop(max_hw=None, pad=2) -> (C_occ, H, W) float32 tensor
#   - get_layout_graph() -> nodes, edges (for GNN)
#   - build_step_batch_from_env(env, max_candidates=None)  # helper for batching
#
# Notes:
# - This module is torch/numpy friendly but does not require CUDA directly.
# - Keep it deterministic for unit tests by setting seed.
# - Canvas stored sparsely: placed tiles dict + fast occupancy index.
# - BBox tracks the min/max occupied coords for reward shaping.
#
# Author: shora-2025

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
import torch


# -------------------------------
# Data containers
# -------------------------------

@dataclass
class Tile:
    """A single tile as an integer grid (0 = empty, 1..K = symbol ids)."""
    id: int
    arr: np.ndarray  # shape (n, n), dtype=np.int16 or np.int32
    # Precomputed non-zero cell coordinates grouped by symbol for fast overlaps
    nz_by_symbol: Dict[int, np.ndarray]  # symbol -> array of (i,j)

@dataclass
class Placement:
    tile_id: int
    x: int  # absolute top-left x
    y: int  # absolute top-left y

@dataclass
class Candidate:
    tile_id: int
    x: int
    y: int
    kind: str  # "overlap" or "adjacency"
    from_tile_id: Optional[int] = None  # which placed tile produced it (optional)
    ov_cells: int = 0  # number of overlapping equal-symbol cells (for features)

@dataclass
class CandidateBatch:
    """Flattened view for all remaining tiles in the step."""
    # concatenated candidate list
    tiles: np.ndarray        # [A] tile_id for each candidate
    xy: np.ndarray           # [A,2] int32 absolute positions
    kinds: np.ndarray        # [A] 0=overlap, 1=adjacency
    from_tiles: np.ndarray   # [A] int32 or -1
    ov_cells: np.ndarray     # [A] int32 overlap cardinality
    # indices to slice per tile: map tile_id -> (start, end)
    idx_per_tile: Dict[int, Tuple[int, int]]
    # useful masks
    mask: np.ndarray         # [A] bool (always True here; reserve for pruning)

# -------------------------------
# Utility
# -------------------------------

def _as_int_tuple(a) -> Tuple[int, int]:
    return int(a[0]), int(a[1])

def _bbox_union(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    if b1 is None: return b2
    if b2 is None: return b1
    x0 = min(b1[0], b2[0]); y0 = min(b1[1], b2[1])
    x1 = max(b1[2], b2[2]); y1 = max(b1[3], b2[3])
    return (x0, y0, x1, y1)

def _bbox_of_tile_at(x: int, y: int, n: int) -> Tuple[int,int,int,int]:
    return (x, y, x + n - 1, y + n - 1)

def _bbox_area(b: Optional[Tuple[int,int,int,int]]) -> int:
    if b is None: return 0
    (x0, y0, x1, y1) = b
    return (x1 - x0 + 1) * (y1 - y0 + 1)

# -------------------------------
# Environment
# -------------------------------

class TilePlacementEnv:
    """
    Environment for placing n x n symbol tiles onto an infinite canvas.

    State:
      - placed: Dict[tile_id] -> (x,y)
      - remaining: set[tile_id]
      - occ_index: Dict[(x,y)] -> symbol (>0), tracks current canvas occupancy
      - occ_multi: Dict[(x,y)] -> True if multiple tiles stacked (equal symbol)
      - bbox: Optional[(x0,y0,x1,y1)] bounding box of occupied cells
    """
    def __init__(
        self,
        tiles: List[np.ndarray],
        tile_size: int,
        alphabet: int,
        seed: int = 0,
    ):
        assert len(tiles) > 0
        self.n = int(tile_size)
        self.K = int(alphabet)
        self.rng = np.random.RandomState(seed)

        # Normalize tiles and precompute nz_by_symbol
        self.tiles: Dict[int, Tile] = {}
        for i, arr in enumerate(tiles):
            assert arr.shape == (self.n, self.n)
            arr = arr.astype(np.int32, copy=False)
            nz_by_symbol = {}
            for s in range(1, self.K + 1):
                coords = np.argwhere(arr == s)
                if coords.size:
                    nz_by_symbol[s] = coords.astype(np.int16, copy=False)
            self.tiles[i] = Tile(id=i, arr=arr, nz_by_symbol=nz_by_symbol)

        self.all_ids = list(self.tiles.keys())
        self.reset()

    # ------------- Core state -------------

    def reset(self, initial_tiles: Optional[List[int]] = None):
        """Reset to empty canvas or place one random seed tile to break symmetry."""
        self.placed: Dict[int, Tuple[int,int]] = {}
        self.remaining: List[int] = list(self.all_ids)
        self.occ_index: Dict[Tuple[int,int], int] = {}
        self.occ_multi: Dict[Tuple[int,int], bool] = {}
        self.bbox: Optional[Tuple[int,int,int,int]] = None

        # Optionally place one tile at (0,0) to initialize a canvas
        if initial_tiles is None and len(self.remaining) > 0:
            seed_id = self.rng.choice(self.remaining)
            self._apply_place(seed_id, 0, 0)
        elif initial_tiles:
            for tid in initial_tiles:
                self._apply_place(tid, 0, 0)

        return self._obs()

    def done(self) -> bool:
        return len(self.remaining) == 0

    # ------------- Observation helpers -------------

    def _obs(self) -> Dict:
        return {
            "placed": dict(self.placed),
            "remaining": list(self.remaining),
            "bbox": self.bbox,
        }

    # ------------- Canvas ops -------------

    def _check_feasible(self, tile_id: int, x: int, y: int) -> Tuple[bool, int]:
        """
        Check feasibility of placing tile_id at (x,y):
          - Any overlap with occupied cells must match the same symbol (>0).
          - We allow stacking equal symbols (treated as one visible symbol).
        Returns: (feasible, overlap_cells_count)
        """
        t = self.tiles[tile_id]
        overlap = 0
        arr = t.arr
        n = self.n
        # Early exit if it conflicts: loop only over nonzero cells in tile
        for s, coords in t.nz_by_symbol.items():
            # absolute coords of these cells if placed
            abs_xy = coords + np.array([y, x], dtype=np.int32)
            for ij in abs_xy:
                ay, ax = int(ij[0]), int(ij[1])
                if (ax, ay) in self.occ_index:
                    s_occ = self.occ_index[(ax, ay)]
                    if s_occ != s:
                        return (False, 0)
                    overlap += 1
        return (True, overlap)

    def _apply_place(self, tile_id: int, x: int, y: int):
        """Mutating write: place tile, update indices and bbox, remove from remaining."""
        assert tile_id in self.remaining, "tile already placed"
        t = self.tiles[tile_id]
        arr = t.arr
        n = self.n

        # Write into occ_index, mark multi if stacking equal symbol
        for s, coords in t.nz_by_symbol.items():
            abs_xy = coords + np.array([y, x], dtype=np.int32)
            for ij in abs_xy:
                ay, ax = int(ij[0]), int(ij[1])
                key = (ax, ay)
                if key in self.occ_index:
                    # equal symbol stacking
                    self.occ_multi[key] = True
                else:
                    self.occ_index[key] = s

        # Update bbox
        tile_bbox = _bbox_of_tile_at(x, y, n)
        self.bbox = _bbox_union(self.bbox, tile_bbox)

        # Move lists
        self.placed[tile_id] = (x, y)
        self.remaining.remove(tile_id)

    # ------------- Candidate generation -------------

    def _overlap_candidates_for_v(self, v: int) -> List[Candidate]:
        """Generate all feasible overlap candidates for tile v w.r.t current canvas."""
        if not self.placed:
            # If nothing placed (rare if reset with seed), no overlaps exist
            return []

        cand: List[Candidate] = []
        t_v = self.tiles[v]
        # Build a reverse index: for each symbol s, absolute positions of s on canvas
        # Also group by source tile to annotate from_tile_id (optional)
        canvas_s_to_abs = {}
        cell_to_tile = {}

        for u, (xu, yu) in self.placed.items():
            t_u = self.tiles[u]
            for s, coords_u in t_u.nz_by_symbol.items():
                if s not in canvas_s_to_abs:
                    canvas_s_to_abs[s] = []
                abs_u = coords_u + np.array([yu, xu], dtype=np.int32)
                for ij in abs_u:
                    ay, ax = int(ij[0]), int(ij[1])
                    canvas_s_to_abs[s].append((ax, ay))
                    cell_to_tile[(ax, ay)] = u

        # For each (symbol s) present in both v and canvas, compute displacements
        for s, coords_v in t_v.nz_by_symbol.items():
            if s not in canvas_s_to_abs:
                continue
            abs_canvas = canvas_s_to_abs[s]  # list of (ax, ay)
            # For each pair (i,j) in v with symbol s and (ax, ay) in canvas with same s:
            # solve for (x,y): x = ax - j, y = ay - i
            for (i, j) in coords_v.tolist():
                for (ax, ay) in abs_canvas:
                    x = ax - j
                    y = ay - i
                    feasible, ov = self._check_feasible(v, x, y)
                    if feasible:
                        cand.append(
                            Candidate(
                                tile_id=v, x=x, y=y, kind="overlap",
                                from_tile_id=cell_to_tile[(ax, ay)], ov_cells=ov
                            )
                        )
        # Deduplicate identical (x,y) proposals while keeping max ov_cells
        if cand:
            uniq = {}
            for c in cand:
                key = (c.x, c.y)
                if key not in uniq or c.ov_cells > uniq[key].ov_cells:
                    uniq[key] = c
            cand = list(uniq.values())
        return cand

    def _adjacency_candidates_for_v(self, v: int) -> List[Candidate]:
        """For tile v, propose 4*(n+1) positions per placed tile that touch edges."""
        if not self.placed:
            # No placed tiles: adjacency to an empty canvas reduces to (0,0)
            return [Candidate(tile_id=v, x=0, y=0, kind="adjacency", from_tile_id=None)]

        n = self.n
        cand: List[Candidate] = []
        for u, (xu, yu) in self.placed.items():
            # Left of u: v.x = xu - n, v.y = yu + t, t in [-n, 0]
            for t in range(-n, 1):
                x = xu - n
                y = yu + t
                # Must not overlap any occupied cell (strict adjacency)
                feasible, ov = self._check_feasible(v, x, y)
                if feasible and ov == 0:
                    cand.append(Candidate(tile_id=v, x=x, y=y, kind="adjacency", from_tile_id=u))

            # Right of u: v.x = xu + n, v.y = yu + t
            for t in range(-n, 1):
                x = xu + n
                y = yu + t
                feasible, ov = self._check_feasible(v, x, y)
                if feasible and ov == 0:
                    cand.append(Candidate(tile_id=v, x=x, y=y, kind="adjacency", from_tile_id=u))

            # Above u: v.y = yu - n, v.x = xu + t
            for t in range(-n, 1):
                x = xu + t
                y = yu - n
                feasible, ov = self._check_feasible(v, x, y)
                if feasible and ov == 0:
                    cand.append(Candidate(tile_id=v, x=x, y=y, kind="adjacency", from_tile_id=u))

            # Below u: v.y = yu + n, v.x = xu + t
            for t in range(-n, 1):
                x = xu + t
                y = yu + n
                feasible, ov = self._check_feasible(v, x, y)
                if feasible and ov == 0:
                    cand.append(Candidate(tile_id=v, x=x, y=y, kind="adjacency", from_tile_id=u))

        # Deduplicate
        if cand:
            uniq = {}
            for c in cand:
                key = (c.x, c.y)
                # prefer closest adjacency to source tile (arbitrary tie-break)
                if key not in uniq:
                    uniq[key] = c
            cand = list(uniq.values())
        return cand

    def enumerate_candidates(self) -> CandidateBatch:
        """
        Build the *global* candidate list for a step:
          - For each remaining tile v:
            * Try overlap candidates
            * If none, fall back to adjacency candidates
        Returns a flattened CandidateBatch plus per-tile index mapping.
        """
        tiles_all: List[int] = []
        xy_all: List[Tuple[int,int]] = []
        kinds_all: List[int] = []
        from_all: List[int] = []
        ov_all: List[int] = []
        idx_per_tile: Dict[int, Tuple[int,int]] = {}

        A = 0
        for v in list(self.remaining):
            c_ov = self._overlap_candidates_for_v(v)
            c_list = c_ov if len(c_ov) > 0 else self._adjacency_candidates_for_v(v)
            start = A
            for c in c_list:
                tiles_all.append(c.tile_id)
                xy_all.append((c.x, c.y))
                kinds_all.append(0 if c.kind == "overlap" else 1)
                from_all.append(-1 if c.from_tile_id is None else c.from_tile_id)
                ov_all.append(c.ov_cells)
            A += len(c_list)
            idx_per_tile[v] = (start, A)

        if A == 0:
            # No moves possible (should be rare; treat as terminal)
            return CandidateBatch(
                tiles=np.zeros((0,), dtype=np.int32),
                xy=np.zeros((0,2), dtype=np.int32),
                kinds=np.zeros((0,), dtype=np.int8),
                from_tiles=np.zeros((0,), dtype=np.int32),
                ov_cells=np.zeros((0,), dtype=np.int32),
                idx_per_tile=idx_per_tile,
                mask=np.zeros((0,), dtype=bool),
            )

        tiles_arr = np.asarray(tiles_all, dtype=np.int32)
        xy_arr = np.asarray(xy_all, dtype=np.int32)
        kinds_arr = np.asarray(kinds_all, dtype=np.int8)
        from_arr = np.asarray(from_all, dtype=np.int32)
        ov_arr = np.asarray(ov_all, dtype=np.int32)
        mask = np.ones((A,), dtype=bool)

        return CandidateBatch(
            tiles=tiles_arr,
            xy=xy_arr,
            kinds=kinds_arr,
            from_tiles=from_arr,
            ov_cells=ov_arr,
            idx_per_tile=idx_per_tile,
            mask=mask,
        )

    # ------------- Step -------------

    def step(self, tile_id: int, x: int, y: int) -> Dict:
        """
        Apply a placement if feasible. Returns info dict suitable for RL:
          - feasible (bool)
          - overlap_used (int)
          - delta_bbox_area (int)
          - done (bool)
        """
        feasible, ov = self._check_feasible(tile_id, x, y)
        if not feasible:
            return {"feasible": False, "overlap_used": 0, "delta_bbox_area": 0, "done": self.done()}

        pre_area = _bbox_area(self.bbox)
        self._apply_place(tile_id, x, y)
        post_area = _bbox_area(self.bbox)
        return {
            "feasible": True,
            "overlap_used": ov,
            "delta_bbox_area": post_area - pre_area,
            "done": self.done(),
        }

    # ------------- Raster crop -------------

    def rasterize_occ_crop(
        self,
        max_hw: Optional[int] = None,
        pad: int = 2,
        center: str = "bbox",
        dtype=np.float32,
    ) -> torch.Tensor:
        """
        Produce a (C_occ, H, W) raster one-hot including an 'empty' channel.
        Channels = K symbols + 1 empty. Values in {0,1}.
        - If bbox is None (empty), returns a small empty canvas.
        - If max_hw is set, crop to at most max_hw x max_hw (centered on bbox).
        - pad adds zeros around bbox to let CNN see context.
        """
        if self.bbox is None:
            H = W = (max_hw or (self.n + 2 * pad))
            return torch.zeros((self.K + 1, H, W), dtype=torch.float32)

        (x0, y0, x1, y1) = self.bbox
        x0c, y0c = x0 - pad, y0 - pad
        x1c, y1c = x1 + pad, y1 + pad
        W_full = x1c - x0c + 1
        H_full = y1c - y0c + 1

        # Optionally crop further to max_hw
        if max_hw is not None:
            Hc = min(H_full, max_hw)
            Wc = min(W_full, max_hw)
            # Center crop around bbox center
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            x0c = cx - Wc // 2
            y0c = cy - Hc // 2
            x1c = x0c + Wc - 1
            y1c = y0c + Hc - 1
            H_full, W_full = Hc, Wc

        # Build channels
        raster = np.zeros((self.K + 1, H_full, W_full), dtype=np.float32)
        # Fill symbol channels
        for (ax, ay), s in self.occ_index.items():
            if x0c <= ax <= x1c and y0c <= ay <= y1c:
                ii = ay - y0c
                jj = ax - x0c
                raster[s, ii, jj] = 1.0

        # Empty channel = 1 where no symbol
        occ_any = raster[1:].sum(axis=0)  # (H, W)
        raster[0] = (occ_any == 0).astype(np.float32)
        return torch.from_numpy(raster.astype(dtype, copy=False))

    # ------------- Graph view -------------

    def get_layout_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (nodes, edges) for GNN:
          nodes: [M, 3]  -> (tile_id, x, y)
          edges: [E, 5]  -> (src_tile, dst_tile, dx, dy, ov_cells)
        Edges connect tiles that share >=1 overlapping cell (equal symbol).
        """
        if not self.placed:
            return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 5), dtype=np.int32)

        nodes = []
        for tid, (x, y) in self.placed.items():
            nodes.append((tid, x, y))
        nodes = np.asarray(nodes, dtype=np.int32)

        # Build edges by scanning overlaps between each pair via symbol matches.
        placed_ids = list(self.placed.keys())
        edges = []
        for i in range(len(placed_ids)):
            ti = placed_ids[i]
            xi, yi = self.placed[ti]
            t_i = self.tiles[ti]
            # Build set of absolute occupied cells for ti
            occ_i = set()
            for s, coords in t_i.nz_by_symbol.items():
                abs_xy = coords + np.array([yi, xi], dtype=np.int32)
                for ij in abs_xy:
                    occ_i.add((int(ij[1]), int(ij[0])))

            for j in range(i + 1, len(placed_ids)):
                tj = placed_ids[j]
                xj, yj = self.placed[tj]
                t_j = self.tiles[tj]
                # Count equal-symbol overlaps by intersecting absolute coords
                ov = 0
                for s, coords in t_j.nz_by_symbol.items():
                    abs_xy = coords + np.array([yj, xj], dtype=np.int32)
                    for ij in abs_xy:
                        key = (int(ij[1]), int(ij[0]))
                        if key in occ_i and self.occ_index.get(key, 0) == s:
                            ov += 1
                if ov > 0:
                    dx = xj - xi
                    dy = yj - yi
                    edges.append((ti, tj, dx, dy, ov))
                    edges.append((tj, ti, -dx, -dy, ov))

        edges = np.asarray(edges, dtype=np.int32) if edges else np.zeros((0,5), dtype=np.int32)
        return nodes, edges


# -------------------------------
# Public helpers for training
# -------------------------------

def make_synthetic_tiles(n: int, K: int, num_tiles: int, p_fill: float = 0.25, seed: int = 0) -> List[np.ndarray]:
    """
    Utility to synthesize random n x n tiles with integer symbols in [0..K].
    p_fill controls fraction of non-zero cells; non-zero cells pick uniform symbol in [1..K].
    """
    rng = np.random.RandomState(seed)
    tiles = []
    for _ in range(num_tiles):
        mask = (rng.rand(n, n) < p_fill)
        arr = np.zeros((n, n), dtype=np.int32)
        arr[mask] = rng.randint(1, K + 1, size=mask.sum(), dtype=np.int32)
        tiles.append(arr)
    return tiles


def build_step_batch_from_env(
    env: TilePlacementEnv,
    max_candidates: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Create tensors the model can consume directly:
      Returns dict with keys:
        - cand_tiles:    [A] int64 (tile ids)
        - cand_xy:       [A,2] int32
        - cand_kind:     [A] int8 (0 overlap, 1 adjacency)
        - cand_from:     [A] int32 (source placed tile or -1)
        - cand_ov:       [A] int32
        - cand_mask:     [A] bool
        - tile_idx_ptr:  [T+1] int32 prefix sums for per-tile slicing (ordered by env.remaining)
        - remaining:     [T] int32 tile ids in this step
        - occ_crop:      (C_occ,H,W) float32 tensor (already on device if device set)
        - layout_nodes:  [M,3] int32
        - layout_edges:  [E,5] int32
    """
    cb = env.enumerate_candidates()

    # Optionally subsample candidates globally for speed (e.g., during warmup)
    if max_candidates is not None and cb.tiles.shape[0] > max_candidates:
        # simple uniform downsample, but keep at least one per tile if possible
        keep_mask = np.zeros(cb.tiles.shape[0], dtype=bool)
        for v, (s, e) in cb.idx_per_tile.items():
            span = e - s
            if span <= 0: continue
            # always keep one deterministic index per tile
            keep_mask[s] = True

        remain_idx = np.where(~keep_mask)[0]
        need = max(0, max_candidates - keep_mask.sum())
        if need > 0 and remain_idx.size > 0:
            pick = np.random.choice(remain_idx, size=min(need, remain_idx.size), replace=False)
            keep_mask[pick] = True

        sel = np.where(keep_mask)[0]
        tiles = cb.tiles[sel]
        xy = cb.xy[sel]
        kinds = cb.kinds[sel]
        froms = cb.from_tiles[sel]
        ovs = cb.ov_cells[sel]
        mask = cb.mask[sel]
        # Rebuild idx_per_tile on the fly (dense rebuild by scanning)
        idx_per_tile = {}
        cursor = 0
        for v in env.remaining:
            # collect indices for this v
            vids = np.where(tiles == v)[0]
            start = cursor
            if vids.size > 0:
                # keep order
                vids_sorted = np.sort(vids)
                tiles[cursor:cursor+vids_sorted.size] = tiles[vids_sorted]
                xy[cursor:cursor+vids_sorted.size] = xy[vids_sorted]
                kinds[cursor:cursor+vids_sorted.size] = kinds[vids_sorted]
                froms[cursor:cursor+vids_sorted.size] = froms[vids_sorted]
                ovs[cursor:cursor+vids_sorted.size] = ovs[vids_sorted]
                mask[cursor:cursor+vids_sorted.size] = mask[vids_sorted]
                cursor += vids_sorted.size
            idx_per_tile[v] = (start, cursor)

        # final trim
        tiles = tiles[:cursor]; xy = xy[:cursor]; kinds = kinds[:cursor]
        froms = froms[:cursor]; ovs = ovs[:cursor]; mask = mask[:cursor]
    else:
        tiles = cb.tiles; xy = cb.xy; kinds = cb.kinds
        froms = cb.from_tiles; ovs = cb.ov_cells; mask = cb.mask
        idx_per_tile = cb.idx_per_tile

    # Build prefix pointers in the order of env.remaining
    ptr = [0]
    acc = 0
    for v in env.remaining:
        s, e = idx_per_tile.get(v, (acc, acc))
        count_v = e - s
        acc += count_v
        ptr.append(acc)
    tile_idx_ptr = np.asarray(ptr, dtype=np.int32)

    # Observations
    occ_crop = env.rasterize_occ_crop()
    nodes, edges = env.get_layout_graph()

    # To tensors
    todev = (lambda x: torch.as_tensor(x, device=device)) if device is not None else torch.as_tensor

    batch = {
        "cand_tiles": todev(tiles.astype(np.int64)),
        "cand_xy": todev(xy.astype(np.int32)),
        "cand_kind": todev(kinds.astype(np.int8)),
        "cand_from": todev(froms.astype(np.int32)),
        "cand_ov": todev(ovs.astype(np.int32)),
        "cand_mask": todev(mask.astype(np.bool_)),
        "tile_idx_ptr": todev(tile_idx_ptr),
        "remaining": todev(np.asarray(env.remaining, dtype=np.int32)),
        "occ_crop": occ_crop if device is None else occ_crop.to(device, non_blocking=True),
        "layout_nodes": todev(nodes.astype(np.int32)),
        "layout_edges": todev(edges.astype(np.int32)),
        # Also expose bbox metrics for reward/feature
        "bbox_xyxy": todev(np.asarray([-1,-1,-1,-1] if env.bbox is None else env.bbox, dtype=np.int32)),
        "bbox_area": todev(np.asarray(_bbox_area(env.bbox), dtype=np.int32)),
    }
    return batch


# -------------------------------
# Quick self-check
# -------------------------------

if __name__ == "__main__":
    # Minimal smoke test
    n = 4
    K = 3
    tiles = make_synthetic_tiles(n, K, num_tiles=6, p_fill=0.35, seed=42)
    env = TilePlacementEnv(tiles, tile_size=n, alphabet=K, seed=0)
    print("Placed initially:", env.placed)
    for step_idx in range(16):
        if env.done():
            print("Done at step", step_idx)
            break
        cb = env.enumerate_candidates()
        A = cb.tiles.shape[0]
        if A == 0:
            print("No candidates; terminating.")
            break
        # Pick a random candidate for demonstration
        pick = np.random.randint(0, A)
        v = cb.tiles[pick]
        x, y = cb.xy[pick]
        info = env.step(int(v), int(x), int(y))
        print(f"Step {step_idx}: place v={v} at {(x,y)} ->", info)
