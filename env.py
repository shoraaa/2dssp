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
    
    Args:
        tile: The tile to place (n×n array)
        x, y: Top-left coordinates for placement
        occ: Current occupancy grid mapping coordinates to tile values
        
    Returns:
        True if placement is feasible (no conflicts), False otherwise
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
    
    Args:
        tile: The tile to write (n×n array)
        x, y: Top-left coordinates for placement
        occ: Occupancy grid to modify in-place
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
    
    Args:
        x, y: Top-left coordinates of the tile
        n: Size of the tile (n×n)
        bbox: Bounding box [xmin, xmax, ymin, ymax] to update in-place
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
    
    Args:
        x, y: Top-left coordinates of the proposed tile
        n: Size of the tile (n×n)
        bbox: Current bounding box [xmin, xmax, ymin, ymax]
        
    Returns:
        Increase in maximum dimension (0 if no increase)
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
    
    Args:
        placements: Dictionary mapping tile IDs to their (x, y) positions
        n: Size of each tile (n×n)
        
    Returns:
        Tuple of (max_dimension, [xmin, xmax, ymin, ymax])
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
    Generate adjacency offset positions for a tile of size n×n.
    
    Args:
        n: Size of the tile
        
    Returns:
        List of (dx, dy) offsets for adjacent positions (4n+4 total)
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
    
    For each pair of tiles (u, v), finds all relative positions (dx, dy) where
    tile v can overlap with tile u such that the overlapping regions match exactly.
    
    Args:
        tiles: List of n×n tile arrays
        n: Size of each tile
        
    Returns:
        Dictionary mapping (tile_u, tile_v) to list of (dx, dy, overlap_size) tuples,
        sorted by overlap_size in descending order
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
    
    Args:
        T: Number of tiles to generate
        n: Size of each tile (n×n)
        alphabet: Number of different symbols (0 to alphabet-1)
        seed: Random seed for reproducibility
        
    Returns:
        List of T tiles, each as an n×n numpy array
    """
    rng = np.random.RandomState(seed)
    
    # Generate all tiles at once for better performance
    all_tiles = rng.randint(0, alphabet, size=(T, n, n), dtype=np.int32)
    
    # Convert to list of individual tiles
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
    
    Args:
        occ: Occupancy grid mapping (x, y) coordinates to symbol values
        bbox: Bounding box [xmin, xmax, ymin, ymax] of current layout
        alphabet: Number of different symbols
        pad: Number of tiles to pad around the bounding box
        max_hw: Optional maximum height/width for the output tensor
        
    Returns:
        Tensor of shape (C_occ, H, W) where C_occ = alphabet + 1 (for empty cells).
        The last channel represents empty cells, other channels are one-hot symbol encoding.
    """
    xmin, xmax, ymin, ymax = bbox
    xmin_c = xmin - pad
    ymin_c = ymin - pad
    xmax_c = xmax + pad
    ymax_c = ymax + pad

    H = ymax_c - ymin_c + 1  # along y
    W = xmax_c - xmin_c + 1  # along x

    if max_hw is not None:
        # Optional cap (simple clamp; we skip writing pixels outside the capped window)
        H = min(H, max_hw)
        W = min(W, max_hw)

    C = alphabet + 1  # extra channel for empty
    arr = np.zeros((C, H, W), dtype=np.float32)
    arr[-1, :, :] = 1.0  # empty channel = 1.0 by default

    # Write occupied cells: index as (channel, y, x) = (C, H, W)
    for (X, Y), val in occ.items():
        # skip if outside crop bounds
        if X < xmin_c or X > xmax_c or Y < ymin_c or Y > ymax_c:
            continue
        i = X - xmin_c  # x offset
        j = Y - ymin_c  # y offset
        # guard in case max_hw clipped the array
        if 0 <= j < H and 0 <= i < W:
            arr[-1, j, i] = 0.0
            arr[val, j, i] = 1.0

    return torch.from_numpy(arr)


# ===== Main Environment Class =====

@dataclass
class StepBatch:
    """
    Data structure for a batched step in the tile placement environment.
    
    Attributes:
        occ: Occupancy tensor (B, C_occ, H, W) - visual state of the layout
        tiles_left: Embeddings of remaining tiles (B, M, d_tile)  
        tiles_left_mask: Mask for valid tiles (B, M) - True for available tiles
        cand_feats: Features for each candidate action (B, A, F)
        cand_mask: Mask for valid candidates (B, A) - True for feasible actions
        cand_tile_idx: Which tile each candidate places (B, A) - indices into tiles_left
        expert_action: Optional expert action labels (B,) for supervised learning
    """
    occ: torch.Tensor              
    tiles_left: torch.Tensor       
    tiles_left_mask: torch.Tensor  
    cand_feats: torch.Tensor       
    cand_mask: torch.Tensor        
    cand_tile_idx: torch.Tensor    
    expert_action: Optional[torch.Tensor] = None

class TilePlacementEnv:
    """
    Environment for single-instance tile placement problem.
    
    The goal is to place all tiles on a canvas such that:
    1. Overlapping regions between tiles have matching values
    2. The overall canvas size (maximum dimension) is minimized
    
    The environment uses an overlap-first strategy: for each remaining tile,
    it first tries positions where it overlaps with already placed tiles.
    If no overlap positions are feasible, it falls back to adjacent positions.
    """
    
    def __init__(self, tiles: List[np.ndarray], alphabet: int):
        """
        Initialize the environment.
        
        Args:
            tiles: List of n×n tile arrays, each containing symbols in range [0, alphabet-1]
            alphabet: Number of different symbols used in tiles
        """
        assert len(tiles) >= 1
        self.tiles = tiles
        self.n = tiles[0].shape[0]  # Tile size (assuming square tiles)
        self.T = len(tiles)         # Number of tiles
        self.alphabet = alphabet

        # Precompute overlap relationships between all pairs of tiles
        self.pre_all = compute_pairwise_overlaps(tiles, self.n)
        
        # Precompute adjacent position offsets for efficient candidate generation
        self.adj_offs = _adjacent_offsets(self.n)

        # Initialize state
        self.reset()

    def reset(self, start_tile: int = 0):
        """
        Reset the environment to initial state with one tile placed.
        
        Args:
            start_tile: Index of the tile to place first (at origin)
            
        Returns:
            Current state dictionary
        """
        n = self.n
        self.placements: Dict[int, Tuple[int, int]] = {start_tile: (0, 0)}
        self.placed_ids: List[int] = [start_tile]
        self.remaining: List[int] = [i for i in range(self.T) if i != start_tile]
        self.occ: Dict[Tuple[int, int], int] = {}
        
        # Initialize occupancy with the starting tile
        write_to_occupancy(self.tiles[start_tile], 0, 0, self.occ)
        
        # Initialize bounding box
        self.bbox: List[int] = [0, n-1, 0, n-1]
        self.done = False
        return self.state()

    def spawn_clone(self) -> "TilePlacementEnv":
        """
        Create a lightweight copy that shares immutable precomputed data.
        
        Returns:
            New environment instance with shared precomputations
        """
        clone = TilePlacementEnv.__new__(TilePlacementEnv)
        clone.tiles = self.tiles
        clone.n = self.n
        clone.T = self.T
        clone.alphabet = self.alphabet
        clone.pre_all = self.pre_all  # Shared precomputed overlaps
        clone.adj_offs = self.adj_offs  # Shared adjacency offsets
        clone.reset()
        return clone

    def state(self):
        """
        Get current state as a dictionary.
        
        Returns:
            Dictionary containing placements, placed_ids, remaining tiles, and bbox
        """
        return {
            "placements": dict(self.placements),
            "placed_ids": list(self.placed_ids),
            "remaining": list(self.remaining),
            "bbox": list(self.bbox),
        }

    def generate_candidates(self) -> List[Tuple[int, int, int, bool, int]]:
        """
        Generate all feasible candidate actions for the current state.
        
        Uses overlap-first strategy: for each remaining tile, first tries positions
        where it overlaps with placed tiles. If no overlap positions work,
        falls back to adjacent positions.
        
        Returns:
            List of tuples (tile_id, x, y, is_adjacent, best_overlap_size)
            where (x, y) is the top-left position for placing the tile
        """
        cands: List[Tuple[int, int, int, bool, int]] = []
        n = self.n
        occ = self.occ

        for v in self.remaining:
            # First, try overlap positions
            overlap_positions = set()
            for u in self.placed_ids:
                ux, uy = self.placements[u]
                for (dx, dy, _) in self.pre_all.get((u, v), []):
                    overlap_positions.add((ux + dx, uy + dy))

            feasible_overlap_found = False

            # Check overlap candidates for feasibility
            for (x, y) in overlap_positions:
                if not feasible_on_occupancy(self.tiles[v], x, y, occ):
                    continue
                    
                # Calculate best overlap size among all parent tiles
                best_ov = 0
                for u in self.placed_ids:
                    ux, uy = self.placements[u]
                    dx, dy = x - ux, y - uy
                    for (dx0, dy0, ov) in self.pre_all.get((u, v), []):
                        if dx0 == dx and dy0 == dy:
                            if ov > best_ov: 
                                best_ov = ov
                            break
                
                cands.append((v, x, y, False, best_ov))
                feasible_overlap_found = True

            # If no overlap placements work, try adjacent positions  
            if not feasible_overlap_found:
                adj_positions = set()
                for u in self.placed_ids:
                    ux, uy = self.placements[u]
                    for (dx, dy) in self.adj_offs:
                        adj_positions.add((ux + dx, uy + dy))

                for (x, y) in adj_positions:
                    if not feasible_on_occupancy(self.tiles[v], x, y, occ):
                        continue
                    # Adjacent placements have no overlap by definition
                    cands.append((v, x, y, True, 0))

        return cands

    def step(self, cand: Tuple[int, int, int, bool, int]):
        """
        Apply a candidate action to the environment.
        
        Args:
            cand: Tuple (tile_id, x, y, is_adjacent, best_overlap) representing the action
            
        Returns:
            Updated state dictionary
        """
        v, x, y, is_adj, best_ov = cand
        assert v in self.remaining, f"Tile {v} is not in remaining tiles"
        
        # Update occupancy grid and bounding box
        write_to_occupancy(self.tiles[v], x, y, self.occ)
        update_bbox_for_tile(x, y, self.n, self.bbox)
        
        # Update placement tracking
        self.placements[v] = (x, y)
        self.placed_ids.append(v)
        self.remaining.remove(v)
        
        # Check if all tiles are placed
        if len(self.placed_ids) == self.T:
            self.done = True
            
        return self.state()

# ===== Step Batch Construction =====

def build_step_batch_from_env(
    env: TilePlacementEnv,
    tile_embs: torch.Tensor,          
    raster_pad: int = 1,
    max_crop_hw: Optional[int] = None,
) -> StepBatch:
    """
    Convert environment state into a StepBatch for neural network processing.
    
    This function creates feature representations for:
    1. Visual state (occupancy raster) 
    2. Remaining tiles (embeddings)
    3. Candidate actions (position and overlap features)
    
    Args:
        env: The tile placement environment 
        tile_embs: Precomputed tile embeddings (T, d_tile)
        raster_pad: Padding around bounding box for rasterization
        max_crop_hw: Optional max height/width for rasterization
        
    Returns:
        StepBatch with batch size 1, containing all necessary data for model inference
        
    Candidate features include:
        - H_sum: Total overlap size with all parent tiles
        - delta_m: Increase in maximum canvas dimension  
        - dx_norm, dy_norm: Normalized position relative to current layout
        - is_adj: Whether this is an adjacency (vs overlap) placement
        - best_ov: Best overlap size among parent tiles
        - touches_*: Which sides of current layout this tile would touch
    """
    assert not env.done, "Episode already finished."

    # Rasterize current occupancy state
    occ = rasterize_occ_crop(env.occ, env.bbox, alphabet=env.alphabet, pad=raster_pad, max_hw=max_crop_hw)  
    occ = occ.unsqueeze(0)  # Add batch dimension: (1, C_occ, H, W)

    # Prepare remaining tile embeddings
    remaining = env.remaining
    M = len(remaining)
    d_tile = tile_embs.size(-1)
    tiles_left = tile_embs[remaining].unsqueeze(0)  # (1, M, d_tile)
    tiles_left_mask = torch.ones(1, M, dtype=torch.bool)  # All remaining tiles are valid

    # Generate and featurize candidates
    raw_cands = env.generate_candidates()
    if len(raw_cands) == 0:
        # Edge case: no valid moves (should be rare)
        return StepBatch(
            occ=occ,
            tiles_left=tiles_left,
            tiles_left_mask=tiles_left_mask,
            cand_feats=torch.zeros(1, 1, 10),  # 10 features as documented above
            cand_mask=torch.zeros(1, 1, dtype=torch.bool),
            cand_tile_idx=torch.zeros(1, 1, dtype=torch.long),
            expert_action=None,
        )

    # Compute candidate features
    n = env.n
    xmin, xmax, ymin, ymax = env.bbox
    old_m = max(xmax-xmin+1, ymax-ymin+1)

    feats: List[List[float]] = []
    tile_idx: List[int] = []
    mask: List[bool] = []

    # Pre-compute H_sum (total overlap) for each candidate
    Hs: List[float] = []
    for (v, x, y, is_adj, best_ov) in raw_cands:
        H_sum = 0
        for u in env.placed_ids:
            ux, uy = env.placements[u]
            dx, dy = x - ux, y - uy
            for (dx0, dy0, ov) in env.pre_all.get((u, v), []):
                if dx0 == dx and dy0 == dy:
                    H_sum += ov
                    break
        Hs.append(float(H_sum))

    # Build feature vectors for each candidate
    for idx, (v, x, y, is_adj, best_ov) in enumerate(raw_cands):
        # Canvas size increase if this tile is placed
        inc = bbox_increase_if_place(x, y, n, env.bbox)
        
        # Normalized position relative to current layout
        dxn = (x - xmin) / max(1.0, old_m)
        dyn = (y - ymin) / max(1.0, old_m)
        
        # Which sides of the current layout this tile would touch
        left_touch  = 1.0 if x <= xmin else 0.0
        right_touch = 1.0 if (x + n - 1) >= xmax else 0.0
        up_touch    = 1.0 if y <= ymin else 0.0
        down_touch  = 1.0 if (y + n - 1) >= ymax else 0.0

        feats.append([
            Hs[idx],        # Total overlap size
            float(inc),     # Canvas size increase  
            float(dxn),     # Normalized x position
            float(dyn),     # Normalized y position
            float(is_adj),  # Is adjacency placement (vs overlap)
            float(best_ov), # Best overlap with any parent
            left_touch,     # Touches left edge
            right_touch,    # Touches right edge  
            up_touch,       # Touches top edge
            down_touch      # Touches bottom edge
        ])
        
        # Map tile ID to its index in the remaining tiles list
        tile_idx.append(remaining.index(v))
        mask.append(True)

    # Convert to tensors
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

# ===== Tile Embedding Utilities =====

def precompute_tile_embeddings(tiles: List[np.ndarray], alphabet: int, tile_cnn: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """
    Convert tiles to one-hot representation and compute embeddings using a CNN.
    
    This function:
    1. Converts integer tile arrays to one-hot encoding by symbol
    2. Passes them through the tile CNN to get embeddings
    3. Returns embeddings on CPU for storage
    
    Args:
        tiles: List of n×n integer arrays representing tiles
        alphabet: Number of different symbols (0 to alphabet-1) 
        tile_cnn: Neural network to compute tile embeddings
        device: Device to run computations on
        
    Returns:
        Tensor of shape (T, d_tile) containing tile embeddings on CPU
    """
    T = len(tiles)
    if T == 0:
        return torch.empty(0, 0, device='cpu')
        
    n = tiles[0].shape[0]
    
    # Stack tiles and convert to one-hot encoding
    X = np.stack(tiles, axis=0)  # (T, n, n)
    oh = np.zeros((T, alphabet, n, n), dtype=np.float32)
    for t in range(T):
        oh[t, X[t], np.arange(n)[:,None], np.arange(n)[None,:]] = 1.0
    
    # Compute embeddings
    x = torch.from_numpy(oh)  # (T, C_sym, n, n)
    with torch.no_grad():
        x = x.to(device)
        emb = tile_cnn(x)      # (T, d_tile)
    return emb.cpu()
