
import numpy as np
import random
import json
from typing import Tuple, Dict, List, Any, Optional
from pathlib import Path
import time
import math

# ------------------ Single-instance generators (backward compatible) ------------------

def generate_random_tiles(
    n: int,
    num_tiles: int,
    alphabet_size: int = 2,
    rng_seed: int = 123,
) -> Dict[str, Any]:
    """
    Create a set of truly random n×n tiles and find their optimal arrangement
    using the ground truth brute-force solver.

    Returns a dict with fields including 'optimal_m' and metadata 'gt_source'='bruteforce'.
    """
    assert n >= 1 and num_tiles >= 1
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # Generate completely random tiles
    tiles: List[np.ndarray] = []
    for t in range(num_tiles):
        tile = np.random.randint(0, alphabet_size, size=(n, n), dtype=int)
        tiles.append(tile)

    # Use ground truth solver to find optimal arrangement
    from exact import exact_bruteforce_min_canvas
    gt_start = time.time()
    result = exact_bruteforce_min_canvas(tiles)
    gt_time = time.time() - gt_start
    if result.get("status") != "ok":
        raise RuntimeError("Ground-truth solver failed on random tiles.")

    best_m = int(result["best_m"])
    placements = result["placements"]

    # Compute bbox
    if placements:
        xs = [placements[k][0] for k in placements]
        ys = [placements[k][1] for k in placements]
        xmin, ymin = min(xs), min(ys)
        xmax = max(placements[k][0] + n - 1 for k in placements)
        ymax = max(placements[k][1] + n - 1 for k in placements)
        bbox = [xmin, xmax, ymin, ymax]

        # Build canvas (square m×m, safe fill)
        canvas = np.full((best_m, best_m), -1, dtype=int)
        for tid, (x, y) in placements.items():
            tile = tiles[tid]
            for i in range(n):
                for j in range(n):
                    canvas[y + j, x + i] = int(tile[i, j])
    else:
        bbox = [0, -1, 0, -1]
        canvas = np.array([[]], dtype=int)

    return {
        "tiles": tiles,
        "true_placements": placements,
        "true_bbox": bbox,
        "true_canvas": canvas,
        "optimal_m": best_m,
        "n": n,
        "alphabet_size": alphabet_size,
        "canvas_size": (best_m, best_m),
        "gt_time": gt_time,
        "gt_source": "bruteforce",
    }


def generate_tile_set(
    n: int,
    num_tiles: int,
    canvas_size: Tuple[int, int] = None,
    alphabet_size: int = 2,
    min_overlap: int = None,
    rng_seed: int = 123,
) -> Dict[str, Any]:
    """
    Create a random connected set of n×n tiles by sampling crops from a hidden canvas.
    The ground-truth canvas is the hidden canvas; the resulting m from placements' bbox
    is not necessarily the minimal possible across all permutations, so we flag gt_source='hidden'.
    """
    assert n >= 1 and num_tiles >= 1
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    if canvas_size is None:
        side = max(n + 2, int(round(np.sqrt(num_tiles) * (n - max(1, n//3)))) + n)
        H = W = side
    else:
        H, W = canvas_size

    canvas = np.random.randint(0, alphabet_size, size=(H, W), dtype=int)

    def overlap_area(ax, ay, bx, by):
        x_overlap = max(0, min(ax+n, bx+n) - max(ax, bx))
        y_overlap = max(0, min(ay+n, by+n) - max(ay, by))
        return x_overlap * y_overlap

    placements: Dict[int, Tuple[int, int]] = {}
    xs = list(range(0, W - n + 1))
    ys = list(range(0, H - n + 1))
    if min_overlap is None:
        min_overlap = max(1, (n * n) // 6)

    # first tile anywhere
    x0 = random.choice(xs); y0 = random.choice(ys)
    placements[0] = (x0, y0)
    xmin, ymin = x0, y0
    xmax, ymax = x0 + n - 1, y0 + n - 1

    # subsequent tiles must overlap the current region by at least min_overlap
    for t in range(1, num_tiles):
        placed = False
        attempts = 0
        max_attempts = 2000
        while attempts < max_attempts and not placed:
            attempts += 1
            rx_min = max(0, xmin - (n - 1))
            rx_max = min(W - n, xmax)
            ry_min = max(0, ymin - (n - 1))
            ry_max = min(H - n, ymax)
            if rx_min > rx_max or ry_min > ry_max:
                x = random.choice(xs); y = random.choice(ys)
            else:
                x = random.randint(rx_min, rx_max)
                y = random.randint(ry_min, ry_max)

            total_overlap = 0
            for _, (px, py) in placements.items():
                total_overlap = max(total_overlap, overlap_area(x, y, px, py))
            if total_overlap >= min_overlap:
                placed = True
                placements[t] = (x, y)
                xmin = min(xmin, x); ymin = min(ymin, y)
                xmax = max(xmax, x + n - 1); ymax = max(ymax, y + n - 1)

        if not placed:
            x = random.choice(xs); y = random.choice(ys)
            placements[t] = (x, y)
            xmin = min(xmin, x); ymin = min(ymin, y)
            xmax = max(xmax, x + n - 1); ymax = max(ymax, y + n - 1)

    tiles: List[np.ndarray] = []
    for t in range(num_tiles):
        x, y = placements[t]
        tiles.append(canvas[y:y+n, x:x+n].copy())

    m_gt = max(xmax - xmin + 1, ymax - ymin + 1)

    return {
        "tiles": tiles,
        "true_canvas": canvas,
        "true_placements": placements,
        "true_bbox": [xmin, xmax, ymin, ymax],
        "n": n,
        "alphabet_size": alphabet_size,
        "canvas_size": (H, W),
        "optimal_m": m_gt,
        "gt_time": 0.0,
        "gt_source": "hidden",
    }

def make_synthetic_tiles(T: int, n: int, alphabet: int, seed: Optional[int]=0) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    tiles = []
    for _ in range(T):
        tiles.append(rng.randint(0, alphabet, size=(n,n), dtype=np.int64))
    return {
        "tiles": tiles,
    }

# ------------------ Batch helpers ------------------

def _save_npz_dataset(path: Path, pack: Dict[str, Any]) -> None:
    tiles = np.stack(pack["tiles"], axis=0)
    bbox = np.array(pack["true_bbox"], dtype=int)
    placements = pack["true_placements"]
    plc_items = np.array([[k, placements[k][0], placements[k][1]] for k in sorted(placements.keys())], dtype=int)
    H, W = pack["canvas_size"]
    HW = np.array([H, W], dtype=int)
    n = np.array([pack["n"]], dtype=int)
    alphabet = np.array([pack["alphabet_size"]], dtype=int)
    gt_time = np.array([float(pack.get("gt_time", 0.0))])
    optimal_m = np.array([int(pack.get("optimal_m", 0))])
    gt_source = np.array([str(pack.get("gt_source", "unknown"))])

    params = {
        "n": int(pack["n"]),
        "num_tiles": int(tiles.shape[0]),
        "alphabet_size": int(pack["alphabet_size"]),
        "canvas_size": tuple(pack.get("canvas_size") or (0,0)),
        "gt_source": str(pack.get("gt_source", "unknown")),
    }
    params_json = np.array([json.dumps(params)])

    np.savez_compressed(
        path,
        tiles=tiles, bbox=bbox, placements=plc_items, true_canvas=pack.get("true_canvas"),
        HW=HW, n=n, alphabet=alphabet, gt_time=gt_time, optimal_m=optimal_m,
        gt_source=gt_source, params=params_json
    )

def make_dataset_name(prefix: str, mode: str, n: int, T: int, A: int, seed: int) -> str:
    return f"{prefix}_{mode}_n{n}_T{T}_A{A}_seed{seed}.npz"

def make_synthetic_tiles(T: int, n: int, alphabet: int, seed: Optional[int]=0) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    tiles = []
    for _ in range(T):
        tiles.append(rng.randint(0, alphabet, size=(n,n), dtype=np.int64))
    return tiles


def generate_synthetic_tiles(
    n: int,
    num_tiles: int,
    alphabet_size: int = 2,
    rng_seed: int = 123,
) -> Dict[str, Any]:
    """
    Create a set of synthetic n×n tiles using make_synthetic_tiles.
    This creates a bare dataset without computing optimal placements.
    
    Returns a dict with minimal required fields for dataset storage.
    """
    assert n >= 1 and num_tiles >= 1
    
    # Generate tiles using make_synthetic_tiles
    tiles = make_synthetic_tiles(T=num_tiles, n=n, alphabet=alphabet_size, seed=rng_seed)

    # Create minimal placeholder values since we're not computing optimal solution
    # Empty placements and bbox
    placements = {}
    bbox = [0, -1, 0, -1]  # Empty bbox
    canvas = np.array([[]], dtype=int)  # Empty canvas

    return {
        "tiles": tiles,
        "true_placements": placements,
        "true_bbox": bbox,
        "true_canvas": canvas,
        "optimal_m": 0,  # Placeholder since we don't compute optimal
        "n": n,
        "alphabet_size": alphabet_size,
        "canvas_size": (0, 0),  # Placeholder
        "gt_time": 0.0,
        "gt_source": "synthetic",
    }


def generate_datasets(
    out_dir: str,
    mode: str,
    n: int,
    num_tiles: int,
    alphabet_size: int = 2,
    canvas_m: Optional[int] = None,
    min_overlap: Optional[int] = None,
    count: int = 5,
    base_seed: int = 123,
    prefix: str = "ds"
) -> List[str]:
    """
    Generate 'count' datasets with identical parameters (different seeds) and save them under out_dir.
    mode: 'random' (bruteforce GT), 'hidden' (connected from hidden canvas), or 'synthetic' (bare synthetic tiles).
    Returns list of file paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        seed = base_seed + i
        if mode == "random":
            pack = generate_random_tiles(n=n, num_tiles=num_tiles, alphabet_size=alphabet_size, rng_seed=seed)
        elif mode == "hidden":
            pack = generate_tile_set(
                n=n, num_tiles=num_tiles,
                canvas_size=(canvas_m, canvas_m) if canvas_m else None,
                alphabet_size=alphabet_size, min_overlap=min_overlap, rng_seed=seed
            )
        elif mode == "synthetic":
            pack = generate_synthetic_tiles(n=n, num_tiles=num_tiles, alphabet_size=alphabet_size, rng_seed=seed)
        else:
            raise ValueError("mode must be 'random', 'hidden', or 'synthetic'")

        fname = make_dataset_name(prefix, mode, n, num_tiles, alphabet_size, seed)
        fpath = Path(out_dir) / fname
        _save_npz_dataset(fpath, pack)
        paths.append(str(fpath))
    return paths
