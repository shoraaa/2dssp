# Candidate List Caching Optimization

## Overview
This optimization significantly speeds up the tile placement process by caching and filtering candidate lists between steps instead of regenerating them from scratch each time.

## Key Insight
When a tile is placed at each step:
1. The placed tile is removed from the `remaining` tiles list
2. Some candidate positions become invalid due to occupancy conflicts
3. New candidate positions may emerge that overlap with the newly placed tile
4. **Most candidates from the previous step remain valid**

Instead of calling `generate_candidates()` repeatedly (which scans all placed tiles and remaining tiles), we can **filter and update** the existing candidate list incrementally.

## Implementation

### 1. New Method: `filter_candidates_after_placement()` (in `env.py`)
```python
def filter_candidates_after_placement(
    self,
    prev_cands: List[Tuple[int, int, int, bool, int, int]],
    placed_tile_id: int,
    placed_x: int,
    placed_y: int,
) -> List[Tuple[int, int, int, bool, int, int]]:
```

This method:
- **Filters** existing candidates by:
  - Removing candidates for the just-placed tile
  - Removing candidates that now conflict with the occupancy grid
  - Updating overlap metrics for remaining candidates (adding overlaps with the new tile)
- **Adds** new candidates that overlap with the newly placed tile

### 2. Updated `build_step_batch_from_env()` (in `env.py`)
Added `cached_cands` parameter to accept pre-filtered candidates:
```python
def build_step_batch_from_env(
    env: TilePlacementEnv,
    tile_embs: torch.Tensor,
    ...,
    cached_cands: Optional[List[Tuple[int, int, int, bool, int, int]]] = None,
) -> StepBatch:
```

If `cached_cands` is provided, it uses those; otherwise falls back to `generate_candidates()`.

### 3. Updated All Rollout/Evaluation Functions
Modified the following functions to maintain a candidate cache:
- `rollout_episode()` in `train.py`
- `evaluate_instance()` in `train.py`
- `evaluate_instance()` in `train_cnn.py`
- `rollout_episode()` in `train_cnn.py`
- `run_neural()` in `benchmark.py`

## Performance Impact

### Time Complexity Analysis
**Before:**
- `generate_candidates()` at each step: O(T × P × T) where T = num tiles, P = num placed tiles
- Called T times during episode: **O(T³)**

**After:**
- `filter_candidates_after_placement()`: O(C + T) where C = num candidates
- Called T times: **O(T × C)** which is typically much smaller since C grows slowly

### Expected Speedup
For typical problem sizes (T=30-100 tiles):
- **2-5x speedup** in candidate generation
- **20-40% overall episode speedup** (since candidate generation is a major bottleneck)

The speedup increases with:
- Larger number of tiles (T)
- Higher tile overlap density (more candidates to filter vs regenerate)

## Backward Compatibility
The changes are **fully backward compatible**:
- `cached_cands` parameter is optional and defaults to `None`
- If not provided, behavior is identical to before (calls `generate_candidates()`)
- All existing code continues to work without modification

## Usage Pattern

```python
env.reset()
cached_cands = None  # Start with no cache

while not env.done:
    # Use cached candidates (or generate on first iteration)
    sb, raw_cands = build_step_batch_from_env(
        env, tile_embs, return_cands=True, cached_cands=cached_cands
    )
    
    # ... select action ...
    action_idx = select_action(sb)
    cand = raw_cands[action_idx]
    tile_id, x, y = cand[0], cand[1], cand[2]
    
    # Place the tile
    env.step(cand)
    
    # Filter candidates for next iteration
    cached_cands = env.filter_candidates_after_placement(
        raw_cands, tile_id, x, y
    )
```

## Testing
After applying this optimization, verify:
1. ✓ Training runs complete successfully
2. ✓ Evaluation produces same results as before (deterministic with same seed)
3. ✓ Benchmark scores remain consistent
4. ✓ Speedup is measurable in training logs

## Files Modified
- `env.py`: Added `filter_candidates_after_placement()` method, updated `build_step_batch_from_env()`
- `train.py`: Updated `rollout_episode()` and `evaluate_instance()`
- `train_cnn.py`: Updated `rollout_episode()` and `evaluate_instance()`
- `benchmark.py`: Updated `run_neural()`
