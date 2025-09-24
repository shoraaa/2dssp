# gurobi_min_canvas.py
import math, itertools
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise RuntimeError("This module requires gurobipy. Please ensure Gurobi is installed and licensed.") from e

def _placements_for_m(T:int, n:int, m:int):
    X = range(0, m - n + 1)
    Y = range(0, m - n + 1)
    return [(x,y) for x in X for y in Y]

def _conflict(tA: np.ndarray, x1:int, y1:int, tB: np.ndarray, x2:int, y2:int) -> bool:
    # Return True if placements (tA at (x1,y1)) and (tB at (x2,y2)) overlap on at least one cell with DIFFERENT values
    n = tA.shape[0]
    # Overlap bbox in canvas
    xi0 = max(x1, x2)
    yi0 = max(y1, y2)
    xi1 = min(x1 + n - 1, x2 + n - 1)
    yi1 = min(y1 + n - 1, y2 + n - 1)
    if xi0 > xi1 or yi0 > yi1:
        return False  # no overlap at all
    # Map to local coords and compare
    for X in range(xi0, xi1 + 1):
        for Y in range(yi0, yi1 + 1):
            a = int(tA[Y - y1, X - x1])
            b = int(tB[Y - y2, X - x2])
            if a != b:
                return True
    return False  # overlapping cells (if any) are all equal

def _build_model(tiles: List[np.ndarray], m:int, time_limit: Optional[float] = None, log_to_console: bool = True):
    T = len(tiles)
    n = tiles[0].shape[0]
    positions = _placements_for_m(T, n, m)  # same domain for every tile
    # Create model
    env = None
    if not log_to_console:
        env = gp.Env(params={"OutputFlag": 0})
        model = gp.Model(env=env)
    else:
        model = gp.Model()
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.Threads = 0  # let Gurobi decide
    # Variables: p[t, x, y] ∈ {0,1}
    p = {}
    for t in range(T):
        for (x,y) in positions:
            p[(t,x,y)] = model.addVar(vtype=GRB.BINARY, name=f"p_{t}_{x}_{y}")
    model.update()
    # Each tile placed exactly once
    for t in range(T):
        model.addConstr(gp.quicksum(p[(t,x,y)] for (x,y) in positions) == 1, name=f"place_once_{t}")
    # Pairwise conflict constraints
    # For every pair of placements that conflict, forbid selecting both: p1 + p2 <= 1
    # We'll loop over (t1,(x1,y1)) and (t2,(x2,y2)) with (t1 < t2) to avoid duplicates
    for t1 in range(T):
        A = tiles[t1]
        for t2 in range(t1+1, T):
            B = tiles[t2]
            for (x1,y1) in positions:
                for (x2,y2) in positions:
                    if _conflict(A,x1,y1, B,x2,y2):
                        model.addConstr(p[(t1,x1,y1)] + p[(t2,x2,y2)] <= 1, name=f"conf_{t1}_{x1}_{y1}__{t2}_{x2}_{y2}")
    # Objective: pure feasibility; set dummy objective 0
    model.setObjective(0.0, GRB.MINIMIZE)
    model.update()
    return model, p, positions

def solve_min_canvas_gurobi(tiles: List[np.ndarray], time_limit: Optional[float] = None, log_to_console: bool = True) -> Dict[str, Any]:
    # \"\"\"Iterate m from n upward; first feasible model is optimal. Returns dict with placements and m.\"\"\"
    if len(tiles) == 0:
        return {"status":"ok","best_m":0,"placements":{}, "bbox": (0,-1,0,-1), "canvas": np.array([[]],dtype=int)}
    n = tiles[0].shape[0]
    T = len(tiles)
    LB = n
    side = math.ceil(math.sqrt(T))
    UB = side * n  # worst-case: pack tiles in a side×side grid
    best = None
    for m in range(LB, UB+1):
        model, p, positions = _build_model(tiles, m, time_limit=time_limit, log_to_console=log_to_console)
        model.optimize()
        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            if model.SolCount >= 1:
                # feasible
                placements = {}
                for t in range(T):
                    for (x,y) in positions:
                        if p[(t,x,y)].X > 0.5:
                            placements[t] = (x,y)
                            break
                # build canvas
                canvas = np.full((m,m), -1, dtype=int)
                for t,(x,y) in placements.items():
                    tile = tiles[t]
                    for i in range(n):
                        for j in range(n):
                            canvas[y+i, x+j] = int(tile[i,j])
                bbox = (0, m-1, 0, m-1)
                best = {"status":"ok","best_m":m,"placements":placements,"bbox":bbox,"canvas":canvas}
                # discard model, break
                model.dispose()
                return best
        model.dispose()
    return {"status":"failed","best_m":None,"placements":None,"bbox":None,"canvas":None}
