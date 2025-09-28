#!/usr/bin/env python3
"""
test.py - Test the neural model with make_dataset and compare with baseline methods.
"""

import argparse
import time
import random
import numpy as np
import torch
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import modules from the project
from train import make_dataset, evaluate_instance
from net import NeuralSolver, TileCNN
from env import precompute_tile_embeddings, make_synthetic_tiles, TilePlacementEnv, layout_bbox
from greedy import greedy_place_once, GreedyParams
from aco import solve_with_aco, ACOParams
from exact import exact_bruteforce_min_canvas


def load_neural_model(checkpoint_path: str, alphabet: int, device: torch.device,
                      d_tile: int = 64, d_model: int = 128, cand_feat_dim: int = 10) -> tuple:
    """Load neural model and tile CNN from checkpoint."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading neural model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Create models
    model = NeuralSolver(
        c_occ=alphabet + 1,
        d_tile=d_tile,
        d_model=d_model,
        cand_feat_dim=cand_feat_dim,
    ).to(device)
    
    tile_cnn = TileCNN(in_ch=alphabet, d_tile=d_tile).to(device)
    
    # Load model state
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # Try loading directly
        model.load_state_dict(ckpt)
    
    # Load tile CNN state
    if "tile_cnn" in ckpt:
        tile_cnn.load_state_dict(ckpt["tile_cnn"])
    elif "tilecnn" in ckpt:
        tile_cnn.load_state_dict(ckpt["tilecnn"])
    else:
        print("[Warning] TileCNN weights not found in checkpoint, using random initialization")
    
    model.eval()
    tile_cnn.eval()
    
    return model, tile_cnn


def test_neural_solver(model: NeuralSolver, tile_cnn: TileCNN, tiles: List[np.ndarray], 
                      alphabet: int, device: torch.device, 
                      temperature: float = 0.7, greedy: bool = True,
                      max_steps: int = 2000, verbose: bool = False) -> Dict[str, Any]:
    """Test neural solver on a set of tiles."""
    env = TilePlacementEnv(tiles, alphabet=alphabet)
    
    # Precompute tile embeddings
    tile_embs = precompute_tile_embeddings(
        env.tiles, env.alphabet, tile_cnn, device, keep_on_device=(device.type == "cuda")
    )
    
    start_time = time.time()
    m, steps = evaluate_instance(model, env, tile_embs, device, temperature=temperature, greedy=greedy)
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"Neural Solver: m={m}, time={solve_time:.3f}s")
    
    return {
        "method": "Neural",
        "best_m": int(m),
        "time": solve_time,
        "status": "ok" if env.done else "incomplete",
        "placements": dict(env.placements)
    }


def test_greedy_solver(tiles: List[np.ndarray], seed: int = 42, K: int = 16, 
                      perimeter_limit: int = 64, verbose: bool = False) -> Dict[str, Any]:
    """Test greedy solver on a set of tiles."""
    params = GreedyParams(K=K, rng_seed=seed, perimeter_search_limit=perimeter_limit)
    
    start_time = time.time()
    result = greedy_place_once(tiles, params)
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"Greedy Solver: m={result['best_m']}, time={solve_time:.3f}s")
    
    return {
        "method": "Greedy",
        "best_m": result["best_m"],
        "time": solve_time,
        "status": result["status"],
        "placements": result["placements"]
    }


def test_aco_solver(tiles: List[np.ndarray], seed: int = 42, iterations: int = 50, 
                   n_ants: int = 8, n_workers: int = 4, verbose: bool = False,
                   enable_wandb: bool = False) -> Dict[str, Any]:
    """Test ACO solver on a set of tiles."""
    params = ACOParams(
        alpha=1.0, beta=3.0, gamma=1.0, lam=0.05, epsilon=1.0, rho=0.10,
        Q=float(tiles[0].shape[0] ** 2), n_ants=n_ants, iterations=iterations,
        random_seed=seed, perimeter_search_limit=32, enable_compaction=False,
        n_workers=n_workers, enable_wandb=enable_wandb
    )
    
    start_time = time.time()
    result = solve_with_aco(tiles, params)
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"ACO Solver: m={result['best_m']}, time={solve_time:.3f}s")
    
    return {
        "method": "ACO",
        "best_m": result["best_m"],
        "time": solve_time,
        "status": result["status"],
        "placements": result["placements"]
    }


def test_exact_solver(tiles: List[np.ndarray], time_limit: float = 30.0, 
                     verbose: bool = False) -> Dict[str, Any]:
    """Test exact solver on a set of tiles."""
    start_time = time.time()
    result = exact_bruteforce_min_canvas(tiles, time_limit=time_limit, verbose=verbose)
    solve_time = time.time() - start_time
    
    if verbose:
        print(f"Exact Solver: m={result['best_m']}, time={solve_time:.3f}s")
    
    return {
        "method": "Exact",
        "best_m": result["best_m"],
        "time": solve_time,
        "status": result["status"],
        "placements": result.get("best_placements", {})
    }


def compare_methods(tiles: List[np.ndarray], alphabet: int, 
                   neural_model=None, tile_cnn=None, device=None,
                   test_greedy: bool = True, test_aco: bool = True, 
                   test_exact: bool = False, seed: int = 42,
                   verbose: bool = False, enable_aco_wandb: bool = False) -> Dict[str, Any]:
    """Compare all available methods on a set of tiles."""
    results = {}
    
    if verbose:
        print(f"\n=== Testing on {len(tiles)} tiles of size {tiles[0].shape[0]}x{tiles[0].shape[0]} ===")
    
    # Test neural solver if model is provided
    if neural_model is not None and tile_cnn is not None and device is not None:
        try:
            result = test_neural_solver(neural_model, tile_cnn, tiles, alphabet, device, verbose=verbose)
            results["neural"] = result
        except Exception as e:
            print(f"Neural solver failed: {e}")
            results["neural"] = {"method": "Neural", "error": str(e)}
    
    # Test greedy solver
    if test_greedy:
        try:
            result = test_greedy_solver(tiles, seed=seed, verbose=verbose)
            results["greedy"] = result
        except Exception as e:
            print(f"Greedy solver failed: {e}")
            results["greedy"] = {"method": "Greedy", "error": str(e)}
    
    # Test ACO solver
    if test_aco:
        try:
            result = test_aco_solver(tiles, seed=seed, verbose=verbose)
            results["aco"] = result
        except Exception as e:
            print(f"ACO solver failed: {e}")
            results["aco"] = {"method": "ACO", "error": str(e)}
    
    # Test exact solver (only for small instances)
    if test_exact:
        try:
            result = test_exact_solver(tiles, verbose=verbose)
            results["exact"] = result
        except Exception as e:
            print(f"Exact solver failed: {e}")
            results["exact"] = {"method": "Exact", "error": str(e)}
    elif test_exact:
        if verbose:
            print("Skipping exact solver for large instance (>8 tiles)")
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a comparison table of results."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Method':<12} {'Best M':<8} {'Time (s)':<10} {'Status':<12}")
    print("-"*80)
    
    for method_name, result in results.items():
        if "error" in result:
            print(f"{result['method']:<12} {'ERROR':<8} {'-':<10} {result['error']:<12}")
        else:
            print(f"{result['method']:<12} {result['best_m']:<8} "
                  f"{result['time']:<10.3f} {result.get('status', 'N/A'):<12}")
    
    # Find best result
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_method = min(valid_results.keys(), key=lambda k: valid_results[k]["best_m"])
        best_m = valid_results[best_method]["best_m"]
        print("-"*80)
        print(f"Best solution: {valid_results[best_method]['method']} with M = {best_m}")


def save_results_to_csv(all_results: List[Dict[str, Dict[str, Any]]], 
                        args: argparse.Namespace, filename: str):
    """Save results to CSV file in results directory."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / filename
    
    # Prepare CSV data
    csv_data = []
    
    for instance_idx, instance_results in enumerate(all_results):
        for method_name, result in instance_results.items():
            row = {
                'timestamp': datetime.now().isoformat(),
                'instance_id': instance_idx,
                'method': result.get('method', method_name),
                'T': args.T,
                'n': args.n,
                'alphabet': args.alphabet,
                'seed': args.seed + instance_idx,
                'best_m': result.get('best_m', None),
                'time_seconds': result.get('time', None),
                'status': result.get('status', 'error' if 'error' in result else 'unknown'),
                'error_message': result.get('error', ''),
            }
            
            # Add neural model specific parameters
            if hasattr(args, 'checkpoint') and args.checkpoint:
                row['checkpoint'] = args.checkpoint
            if hasattr(args, 'temperature'):
                row['temperature'] = args.temperature
            if hasattr(args, 'greedy_inference'):
                row['greedy_inference'] = args.greedy_inference
                
            # Add ACO specific parameters
            if hasattr(args, 'aco_iterations'):
                row['aco_iterations'] = args.aco_iterations
            if hasattr(args, 'aco_ants'):
                row['aco_ants'] = args.aco_ants
            
            csv_data.append(row)
    
    # Write to CSV
    if csv_data:
        fieldnames = csv_data[0].keys()
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"Results saved to {csv_path}")
    else:
        print("No results to save")


def save_summary_to_csv(all_results: List[Dict[str, Dict[str, Any]]], 
                       args: argparse.Namespace, filename: str):
    """Save summary statistics to CSV file."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / filename
    
    # Collect method names
    method_names = set()
    for result in all_results:
        method_names.update(result.keys())
    
    # Prepare summary data
    summary_data = []
    
    for method in sorted(method_names):
        valid_results = []
        errors = 0
        
        for result in all_results:
            if method in result:
                if "error" not in result[method]:
                    valid_results.append(result[method])
                else:
                    errors += 1
        
        if valid_results:
            best_ms = [r["best_m"] for r in valid_results]
            times = [r["time"] for r in valid_results]
            
            summary_row = {
                'timestamp': datetime.now().isoformat(),
                'method': valid_results[0]['method'],
                'T': args.T,
                'n': args.n,
                'alphabet': args.alphabet,
                'test_instances': len(all_results),
                'successful_runs': len(valid_results),
                'errors': errors,
                'best_m_mean': np.mean(best_ms),
                'best_m_std': np.std(best_ms),
                'best_m_min': min(best_ms),
                'best_m_max': max(best_ms),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'time_min': min(times),
                'time_max': max(times),
            }
            
            # Add configuration parameters
            if hasattr(args, 'checkpoint') and args.checkpoint:
                summary_row['checkpoint'] = args.checkpoint
            if hasattr(args, 'temperature'):
                summary_row['temperature'] = args.temperature
            if hasattr(args, 'greedy_inference'):
                summary_row['greedy_inference'] = args.greedy_inference
            if hasattr(args, 'aco_iterations'):
                summary_row['aco_iterations'] = args.aco_iterations
            if hasattr(args, 'aco_ants'):
                summary_row['aco_ants'] = args.aco_ants
            
            summary_data.append(summary_row)
    
    # Write to CSV
    if summary_data:
        fieldnames = summary_data[0].keys()
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        
        print(f"Summary saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Test neural model with baseline comparisons")
    
    # Dataset parameters
    parser.add_argument("--T", type=int, default=30, help="Number of tiles")
    parser.add_argument("--n", type=int, default=4, help="Tile size (nxn)")
    parser.add_argument("--alphabet", type=int, default=2, help="Alphabet size")
    parser.add_argument("--test_size", type=int, default=10, help="Number of test instances")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Neural model parameters
    parser.add_argument("--checkpoint", type=str, help="Path to neural model checkpoint")
    parser.add_argument("--d_tile", type=int, default=64, help="Tile embedding dimension")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--cand_feat_dim", type=int, default=10, help="Candidate feature dimension")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--greedy_inference", action="store_true", help="Use greedy inference")
    
    # Baseline parameters
    parser.add_argument("--compare_baseline", action="store_true", help="Compare with baseline methods")
    parser.add_argument("--test_greedy", action="store_true", default=True, help="Test greedy solver")
    parser.add_argument("--test_aco", action="store_true", help="Test ACO solver")
    parser.add_argument("--test_exact", action="store_true", help="Test exact solver (small instances only)")
    parser.add_argument("--aco_iterations", type=int, default=50, help="ACO iterations")
    parser.add_argument("--aco_ants", type=int, default=8, help="Number of ACO ants")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--csv_out", action="store_true", help="Save results to CSV files")
    parser.add_argument("--csv_prefix", type=str, default="test_results", help="CSV file prefix")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load neural model if checkpoint is provided
    neural_model, tile_cnn = None, None
    if args.checkpoint:
        try:
            neural_model, tile_cnn = load_neural_model(
                args.checkpoint, args.alphabet, device,
                args.d_tile, args.d_model, args.cand_feat_dim
            )
            print(f"Successfully loaded neural model from {args.checkpoint}")
        except Exception as e:
            print(f"Failed to load neural model: {e}")
            return
    
    # Generate test dataset
    print(f"Generating {args.test_size} test instances with T={args.T}, n={args.n}, alphabet={args.alphabet}")
    
    all_results = []
    
    for i in range(args.test_size):
        # Generate tiles for this instance
        tiles = make_synthetic_tiles(T=args.T, n=args.n, alphabet=args.alphabet, seed=args.seed + i)
        
        if args.verbose:
            print(f"\n--- Test Instance {i+1}/{args.test_size} ---")
        
        # Compare methods on this instance
        instance_results = compare_methods(
            tiles, args.alphabet, 
            neural_model=neural_model, tile_cnn=tile_cnn, device=device,
            test_greedy=args.test_greedy or args.compare_baseline,
            test_aco=args.test_aco or args.compare_baseline, 
            test_exact=args.test_exact or (args.compare_baseline and args.T <= 8),
            seed=args.seed + i,
            verbose=args.verbose
        )
        
        all_results.append(instance_results)
        
        if not args.verbose and (i + 1) % max(1, args.test_size // 10) == 0:
            print(f"Completed {i+1}/{args.test_size} instances")
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    
    method_names = set()
    for result in all_results:
        method_names.update(result.keys())
    
    for method in sorted(method_names):
        valid_results = []
        errors = 0
        
        for result in all_results:
            if method in result:
                if "error" not in result[method]:
                    valid_results.append(result[method])
                else:
                    errors += 1
        
        if valid_results:
            best_ms = [r["best_m"] for r in valid_results]
            times = [r["time"] for r in valid_results]
            
            print(f"\n{valid_results[0]['method']} Solver:")
            print(f"  Successful runs: {len(valid_results)}/{args.test_size}")
            if errors > 0:
                print(f"  Errors: {errors}")
            print(f"  Best M - Mean: {np.mean(best_ms):.2f}, Std: {np.std(best_ms):.2f}, Min: {min(best_ms)}, Max: {max(best_ms)}")
            print(f"  Time - Mean: {np.mean(times):.3f}s, Std: {np.std(times):.3f}s")
    
    # Print individual instance results if verbose or small test size
    if args.verbose or args.test_size <= 5:
        for i, result in enumerate(all_results):
            print(f"\n--- Instance {i+1} ---")
            print_comparison_table(result)
    
    # Save results to CSV if requested
    if args.csv_out:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_csv = f"{args.csv_prefix}_detailed_{timestamp}.csv"
        save_results_to_csv(all_results, args, detailed_csv)
        
        # Save summary results
        summary_csv = f"{args.csv_prefix}_summary_{timestamp}.csv"
        save_summary_to_csv(all_results, args, summary_csv)


if __name__ == "__main__":
    main()