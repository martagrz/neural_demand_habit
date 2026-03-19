#!/usr/bin/env python
"""
run_neural_demand_dominicks.py
================================
Top-level runner for all Neural Demand paper **Dominick's** experiments.

Usage
-----
  python run_neural_demand_dominicks.py \\
      --weekly PATH_TO_WBER.csv \\
      --upc    PATH_TO_UPC.csv  \\
      [--fast] [--exp EXP [EXP ...]]

  --weekly     Path to the Dominick's weekly movement file (WBER).
  --upc        Path to the UPC catalogue file.
  --fast       Reduced N_RUNS and training epochs for quick testing.
  --exp        Subset of experiments to run: 01 02 03 04 05 06 07 08 09 12 (default: all).

Outputs are written to
  results/neural_demand/dominicks/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import re

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10000 

np.random.seed(42)
torch.manual_seed(42)

BASE_CFG = dict(
    # Data paths (overridden by CLI args)
    weekly_path = "./data/wana.csv",
    upc_path    = "./data/upcana.csv",

    # Data prep
    std_tablets  = 100,
    min_store_wks= 20,      # match dominicks_multiple_runs.py (was 52)
    habit_decay  = 0.7,
    test_cutoff  = 351,     # match dominicks_multiple_runs.py (was 375)
    shock_good   = 0,       # good index for welfare shocks (0 = Aspirin)
    shock_pct    = 0.10,    # 10% price shock for welfare
    cv_steps     = 100,

    # Experiment-level
    N_RUNS       = 5,
    device       = DEVICE,

    # LA-AIDS / QUAIDS / Series
    aids_epochs  = None,    # fit is OLS — no epochs needed

    # Neural IRL (static)
    nirl_hidden      = 128,
    nirl_epochs      = EPOCHS,
    nirl_lr          = 1e-4,
    nirl_batch       = 256,
    nirl_lam_mono    = 0.2,
    nirl_lam_slut    = 0.05,
    nirl_slut_start  = 0.3,

    # MDP / Habit neural IRL
    mdp_hidden       = 128,
    mdp_epochs       = EPOCHS,
    mdp_lr           = 1e-4,
    mdp_batch        = 256,
    mdp_lam_mono     = 0.3,
    mdp_lam_slut     = 0.1,
    mdp_slut_start   = 0.25,

    # MDP E2E (habit-augmented Neural Demand)
    mdp_e2e_hidden       = 128,
    mdp_e2e_epochs       = EPOCHS,
    mdp_e2e_lr           = 1e-4,
    mdp_e2e_batch        = 256,
    mdp_e2e_lam_mono     = 0.3,
    mdp_e2e_lam_slut     = 0.1,
    mdp_e2e_slut_start   = 0.25,

    # Linear IRL
    lirl_epochs  = EPOCHS,
    lirl_lr      = 0.05,
    lirl_l2      = 1e-4,
    lirl_alpha   = 1e-3,

    # δ sweep for identification experiment
    delta_grid_identification = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),

    # Outputs
    out_dir = "results/neural_demand/dominicks",
    fig_dir = "results/neural_demand/dominicks/figures",
    # model_cache_dir set dynamically in main()
)

FAST_CFG = dict(
    N_RUNS           = 2,
    nirl_epochs      = 100,
    mdp_epochs       = 100,
    mdp_e2e_epochs   = 100,
    lirl_epochs      = 100,
    delta_grid_identification = np.array([0.2, 0.4, 0.6, 0.8]),
)


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Neural Demand Dominick's experiments")
    p.add_argument("--weekly", type=str, default=None,
                   help="Path to Dominick's weekly movement CSV")
    p.add_argument("--upc",    type=str, default=None,
                   help="Path to Dominick's UPC catalogue CSV")
    p.add_argument("--fast",   action="store_true",
                   help="Use reduced settings for quick testing")
    p.add_argument("--load",   action="store_true",
                   help="Load pre-trained models from cache (default: train from scratch)")
    p.add_argument("--verbose", action="store_true",
                   help="Verbose training logs (default: quiet; prints only KL_min summaries)")
    p.add_argument("--exp", nargs="+", type=str, default=None,
                   help="Experiments to run: 01 02 03 04 05 06 07 08 09 12 (default: all)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

def _run_exp01(splits, cfg):
    from experiments.dominicks.exp01_predictive_accuracy import run
    return run(splits, cfg)


def _run_exp02(splits, cfg):
    from experiments.dominicks.exp02_elasticities import run
    return run(splits, cfg)


def _run_exp03(splits, cfg):
    from experiments.dominicks.exp03_welfare import run
    return run(splits, cfg)


def _run_exp04(splits, cfg):
    from experiments.dominicks.exp04_demand_curves import run
    return run(splits, cfg)


def _run_exp05(splits, cfg):
    from experiments.dominicks.exp05_delta_identification import run
    return run(splits, cfg)


def _run_exp06(splits, cfg):
    from experiments.dominicks.exp06_cf_decomposition import run
    return run(splits, cfg)


def _run_exp07(splits, cfg):
    """Full model sweep — generates all main paper figures.

    Trains every model (Neural Demand static/habit/joint/window, LDS variants,
    store-FE variants, CF variants) for N_RUNS seeds using
    the same ``run_once`` / ``aggregate`` pipeline as the original
    Dominick's multiple-runs script, then writes:

    * fig_convergence.{pdf,png}         — Training convergence (KL + δ)
    * fig_mdp_advantage.{pdf,png}       — Cross-price demand matrix (3×3)
    * fig_scatter.{pdf,png}             — Observed vs predicted scatter
    * fig_demand_curves.{pdf,png}       — Demand curves (shock good)
    * fig_cross_elast_heatmap.{pdf,png} — Cross-price elasticity heatmaps
    * fig_segmentation_sorting.{pdf,png}— Market segmentation diagnostics
    * fig_mdp_decomposition.{pdf,png}   — MDP structural decomposition
    * fig_rmse_bars.{pdf,png}           — RMSE bar chart (N_RUNS > 1)
    * dominicks_latex.tex + table*.csv  — LaTeX tables and CSVs
    """
    import time as _time
    from experiments.dominicks.exp01_main_runs import (
        run_once as _run_once,
        aggregate as _aggregate,
        _make_figures,
        _make_tables,
    )

    N_RUNS = cfg.get("N_RUNS", 1)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Exp 07 — Full Model Figures (Training Convergence, Demand, etc.)")
    print("=" * 68)

    pre = cfg.get("_dom_main_runs")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed full-model runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15          # mirrors dominicks_multiple_runs.py seeds
            t0   = _time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = _run_once(seed, splits, cfg)
            all_results.append(r)
            r_n = r["perf"].get("Neural Demand (static)", {}).get("RMSE", float("nan"))
            r_m = r["perf"].get("Neural Demand (habit)",  {}).get("RMSE", float("nan"))
            print(f"    Done in {_time.time()-t0:.0f}s  "
                  f"ND_static_RMSE={r_n:.5f}  ND_habit_RMSE={r_m:.5f}")

    agg = _aggregate(all_results)
    _make_figures(agg, splits, cfg)
    _make_tables(agg, splits, cfg)
    return all_results, agg


def _run_exp08(splits, cfg):
    from experiments.dominicks.exp07_first_stage import run
    return run(splits, cfg)


def _run_exp09(splits, cfg):
    from experiments.dominicks.exp09_regularity_dashboard import run
    return run(splits, cfg)


def _run_exp12(splits, cfg):
    from argparse import Namespace
    from experiments.dominicks.exp12_per_item_blp_scaling import run

    data_dir = os.path.dirname(cfg.get("weekly_path", "data/wana.csv")) or "data"
    args = Namespace(
        weekly=cfg.get("weekly_path", "data/wana.csv"),
        upc=cfg.get("upc_path", "data/upcana.csv"),
        data_dir=data_dir,
        fast=bool(cfg.get("N_RUNS", 5) <= 2),
        force_retrain=bool(cfg.get("force_retrain", False)),
        verbose=bool(cfg.get("verbose", False)),
        test_cutoff=int(cfg.get("test_cutoff", 351)),
        min_store_weeks=int(cfg.get("min_store_wks", 20)),
        seed=42,
        goods_grid="8,12,16,24,32",
        n_upc_cap=0,
    )
    return run(args)


EXPERIMENTS = {
    "01": ("Predictive Accuracy",        _run_exp01),
    "02": ("Elasticities",               _run_exp02),
    "03": ("Welfare (CV)",               _run_exp03),
    "04": ("Demand Curves",              _run_exp04),
    "05": ("δ Identification",           _run_exp05),
    "06": ("CF Decomposition (Sec 2.4)", _run_exp06),
    "07": ("Full Model Figures",         _run_exp07),
    "08": ("First-Stage Diagnostics",    _run_exp08),
    "09": ("Regularity Dashboard",       _run_exp09),
    "12": ("UPC-level BLP vs NDS",       _run_exp12),
}


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # Build config
    cfg = dict(BASE_CFG)
    if args.fast:
        cfg.update(FAST_CFG)
        cfg["model_cache_dir"] = "results/neural_demand/dominicks/models/fast"
        print("[fast mode] Using reduced N_RUNS / training epochs")
    else:
        cfg["model_cache_dir"] = "results/neural_demand/dominicks/models/full"

    # If --load is NOT set, force retraining (ignore existing cache but save new models)
    cfg["force_retrain"] = not args.load
    cfg["verbose"] = bool(args.verbose)

    if args.weekly:
        cfg["weekly_path"] = args.weekly
    if args.upc:
        cfg["upc_path"] = args.upc

    # Check data paths
    for key in ("weekly_path", "upc_path"):
        if not os.path.isfile(cfg[key]):
            print(f"[warn] {key!r} not found at {cfg[key]!r}.  "
                  "Pass --weekly / --upc or edit BASE_CFG.")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    # Which experiments to run
    if args.exp is None:
        exps_to_run = sorted(EXPERIMENTS.keys())
    else:
        exps_to_run = []
        for e in args.exp:
            key = e.zfill(2)
            if key not in EXPERIMENTS:
                print(f"[warn] Unknown experiment key {e!r}, skipping.")
            else:
                exps_to_run.append(key)

    print("\n" + "#" * 68)
    print("  Neural Demand — Dominick's Experiments")
    print(f"  Device : {DEVICE}")
    print(f"  N_RUNS : {cfg['N_RUNS']}")
    print(f"  Running: {', '.join(exps_to_run)}")
    print("#" * 68)

    # ── Load data once ─────────────────────────────────────────────────────────
    print("\n[Data] Loading Dominick's data...")
    t_load = time.time()
    try:
        from experiments.dominicks.data import load
        _, splits = load(cfg)
    except Exception as exc:
        import traceback
        print("[ERROR] Data loading failed:")
        traceback.print_exc()
        sys.exit(1)
    print(f"[Data] Loaded in {time.time()-t_load:.0f}s")

    # ── Precompute seed-level runs once, then reuse in experiment run() calls ──
    def _cached_seeds_for_habit_grid(cache_dir: str):
        if not os.path.isdir(cache_dir):
            return []
        delta_grid = cfg.get("delta_grid_identification", np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        dset = {f"{float(d):.2f}" for d in np.asarray(delta_grid, dtype=float)}
        by_seed = {}
        pat = re.compile(r"^ND_Habit_s(\d+)_d(\d\.\d{2})\.pt$")
        for fn in os.listdir(cache_dir):
            m = pat.match(fn)
            if not m:
                continue
            sd = int(m.group(1))
            dd = m.group(2)
            by_seed.setdefault(sd, set()).add(dd)
        complete = [s for s, ds in by_seed.items() if dset.issubset(ds)]
        return sorted(complete)

    seeds = [42 + ri * 15 for ri in range(cfg["N_RUNS"])]
    if not cfg.get("force_retrain", False):
        cached = _cached_seeds_for_habit_grid(cfg.get("model_cache_dir", ""))
        planned = set(seeds)
        if planned.issubset(set(cached)):
            print(f"[Precompute] Using planned fixed seed set (fully cached): {seeds}")
        else:
            missing = [s for s in seeds if s not in set(cached)]
            print(f"[Precompute] Keeping planned seed set {seeds}; missing cache for {missing}")
    if exps_to_run:
        print("\n[Precompute] Building shared seed cache...")

    if any(k in exps_to_run for k in ("01", "02", "03", "07")):
        from experiments.dominicks.exp01_main_runs import run_once as _main_run_once
        dom_main_runs = []
        for ri, seed in enumerate(seeds, 1):
            print(f"  [main] seed={seed} ({ri}/{len(seeds)})")
            dom_main_runs.append(_main_run_once(seed, splits, cfg))
        cfg["_dom_main_runs"] = dom_main_runs
        if "01" in exps_to_run:
            cfg["_precomputed_dom_exp01"] = [
                {
                    "perf": r["perf"],
                    "delta_hat": r.get("delta_mdp", np.nan),
                    "delta_hat_fe": r.get("delta_mdp_fe", np.nan),
                    "seed": seeds[i],
                }
                for i, r in enumerate(dom_main_runs)
            ]
        if "02" in exps_to_run:
            cfg["_precomputed_dom_exp02"] = [
                {
                    "own_eps": r["elast"],
                    "cross_eps": r["cross_elast"],
                    "p_mn": splits["p_mn"],
                    "delta_hat": r.get("delta_mdp", np.nan),
                    "seed": seeds[i],
                }
                for i, r in enumerate(dom_main_runs)
            ]
        if "03" in exps_to_run:
            # Reuse full-model runs from Exp 01 pipeline so Exp 03 does not retrain.
            cfg["_precomputed_dom_exp03"] = [
                {
                    "cv": r.get("welf", {}),
                    "cv_by_pct": r.get("welf_by_pct", {}),
                    "p0w": splits.get("p0w"),
                    "p1w": splits.get("p1w"),
                    "shock_good": cfg.get("shock_good", 0),
                    "shock_pct": cfg.get("shock_pct", 0.10),
                    "delta_hat": r.get("delta_mdp", np.nan),
                    "seed": seeds[i],
                }
                for i, r in enumerate(dom_main_runs)
            ]

    # Keep the rest of experiments from retraining inside run() by precomputing
    # their seed-level outputs now and passing them through cfg.
    _dom_pre_specs = [
        ("03", "experiments.dominicks.exp03_welfare", "_precomputed_dom_exp03"),
        ("04", "experiments.dominicks.exp04_demand_curves", "_precomputed_dom_exp04"),
        ("05", "experiments.dominicks.exp05_delta_identification", "_precomputed_dom_exp05"),
        ("06", "experiments.dominicks.exp06_cf_decomposition", "_precomputed_dom_exp06"),
    ]
    for exp_key, mod_name, cfg_key in _dom_pre_specs:
        if exp_key not in exps_to_run:
            continue
        if cfg_key in cfg:
            continue
        mod = __import__(mod_name, fromlist=["run_once"])
        run_once = getattr(mod, "run_once")
        pre = []
        for ri, seed in enumerate(seeds, 1):
            print(f"  [exp {exp_key}] seed={seed} ({ri}/{len(seeds)})")
            pre.append(run_once(seed, splits, cfg))
        cfg[cfg_key] = pre

    # ── Run experiments ────────────────────────────────────────────────────────
    t_total = time.time()
    results = {}
    for key in exps_to_run:
        name, fn = EXPERIMENTS[key]
        t0 = time.time()
        print(f"\n{'='*68}")
        print(f"  Exp {key} — {name}")
        print(f"{'='*68}")
        try:
            results[key] = fn(splits, cfg)
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] Experiment {key} ({name}) failed:")
            traceback.print_exc()
            results[key] = None
        elapsed = time.time() - t0
        status  = "OK" if results[key] is not None else "FAILED"
        print(f"\n[{key}] {name}: {status} in {elapsed:.0f}s")

    print("\n" + "#" * 68)
    print(f"  All done in {time.time()-t_total:.0f}s")
    print(f"  Outputs → {cfg['out_dir']}")
    print("#" * 68)
    return results


if __name__ == "__main__":
    main()
