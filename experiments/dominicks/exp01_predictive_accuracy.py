"""
experiments/neural_demand/dominicks/exp01_predictive_accuracy.py
=================================================================
Section 5.1 — Predictive Accuracy on Dominick's Analgesics.

Trains all Neural Demand paper models on the Dominick's training split and
evaluates out-of-sample RMSE, MAE, and KL divergence on the held-out test
weeks.

Models evaluated
----------------
  LA-AIDS, Logit-IV, QUAIDS, Series Est.,
  Linear Demand (Shared), Linear Demand (GoodSpec), Linear Demand (Orth),
  Neural Demand (static), Neural Demand (habit),
  Neural Demand (FE), Neural Demand (habit, FE)

Produces
--------
results/neural_demand/dominicks/
  table_dom_predictive_accuracy.csv / .tex
  fig_dom_predictive_accuracy.{pdf,png}
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.dominicks import (
    LAAIDS, BLPLogitIV, QUAIDS, SeriesDemand,
    StaticND, StaticND_FE,
    HabitND,
        WindowND,
    _train,
    build_window_features,
    train_window_irl,
    cf_first_stage,
    compute_xbar_e2e,
    feat_good_specific, feat_orth, feat_shared,
    run_lirl,
)
from experiments.dominicks.data import G, GOODS
from experiments.dominicks.data import G as _G
from experiments.dominicks.utils import (
    predict,
    metrics,
    kl_divergence,
    fit_nd_delta_grid_dom,
    ALL_MODEL_NAMES,
    STYLE,
    bar_chart,
    make_performance_table,
    aggregate_runs,
    BAND,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int, splits: dict, cfg: dict) -> dict:
    """Delegate to unified main-runs pipeline and adapt output shape."""
    from experiments.dominicks.exp01_main_runs import run_once as _run_once_main

    r = _run_once_main(seed, splits, cfg)
    return dict(
        perf=r["perf"],
        delta_hat=r.get("delta_mdp", np.nan),
        delta_hat_fe=r.get("delta_mdp_fe", np.nan),
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(all_results: list) -> dict:
    return aggregate_runs(all_results, ALL_MODEL_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(perf_agg: dict, cfg: dict, n_runs: int = 1) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    se_note = f"  ({n_runs} runs, ±1 SE)" if n_runs > 1 else ""

    model_names = list(perf_agg.keys())
    rmse_means  = {nm: perf_agg[nm]["RMSE_mean"] for nm in model_names}
    rmse_ses    = {nm: perf_agg[nm]["RMSE_se"]   for nm in model_names}
    kl_means    = {nm: perf_agg[nm]["KL_mean"]   for nm in model_names}
    kl_ses      = {nm: perf_agg[nm]["KL_se"]     for nm in model_names}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bar_chart(rmse_means, rmse_ses, "Out-of-Sample RMSE",
              f"RMSE — Dominick's Analgesics{se_note}",
              ax=axes[0], n_runs=n_runs)
    bar_chart(kl_means, kl_ses, "KL Divergence KL(truth‖pred)",
              f"KL Divergence — Dominick's Analgesics{se_note}",
              ax=axes[1], n_runs=n_runs)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_dir}/fig_dom_predictive_accuracy.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig_dom_predictive_accuracy.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(splits: dict, cfg: dict) -> tuple:
    """Run predictive accuracy experiment over multiple seeds.

    Parameters
    ----------
    splits : from experiments.dominicks.data.load()
    cfg    : config dict (must include 'N_RUNS', 'out_dir', 'fig_dir')
    """
    N_RUNS = cfg.get("N_RUNS", 1)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Dominick's Exp 01: Predictive Accuracy")
    print("=" * 68)

    pre = cfg.get("_precomputed_dom_exp01")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed Exp01 runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = run_once(seed, splits, cfg)
            all_results.append(r)
            nd_rmse = r["perf"].get("Neural Demand (static)", {}).get("RMSE", np.nan)
            nh_rmse = r["perf"].get("Neural Demand (habit)",  {}).get("RMSE", np.nan)
            print(f"    Done in {time.time()-t0:.0f}s  "
                  f"NDS_RMSE={nd_rmse:.5f}  ND+Habit_RMSE={nh_rmse:.5f}")

    perf_agg = aggregate(all_results)
    make_figures(perf_agg, cfg, n_runs=N_RUNS)
    make_performance_table(
        perf_agg,
        out_dir=cfg["out_dir"],
        label="table_dom_predictive_accuracy",
        caption=(r"Predictive Accuracy --- Dominick's Analgesics. "
                 r"Out-of-sample RMSE, MAE, and KL divergence on held-out test weeks. "
                 r"Best result per column in \textbf{bold}."),
        n_runs=N_RUNS,
    )

    print("\n── Predictive Accuracy Summary ─────────────────────────────────────")
    for nm, d in perf_agg.items():
        print(f"  {nm:40s}  RMSE={d['RMSE_mean']:.5f}±{d['RMSE_se']:.5f}"
              f"  KL={d['KL_mean']:.5f}±{d['KL_se']:.5f}")

    return all_results, perf_agg
