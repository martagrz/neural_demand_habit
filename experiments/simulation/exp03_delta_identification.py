"""
experiments/neural_demand/simulation/exp03_delta_identification.py
==================================================================
Section 4 — Non-Identification of the Habit-Decay Parameter δ.

This experiment documents the *within-sample* non-identification of δ by
profiling the validation KL divergence over a fine grid of frozen-δ values.
The key result in the paper is that many δ values achieve near-identical KL,
forming a flat plateau that defines the identified set (IS).

Outputs
-------
results/neural_demand/simulations/
  table_delta_identification.csv / .tex
  fig_delta_profile_kl.{pdf,png}   — profile KL ± 1 SE (aggregated)
  fig_delta_id_set.{pdf,png}        — IS width / coverage across seeds
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.simulation import (
    HabitFormationConsumer,
    compute_xbar_e2e,
)
from experiments.simulation.utils import (
    P_GRID,
    AVG_Y,
    BAND,
    predict_shares,
    kl_div,
    fit_neural_demand_delta_grid,
)

warnings.filterwarnings("ignore")

TEAL   = "#00897B"
ORANGE = "#FB8C00"
RED    = "#E53935"


# ─────────────────────────────────────────────────────────────────────────────
#  Single-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(seed: int, cfg: dict, verbose: bool = False) -> dict:
    """Profile KL divergence over the δ grid for one seed.

    Returns
    -------
    dict with:
        delta_grid  (K,)  swept δ values
        kl_grid     (K,)  mean val KL at each δ
        se_grid     (K,)  SE of per-obs val KL
        delta_hat   float  argmin KL
        id_set      (lo, hi)
        id_mask     (K,) bool
        delta_true  float
        delta_in_id_set  bool
    """
    N         = cfg["N_OBS"]
    DEVICE    = cfg["DEVICE"]
    DELTA_GRID = np.asarray(cfg["DELTA_GRID"], dtype=float)
    EPOCHS    = cfg["EPOCHS"]
    TRUE_DELTA = float(cfg.get("TRUE_DELTA", 0.7))
    HIDDEN    = int(cfg.get("hidden_dim", 128))
    CACHE_DIR = cfg.get("model_cache_dir")
    FORCE_RETRAIN = cfg.get("force_retrain", False)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Simulate data from a habit DGP ────────────────────────────────────────
    p_tr   = np.clip(np.random.uniform(1, 5, (N, 3))
                     + np.random.normal(0, 0.1, (N, 3)), 1e-3, None)
    income = np.random.uniform(1200, 2000, N)

    hc     = HabitFormationConsumer()
    w_tr, xbar_tr = hc.solve_demand(p_tr, income, return_xbar=True)

    q_tr   = w_tr * income[:, None] / np.maximum(p_tr, 1e-8)
    lq_tr  = np.log(np.maximum(q_tr, 1e-6))

    # ── Validation split ───────────────────────────────────────────────────────
    rng    = np.random.default_rng(seed + 9999)
    N_val  = max(N // 5, 100)
    p_val  = np.clip(rng.uniform(1, 5, (N_val, 3))
                     + rng.normal(0, 0.1, (N_val, 3)), 1e-3, None)
    y_val  = rng.uniform(1200, 2000, N_val)
    hcv    = HabitFormationConsumer()
    w_val, _ = hcv.solve_demand(p_val, y_val, return_xbar=True)
    q_val  = w_val * y_val[:, None] / np.maximum(p_val, 1e-8)
    lq_val = np.log(np.maximum(q_val, 1e-6))

    # ── δ sweep (frozen, dense grid) ─────────────────────────────────────────
    sweep = fit_neural_demand_delta_grid(
        p_tr, income, w_tr, lq_tr,
        p_val, y_val, w_val, lq_val,
        delta_grid=DELTA_GRID,
        epochs=EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.3, lam_slut=0.1,
        hidden_dim=HIDDEN,
        device=DEVICE,
        tag=f"nd-delta-id-s{seed}",
        cache_dir=CACHE_DIR,
        force_retrain=FORCE_RETRAIN,
    )

    return dict(
        delta_grid=DELTA_GRID.copy(),
        kl_grid=sweep["kl_grid"],
        se_grid=sweep["se_grid"],
        delta_hat=sweep["delta_hat"],
        id_set=sweep["id_set"],
        id_mask=sweep["id_mask"],
        delta_true=TRUE_DELTA,
        delta_in_id_set=bool(sweep["id_set"][0] <= TRUE_DELTA <= sweep["id_set"][1]),
        all_models=sweep["all_models"],
        all_hists=sweep["all_hists"],
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray([x for x in arr if not np.isnan(float(x))], float)
    return float(np.std(a, ddof=1) / np.sqrt(max(len(a), 1)))


def aggregate(all_results: list) -> dict:
    n    = len(all_results)
    dg   = all_results[0]["delta_grid"]   # (K,) same across all seeds

    kl_stack  = np.stack([r["kl_grid"] for r in all_results], 0)   # (n, K)
    se_stack  = np.stack([r["se_grid"] for r in all_results], 0)   # (n, K)

    kl_mean   = kl_stack.mean(0)
    kl_se     = kl_stack.std(0, ddof=max(1, n - 1)) / np.sqrt(n)

    delta_hats   = [r["delta_hat"] for r in all_results]
    id_set_lo    = [r["id_set"][0] for r in all_results]
    id_set_hi    = [r["id_set"][1] for r in all_results]
    id_widths    = [hi - lo for lo, hi in zip(id_set_lo, id_set_hi)]
    in_id_frac   = float(np.mean([r["delta_in_id_set"] for r in all_results]))

    return dict(
        delta_grid=dg,
        kl_mean=kl_mean, kl_se=kl_se,
        delta_hat_mean=float(np.nanmean(delta_hats)),
        delta_hat_se=_se(delta_hats),
        delta_hat_all=delta_hats,
        id_set_lo_mean=float(np.nanmean(id_set_lo)),
        id_set_hi_mean=float(np.nanmean(id_set_hi)),
        id_set_lo_all=id_set_lo,
        id_set_hi_all=id_set_hi,
        id_width_mean=float(np.nanmean(id_widths)),
        id_width_se=_se(id_widths),
        in_id_frac=in_id_frac,
        true_delta=all_results[0]["delta_true"],
        n_runs=n,
        last=all_results[-1],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(agg: dict, cfg: dict) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    N_RUNS  = agg["n_runs"]
    se_note = f"  ({N_RUNS} seeds, ±1 SE)" if N_RUNS > 1 else ""
    dg      = agg["delta_grid"]

    # ── Profile KL ─────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    mu = agg["kl_mean"]
    se = agg["kl_se"]
    ax1.plot(dg, mu, color=TEAL, lw=2.5, label="Val. KL (mean)")
    if N_RUNS > 1:
        ax1.fill_between(dg, mu - se, mu + se, color=TEAL, alpha=BAND)
    ax1.axvline(agg["true_delta"], color=RED, ls="--", lw=2.0,
                label=f"True δ = {agg['true_delta']:.1f}")
    ax1.axvline(agg["delta_hat_mean"], color=ORANGE, ls=":", lw=2.0,
                label=f"δ̂ = {agg['delta_hat_mean']:.2f} (mean)")

    # shade identified set
    lo, hi = agg["id_set_lo_mean"], agg["id_set_hi_mean"]
    ax1.axvspan(lo, hi, color=TEAL, alpha=0.08,
                label=f"Identified set [{lo:.2f}, {hi:.2f}]")
    ax1.set_xlabel("Habit-decay parameter δ", fontsize=13)
    ax1.set_ylabel("Validation KL divergence", fontsize=13)
    # ax1.set_title(f"Profile KL — δ Non-Identification{se_note}",
    #               fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    for ext in ("pdf", "png"):
        fig1.savefig(f"{fig_dir}/fig_delta_profile_kl.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {fig_dir}/fig_delta_profile_kl.pdf/png")

    # ── Identified-set width / coverage across seeds ─────────────────────────
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))

    seed_idx = np.arange(1, N_RUNS + 1)
    id_lo    = np.asarray(agg["id_set_lo_all"])
    id_hi    = np.asarray(agg["id_set_hi_all"])
    id_mid   = (id_lo + id_hi) / 2

    # vertical interval plot
    for i in range(N_RUNS):
        col = TEAL if agg["last"]["delta_true"] >= id_lo[i] and \
                      agg["last"]["delta_true"] <= id_hi[i] else RED
        axA.plot([seed_idx[i], seed_idx[i]], [id_lo[i], id_hi[i]],
                 color=col, lw=1.5, alpha=0.7)
        axA.scatter(seed_idx[i], agg["delta_hat_all"][i], color=ORANGE, s=18, zorder=3)
    axA.axhline(agg["true_delta"], color=RED, ls="--", lw=2.0,
                label=f"True δ={agg['true_delta']:.1f}")
    axA.set_xlabel("Seed index", fontsize=12)
    axA.set_ylabel("Identified set / δ̂", fontsize=12)
    # axA.set_title("IS per seed  (teal = covers truth)", fontsize=11, fontweight="bold")
    axA.legend(fontsize=10)
    axA.grid(True, alpha=0.3)

    id_widths_arr = id_hi - id_lo
    axB.hist(id_widths_arr, bins=min(15, N_RUNS), color=TEAL, edgecolor="k", alpha=0.8)
    axB.set_xlabel("IS width (hi − lo)", fontsize=12)
    axB.set_ylabel("Count", fontsize=12)
    # axB.set_title(f"Distribution of IS Width  "
    #               f"(mean={agg['id_width_mean']:.2f}±{agg['id_width_se']:.2f})",
    #               fontsize=11, fontweight="bold")
    axB.grid(True, alpha=0.3)

    # fig2.suptitle(f"Identified-Set Coverage{se_note}  "
    #               f"[coverage={100*agg['in_id_frac']:.0f}%]",
    #               fontsize=12, fontweight="bold")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(f"{fig_dir}/fig_delta_id_set.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {fig_dir}/fig_delta_id_set.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def make_tables(agg: dict, cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    N_RUNS  = agg["n_runs"]
    os.makedirs(out_dir, exist_ok=True)

    dg = agg["delta_grid"]
    df = pd.DataFrame({
        "delta":  dg,
        "kl_mean": agg["kl_mean"],
        "kl_se":   agg["kl_se"],
    })
    df.round(8).to_csv(f"{out_dir}/table_delta_identification.csv", index=False)
    print(f"  Saved: {out_dir}/table_delta_identification.csv")

    # Summary scalar table
    summary = {
        "True δ":             agg["true_delta"],
        "δ̂ (mean ± SE)":     f"{agg['delta_hat_mean']:.3f} ± {agg['delta_hat_se']:.3f}",
        "IS lo (mean)":       f"{agg['id_set_lo_mean']:.3f}",
        "IS hi (mean)":       f"{agg['id_set_hi_mean']:.3f}",
        "IS width (mean±SE)": f"{agg['id_width_mean']:.3f} ± {agg['id_width_se']:.3f}",
        "Coverage (%)":       f"{100*agg['in_id_frac']:.0f}",
    }

    lines = [
        r"% ============================================================",
        r"% Neural Demand — δ Identification Table (auto-generated)",
        f"% N_RUNS = {N_RUNS}",
        r"% ============================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Non-Identification of $\delta$ (Habit-Decay Parameter) "
        rf"--- {N_RUNS} seeds}}",
        r"  \label{tab:sim_delta_identification}",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    \textbf{Quantity} & \textbf{Value} \\",
        r"    \midrule",
    ]
    for k, v in summary.items():
        lines.append(f"    {k} & {v} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption*{{Profile KL threshold = $\hat{{KL}}_{{min}} + 2\,\widehat{{SE}}$."
        rf"  True $\delta={agg['true_delta']:.1f}$.  Identified set covers the truth in"
        rf"  {100*agg['in_id_frac']:.0f}\% of simulation seeds.}}",
        r"\end{table}", "",
    ]

    tex_path = f"{out_dir}/table_delta_identification.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> tuple:
    """Run all seeds, aggregate, produce outputs.

    Parameters
    ----------
    cfg : dict with keys:
        N_RUNS      int
        N_OBS       int
        EPOCHS      int
        DELTA_GRID  array — δ values to profile (should be fine, e.g. 0.1..0.99, 30 pts)
        DEVICE      str
        TRUE_DELTA  float  (default 0.7)
        hidden_dim  int    (default 128)
        out_dir     str
        fig_dir     str
    """
    N_RUNS = cfg["N_RUNS"]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand  —  Simulation Exp 03: δ Identification")
    print("=" * 68)

    pre = cfg.get("_precomputed_sim_exp03")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed Exp03 runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = run_one_seed(seed, cfg, verbose=(ri == N_RUNS - 1))
            all_results.append(r)
            print(f"    Done in {time.time()-t0:.0f}s  "
                  f"δ̂={r['delta_hat']:.2f}  "
                  f"IS=[{r['id_set'][0]:.2f}, {r['id_set'][1]:.2f}]  "
                  f"in_IS={r['delta_in_id_set']}")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    make_tables(agg, cfg)

    print(f"\n── δ Identification Summary ──────────────────────────────────────")
    print(f"  True δ          = {agg['true_delta']:.2f}")
    print(f"  δ̂ (mean ± SE)   = {agg['delta_hat_mean']:.3f} ± {agg['delta_hat_se']:.3f}")
    print(f"  IS width        = {agg['id_width_mean']:.3f} ± {agg['id_width_se']:.3f}")
    print(f"  Coverage        = {100*agg['in_id_frac']:.0f}%\n")

    return all_results, agg
