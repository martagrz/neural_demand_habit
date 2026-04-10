#!/usr/bin/env python
"""
run_neural_demand_simulation.py
================================
Top-level runner for all Neural Demand paper **simulation** experiments.

Usage
-----
  python run_neural_demand_simulation.py [--fast] [--exp EXP [EXP ...]]

  --fast        Use reduced N_OBS / EPOCHS / N_RUNS for quick sanity checks.
  --exp         Comma- or space-separated subset of experiments to run:
                  01  DGP recovery  (Section 2)
                  02  Habit advantage  (Section 3)
                  03  δ identification  (Section 4)
                  04  CF endogeneity correction  (Section 2.4)
                If omitted, all experiments are run.

Outputs are written to
  results/neural_demand/simulations/

Paper figures are written to
  results/neural_demand/simulations/figures/paper_*
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CFG = dict(
    # Experiment-level
    N_RUNS   = 5,
    N_OBS    = 800,
    EPOCHS   = 10000,
    DEVICE   = DEVICE,

    # Neural Demand (static) hyper-params
    hidden_dim      = 128,

    # δ sweep
    DELTA_GRID  = np.linspace(0.4,0.9,16),
    TRUE_DELTA  = 0.7,

    # Exp 04 — CF endogeneity correction
    RHO_GRID   = [0.0, 0.3, 0.6, 0.9],   # endogeneity strength grid
    DELTA_HAB  = 0.7,                     # fixed δ for habit-CF model

    # Output paths
    out_dir = "results/neural_demand/simulations",
    fig_dir = "results/neural_demand/simulations/figures",
    # model_cache_dir set dynamically in main() based on fast/full mode
)

# "Fast" override for rapid development / CI
FAST_CFG = dict(
    N_RUNS      = 2,
    N_OBS       = 100,
    EPOCHS      = 100,
    DELTA_GRID  = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    RHO_GRID    = [0.0, 0.6],
)


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Neural Demand simulation experiments")
    p.add_argument("--fast", action="store_true",
                   help="Use reduced settings for quick testing")
    p.add_argument("--load", action="store_true",
                   help="Load pre-trained models from cache (default: train from scratch)")
    p.add_argument("--exp", nargs="+", type=str, default=None,
                   help="Experiments to run: 01 02 03 04 (default: all)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training epochs (default: 10000)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

def _run_exp01(cfg):
    from experiments.simulation.exp01_dgp_recovery import run
    print("\n" + "=" * 68)
    print("  Exp 01 — DGP Recovery")
    print("=" * 68)
    return run(cfg)


def _run_exp02(cfg):
    from experiments.simulation.exp02_habit_advantage import run
    print("\n" + "=" * 68)
    print("  Exp 02 — Habit Advantage")
    print("=" * 68)
    return run(cfg)


def _run_exp03(cfg):
    from experiments.simulation.exp03_delta_identification import run
    # Use a finer delta grid for the identification experiment
    id_cfg = dict(cfg)
    id_cfg["DELTA_GRID"] = np.linspace(0.1, 0.95, 30)
    if "FAST_DELTA_GRID" in cfg:
        id_cfg["DELTA_GRID"] = cfg["FAST_DELTA_GRID"]
    print("\n" + "=" * 68)
    print("  Exp 03 — δ Identification")
    print("=" * 68)
    return run(id_cfg)


def _run_exp04(cfg):
    from experiments.simulation.exp04_cf_endogeneity import run
    print("\n" + "=" * 68)
    print("  Exp 04 — CF Endogeneity Correction  (Section 2.4)")
    print("=" * 68)
    return run(cfg)


EXPERIMENTS = {
    "01": ("DGP Recovery",             _run_exp01),
    "02": ("Habit Advantage",          _run_exp02),
    "03": ("δ Identification",         _run_exp03),
    "04": ("CF Endogeneity Correction", _run_exp04),
}


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Display names for the models used in the neural demand simulation experiments
_MODEL_DISPLAY = {
    "Truth":                        "Truth (Habit)",
    "LA-AIDS":                      "LA-AIDS",
    "QUAIDS":                       "QUAIDS",
    "Series Estm.":                 "Series Estimator",
    "Linear Demand (Shared)":                 "Linear Demand (Shared)",
    "Linear Demand (GoodSpec)":               "Linear Demand (GoodSpec)",
    "Linear Demand (Orth)":                   "Linear Demand (Orth)",
    "Neural Demand (static)":       "Neural Demand (static)",
    "Neural Demand (habit)":        "Neural Demand (habit)",
    "Neural Demand (CF)":           "Neural Demand (CF)",
    "Neural Demand (habit, CF)":    "Neural Demand (habit, CF)",
    "Ground Truth":                 "Ground Truth",
}

# Plotting styles aligned with paper figures
_PAPER_STYLE = {
    "Truth (Habit)":                dict(color="k",       ls="-",  lw=2.5),
    "LA-AIDS":                      dict(color="#E53935", ls="--", lw=2.0),
    "QUAIDS":                       dict(color="#43A047", ls="-.", lw=2.0),
    "Series Estimator":             dict(color="#FB8C00", ls=":",  lw=2.0),
    "Linear Demand (Shared)":                 dict(color="#039BE5", ls=":",  lw=1.5),
    "Linear Demand (GoodSpec)":               dict(color="#00ACC1", ls=":",  lw=1.5),
    "Linear Demand (Orth)":                   dict(color="#006064", ls=":",  lw=1.5),
    "Neural Demand (static)":       dict(color="#1E88E5", ls="-.", lw=2.0),
    "Neural Demand (habit)":        dict(color="#00897B", ls="-",  lw=2.5),
    "Neural Demand (CF)":           dict(color="#283593", ls="--", lw=2.0),
    "Neural Demand (habit, CF)":    dict(color="#1B5E20", ls="--", lw=2.0),
}

# Good labels for the 3-good system (good 0 = Food, 1 = Fuel, 2 = Other)
_GOOD_LABEL  = ["Food", "Fuel", "Other"]
_GOOD_YLABEL = [
    "Food budget share",
    "Fuel budget share",
    "Other budget share",
]

# DGP display names (for heatmap rows)
_DGP_ORDER = ["CES", "Quasilinear", "Leontief", "Stone–Geary", "Habit", "Endogenous CES"]

# Models to highlight in the RMSE heatmap (subset for clarity)
_HEATMAP_MODELS = [
    "LA-AIDS",
    "Linear Demand (Orth)",
    "Neural Demand (static)",
    "Neural Demand (habit)",
    "Neural Demand (habit, CF)",
]
_HEATMAP_COL_LABELS = [
    "AIDS",
    "Linear Demand (Orth)",
    "Neural Demand (static)",
    "Neural Demand (habit)",
    "Neural Demand (habit, CF)",
]


def _save_fig(fig, fig_dir: str, stem: str) -> None:
    """Save figure to both PDF and PNG with tight layout."""
    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [paper fig] Saved: {fig_dir}/{stem}.pdf/png")


def _plot_rmse_heatmap(agg1: dict, fig_dir: str) -> None:
    """Heatmap of out-of-sample RMSE: rows = DGPs, columns = selected models."""
    acc = agg1.get("acc_agg")
    if acc is None or acc.empty:
        return

    n_runs = agg1.get("n_runs", 1)
    dgp_order  = [d for d in _DGP_ORDER if d in acc["DGP"].values]
    model_keys = [m for m in _HEATMAP_MODELS if m in acc["Model"].values]
    col_labels = [_HEATMAP_COL_LABELS[_HEATMAP_MODELS.index(m)] for m in model_keys]

    nrows, ncols = len(dgp_order), len(model_keys)
    rmse_mat = np.full((nrows, ncols), np.nan)
    se_mat   = np.full((nrows, ncols), np.nan)

    for r, dgp in enumerate(dgp_order):
        for c, mn in enumerate(model_keys):
            sub = acc.loc[(acc["DGP"] == dgp) & (acc["Model"] == mn)]
            if len(sub):
                rmse_mat[r, c] = float(sub["RMSE"].iloc[0])
                se_mat[r, c]   = float(sub["RMSE_se"].iloc[0])

    fig, ax = plt.subplots(figsize=(max(5, ncols * 1.8), max(3.5, nrows * 0.9)))
    vmax = np.nanmax(rmse_mat)
    cmap = plt.get_cmap("RdYlGn_r")
    im   = ax.imshow(rmse_mat, cmap=cmap, aspect="auto",
                     vmin=0.0, vmax=vmax)

    # Annotations
    for r in range(nrows):
        for c in range(ncols):
            v = rmse_mat[r, c]
            s = se_mat[r, c]
            if np.isnan(v):
                txt = "---"
            elif n_runs > 1 and not np.isnan(s):
                txt = f"{v:.4f}\n({s:.4f})"
            else:
                txt = f"{v:.4f}"
            contrast = "white" if (v / vmax if vmax > 0 else 0) > 0.55 else "black"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=8.5, color=contrast, fontweight="bold")

    ax.set_xticks(range(ncols))
    # Rotate labels 45 degrees and align them to the left
    ax.set_xticklabels(
        col_labels, 
        fontsize=10, 
        rotation=45, 
        ha="left", 
        rotation_mode="anchor"
    )
    
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(dgp_order, fontsize=10)
    
    # Ensure the labels appear at the top
    ax.tick_params(axis="x", top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean RMSE", fontsize=9)

    se_note = f" · mean (SE) over {n_runs} seeds" if n_runs > 1 else ""
    # ax.set_title(f"Out-of-Sample RMSE by DGP and Model{se_note}",
    #              fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    _save_fig(fig, fig_dir, "paper_rmse_heatmap")


def _plot_demand_curves_by_good(agg2: dict, fig_dir: str,
                                dgp_label: str = "Habit") -> None:
    """Save one figure per good — budget share vs good-1 (Fuel) price."""
    curves_mean = agg2.get("curves_all_mean", {})
    curves_se   = agg2.get("curves_all_se",   {})
    n_runs      = agg2.get("n_runs", 1)

    if not curves_mean:
        return

    p_grid  = np.linspace(1, 10, 80)
    ordered = ["Truth"] + [k for k in curves_mean.keys() if k != "Truth"]
    if dgp_label == "Habit":
        ordered = [k for k in ordered if k not in ("Linear Demand (Shared)", "Linear Demand (GoodSpec)")]

    for good_idx, ylabel in enumerate(_GOOD_YLABEL):
        fig, ax = plt.subplots(figsize=(8, 5))

        for lbl in ordered:
            if lbl not in curves_mean:
                continue
            mu_all = curves_mean[lbl]
            se_all = curves_se.get(lbl, np.zeros_like(mu_all))
            if mu_all.ndim < 2 or mu_all.shape[1] <= good_idx:
                continue
            mu  = mu_all[:, good_idx]
            sig = se_all[:, good_idx]

            disp = _MODEL_DISPLAY.get(lbl, lbl)
            sty  = _PAPER_STYLE.get(disp, dict(color="#888", ls="--", lw=1.5))

            ax.plot(p_grid, mu, label=disp, **sty)
            if n_runs > 1:
                ax.fill_between(p_grid, mu - sig, mu + sig,
                                color=sty["color"], alpha=0.12)

        ax.set_xlabel(r"Good-1 (Fuel) price $p_1$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=8.5, ncol=1, loc="best",
                  framealpha=0.9, edgecolor="#ccc")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        stem = f"paper_demand_curves_{dgp_label.lower()}_good{good_idx}"
        _save_fig(fig, fig_dir, stem)


def _plot_elasticity_heatmaps(agg1: dict, fig_dir: str) -> None:
    """Cross-price elasticity heatmaps, one panel per model (CES DGP)."""
    elast_mean = agg1.get("cross_elast_mean", {})
    elast_se   = agg1.get("cross_elast_se",   {})
    n_runs     = agg1.get("n_runs", 1)

    if not elast_mean:
        return

    model_order = ["Ground Truth", "LA-AIDS", "QUAIDS", "Neural Demand (static)"]
    models_avail = [m for m in model_order if m in elast_mean]
    if not models_avail:
        return

    n_panels = len(models_avail)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # Diverging colormap: blue = complement (negative), red = substitute (positive)
    # Diagonal (own-price) is always negative → shown in blue
    vabs = max(
        abs(np.nanmax([elast_mean[m] for m in models_avail])),
        abs(np.nanmin([elast_mean[m] for m in models_avail])),
        0.1,
    )
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    good_labels = _GOOD_LABEL

    for ax, mn in zip(axes, models_avail):
        mat = elast_mean[mn]
        se  = elast_se.get(mn, np.zeros_like(mat))

        im  = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")

        for i in range(3):
            for j in range(3):
                v = mat[i, j]; s = se[i, j]
                sign = "+" if v >= 0 else ""
                if n_runs > 1 and not np.isnan(s):
                    txt = f"{sign}{v:.2f}\n({s:.2f})"
                else:
                    txt = f"{sign}{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")

        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{g}\n($w_{j}$)" for j, g in enumerate(good_labels)],
                           fontsize=9)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"{g}\n($p_{i}$)" for i, g in enumerate(good_labels)],
                           fontsize=9)
        disp_title = _MODEL_DISPLAY.get(mn, mn)
        ax.set_title(disp_title, fontsize=10, fontweight="bold")

        if mn == models_avail[-1]:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r"$\varepsilon_{ij} = \partial\log x_j/\partial\log p_i$",
                           fontsize=8)

    se_note = f" · mean (SE), {n_runs} seeds" if n_runs > 1 else ""
    # fig.suptitle(
    #     "Cross-Price Demand Elasticity Heatmaps — CES Ground Truth\n"
    #     r"Diagonal = own-price ($<0$)  ·  Off-diagonal = cross-price"
    #     f"{se_note}",
    #     fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, fig_dir, "paper_elasticity_heatmaps")


def _plot_profile_kl(agg2: dict, agg3: dict | None, fig_dir: str) -> None:
    """Profile KL divergence over the δ grid (from exp02 and optionally exp03)."""
    TEAL   = "#00897B"
    ORANGE = "#FB8C00"
    RED    = "#E53935"

    # Prefer the finer exp03 grid if available
    if agg3 is not None and "kl_mean" in agg3:
        dg      = agg3["delta_grid"]
        mu      = agg3["kl_mean"]
        se      = agg3["kl_se"]
        d_hat   = agg3.get("delta_hat_mean", None)
        n_runs  = agg3.get("n_runs", 1)
        id_lo   = agg3.get("id_set_lo_mean", None)
        id_hi   = agg3.get("id_set_hi_mean", None)
    else:
        dg      = agg2.get("delta_grid", np.array([]))
        mu      = agg2.get("kl_prof_mean", np.zeros_like(dg))
        se      = agg2.get("kl_prof_se",  np.zeros_like(dg))
        d_hat   = agg2.get("delta_hat_mean", None)
        n_runs  = agg2.get("n_runs", 1)
        id_lo   = agg2.get("id_set_lo", None)
        id_hi   = agg2.get("id_set_hi", None)

    true_delta = (agg2 or agg3).get("true_delta", 0.7)
    if agg2 is not None:
        true_delta = agg2.get("true_delta", 0.7)

    if len(dg) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dg, mu, color=TEAL, lw=2.5, label="Val. KL (mean)")
    if n_runs > 1:
        ax.fill_between(dg, mu - se, mu + se, color=TEAL, alpha=0.15,
                        label="±1 SE")

    ax.axvline(true_delta, color=RED, ls="--", lw=2.0,
               label=f"True δ = {true_delta:.1f}")
    if d_hat is not None:
        ax.axvline(d_hat, color=ORANGE, ls=":", lw=2.0,
                   label=f"δ̂ = {d_hat:.2f} (mean)")

    if id_lo is not None and id_hi is not None:
        ax.axvspan(id_lo, id_hi, color=TEAL, alpha=0.08,
                   label=f"Identified set [{id_lo:.2f}, {id_hi:.2f}]")

    ax.set_xlabel("Habit-decay parameter δ", fontsize=13)
    ax.set_ylabel("Validation KL divergence", fontsize=13)
    se_note = f"  ({n_runs} seeds, ±1 SE)" if n_runs > 1 else ""
    # ax.set_title(f"Profile KL — δ Non-Identification{se_note}",
                #  fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_dir, "paper_profile_kl")


def _plot_cf_endogeneity(agg4: dict, fig_dir: str) -> None:
    """RMSE vs endogeneity strength ρ for CF-corrected and uncorrected models."""
    df     = agg4.get("agg")
    n_runs = agg4.get("n_runs", 1)
    if df is None or df.empty:
        return

    rho_grid  = sorted(df["rho"].unique())
    dgp_names = list(df["DGP"].unique())

    _style_cf = {
        "Neural Demand (static)":       dict(color="#1E88E5", ls="-",  lw=2.0,
                                             marker="o", ms=6,
                                             label="Neural Demand (static)"),
        "Neural Demand (CF)":           dict(color="#283593", ls="--", lw=2.0,
                                             marker="s", ms=6,
                                             label="Neural Demand (CF)"),
        "Neural Demand (habit)":        dict(color="#00897B", ls="-",  lw=2.0,
                                             marker="^", ms=6,
                                             label="Neural Demand (habit)"),
        "Neural Demand (habit, CF)":    dict(color="#1B5E20", ls="--", lw=2.0,
                                             marker="v", ms=6,
                                             label="Neural Demand (habit, CF)"),
    }

    model_order = list(_style_cf.keys())

    fig, axes = plt.subplots(1, len(dgp_names),
                             figsize=(6 * len(dgp_names), 5), sharey=False)
    if len(dgp_names) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgp_names):
        sub = df[df["DGP"] == dgp]
        for nm in model_order:
            ms  = sub[sub["Model"] == nm]
            if ms.empty:
                continue
            sty = _style_cf[nm].copy()
            lbl = sty.pop("label")
            ys  = [float(ms[ms["rho"] == r]["RMSE"].mean())    for r in rho_grid]
            ses = [float(ms[ms["rho"] == r]["RMSE_se"].mean()) for r in rho_grid]
            ax.plot(rho_grid, ys, label=lbl, **sty)
            if n_runs > 1:
                ax.fill_between(rho_grid,
                                [y - s for y, s in zip(ys, ses)],
                                [y + s for y, s in zip(ys, ses)],
                                color=sty["color"], alpha=0.15)

        ax.set_xlabel("Endogeneity strength ρ", fontsize=12)
        ax.set_ylabel("RMSE vs structural ground truth", fontsize=11)
        # ax.set_title(f"{dgp} DGP", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    se_note = f" ({n_runs} seeds)" if n_runs > 1 else ""
    # fig.suptitle(
    #     f"Control-Function Endogeneity Correction — RMSE vs ρ{se_note}",
    #     fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, fig_dir, "paper_cf_endogeneity")


def _plot_habit_rmse_bar(agg2: dict, fig_dir: str) -> None:
    """Bar chart: post-shock RMSE for all models on the Habit DGP."""
    rmse_agg = agg2.get("rmse_agg", {})
    n_runs   = agg2.get("n_runs", 1)
    if not rmse_agg:
        return

    model_names = list(rmse_agg.keys())
    model_names = [m for m in model_names if m not in ("Linear Demand (Shared)", "Linear Demand (GoodSpec)")]
    means = [rmse_agg[nm]["mean"] for nm in model_names]
    ses   = [rmse_agg[nm]["se"]   for nm in model_names]
    disps = [_MODEL_DISPLAY.get(nm, nm) for nm in model_names]

    # colors from paper style
    colors = [_PAPER_STYLE.get(d, {}).get("color", "#888888") for d in disps]

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.2), 5))
    x = np.arange(len(model_names))
    bars = ax.bar(x, [m if not np.isnan(m) else 0 for m in means],
                  color=colors, edgecolor="k", alpha=0.85,
                  yerr=[s if n_runs > 1 and not np.isnan(s) else 0 for s in ses],
                  capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(disps, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Post-shock RMSE (Habit DGP)", fontsize=12)
    se_note = f"  ({n_runs} seeds, ±1 SE)" if n_runs > 1 else ""
    # ax.set_title(f"Model Comparison on Habit DGP{se_note}",
    #              fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_dir, "paper_habit_rmse_bar")


def _plot_convergence(agg1: dict | None, agg2: dict | None, fig_dir: str) -> None:
    """Training convergence plots for all neural demand models.

    Produces two figures:

    paper_convergence_habit_dgp
        One subplot per neural model (2 × 2 grid), showing training KL
        divergence vs epoch on the Habit DGP.  Mean ± 1 SE across seeds
        (from Exp 02).

    paper_convergence_by_dgp
        One subplot per DGP (rows) × neural model (columns).  Mean ± 1 SE
        across seeds (from Exp 01).
    """
    # ── colour / style palette aligned with _PAPER_STYLE ─────────────────────
    _CONV_STYLE = {
        "Neural Demand (static)":    dict(color="#1E88E5", ls="-",  lw=2.0),
        "Neural Demand (habit)":     dict(color="#00897B", ls="-",  lw=2.0),
        "Neural Demand (CF)":        dict(color="#283593", ls="--", lw=1.8),
        "Neural Demand (habit, CF)": dict(color="#1B5E20", ls="--", lw=1.8),
    }

    # ── Fig A: Habit DGP — all 4 models, 2×2 grid (from Exp 02) ──────────────
    if agg2 is not None:
        train_conv2 = agg2.get("train_conv", {})
        n_runs2     = agg2.get("n_runs", 1)
        models_2x2  = list(_CONV_STYLE.keys())
        available   = [m for m in models_2x2 if m in train_conv2
                       and len(train_conv2[m].get("epochs", [])) > 0]

        if available:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
            axes_flat  = axes.flatten()

            for i, mn in enumerate(models_2x2):
                ax = axes_flat[i]
                if mn not in available:
                    ax.set_visible(False)
                    continue
                h    = train_conv2[mn]
                eps  = h["epochs"]
                mu   = h["kl_mean"]
                se   = h["kl_se"]
                sty  = _CONV_STYLE[mn]

                ax.plot(eps, mu, label="Mean KL", **sty)
                if n_runs2 > 1:
                    ax.fill_between(eps, mu - se, mu + se,
                                    color=sty["color"], alpha=0.15,
                                    label="±1 SE")

                # Mark best epoch (minimum KL)
                best_i = int(np.argmin(mu))
                ax.axvline(eps[best_i], color=sty["color"], ls=":",
                           lw=1.2, alpha=0.7,
                           label=f"Best ep={eps[best_i]:.0f}")
                ax.scatter([eps[best_i]], [mu[best_i]],
                           color=sty["color"], s=50, zorder=5)

                ax.set_title(_MODEL_DISPLAY.get(mn, mn), fontsize=10,
                             fontweight="bold")
                ax.set_xlabel("Epoch", fontsize=9)
                ax.set_ylabel("Training KL divergence", fontsize=9)
                ax.legend(fontsize=7.5, loc="upper right")
                ax.grid(True, alpha=0.3)

            se_note = f"  ({n_runs2} seeds, ±1 SE)" if n_runs2 > 1 else ""
            fig.suptitle(f"Training Convergence — Habit DGP{se_note}",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            _save_fig(fig, fig_dir, "paper_convergence_habit_dgp")
        else:
            print("  [warn] No convergence histories available from Exp 02.")

    # ── Fig B: All DGPs — one row per DGP, one col per neural model (Exp 01) ──
    if agg1 is not None:
        train_conv1 = agg1.get("train_conv", {})
        n_runs1     = agg1.get("n_runs", 1)
        dgp_order   = [d for d in ["CES", "Quasilinear", "Leontief",
                                    "Stone–Geary", "Habit", "Endogenous CES"]
                       if d in train_conv1]
        nd_models   = list(_CONV_STYLE.keys())

        # Keep only columns where at least one DGP has data
        col_models  = [mn for mn in nd_models
                       if any(len(train_conv1[d].get(mn, {}).get("epochs", [])) > 0
                              for d in dgp_order)]

        if dgp_order and col_models:
            nrows = len(dgp_order)
            ncols = len(col_models)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(4.5 * ncols, 3.5 * nrows),
                                     sharex=False, sharey=False,
                                     squeeze=False)

            for ri, dgp in enumerate(dgp_order):
                dgp_conv = train_conv1[dgp]
                for ci, mn in enumerate(col_models):
                    ax  = axes[ri][ci]
                    h   = dgp_conv.get(mn, {})
                    eps = h.get("epochs", np.array([]))
                    mu  = h.get("kl_mean", np.array([]))
                    se  = h.get("kl_se",   np.array([]))
                    sty = _CONV_STYLE[mn]

                    if len(eps) == 0:
                        ax.set_visible(False)
                        continue

                    ax.plot(eps, mu, **sty)
                    if n_runs1 > 1:
                        ax.fill_between(eps, mu - se, mu + se,
                                        color=sty["color"], alpha=0.15)

                    best_i = int(np.argmin(mu))
                    ax.axvline(eps[best_i], color=sty["color"], ls=":",
                               lw=1.0, alpha=0.6)
                    ax.scatter([eps[best_i]], [mu[best_i]],
                               color=sty["color"], s=30, zorder=5)

                    ax.set_xlabel("Epoch", fontsize=8)
                    ax.set_ylabel("Training KL", fontsize=8)
                    ax.grid(True, alpha=0.3)

                    # Row label (DGP) on left-most column
                    if ci == 0:
                        ax.set_ylabel(f"{dgp}\nTraining KL", fontsize=9,
                                      fontweight="bold")
                    # Column header on top row
                    if ri == 0:
                        ax.set_title(_MODEL_DISPLAY.get(mn, mn),
                                     fontsize=9, fontweight="bold")

            se_note = f"  ({n_runs1} seeds, ±1 SE)" if n_runs1 > 1 else ""
            fig.suptitle(f"Training Convergence by DGP{se_note}",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            _save_fig(fig, fig_dir, "paper_convergence_by_dgp")
        else:
            print("  [warn] No per-DGP convergence histories available from Exp 01.")


def make_paper_figures(results: dict, cfg: dict) -> None:
    """Generate summary paper figures from all experiment results.

    Called at the end of main() after all experiments complete.
    Produces figures in cfg["fig_dir"] with prefix ``paper_``.

    Figures generated
    -----------------
    paper_rmse_heatmap              Out-of-sample RMSE by DGP and model (Exp 01)
    paper_demand_curves_habit       Demand curves for all 3 goods — Habit DGP (Exp 02)
    paper_demand_curves_ces         Demand curves for all 3 goods — CES DGP (Exp 01)
    paper_elasticity_heatmaps       Cross-price elasticity heatmaps — CES (Exp 01)
    paper_profile_kl                Profile KL for δ identification (Exp 02/03)
    paper_cf_endogeneity            RMSE vs ρ for CF correction (Exp 04)
    paper_habit_rmse_bar            RMSE bar chart — Habit DGP (Exp 02)
    paper_convergence_habit_dgp     Training KL curves — 4 neural models on Habit DGP (Exp 02)
    paper_convergence_by_dgp        Training KL curves — 4 neural models × all DGPs (Exp 01)
    """
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "─" * 68)
    print("  Generating paper-style summary figures …")
    print("─" * 68)

    # Unpack experiment outputs  (each is a (all_results, agg) tuple or None)
    exp01 = results.get("01")
    exp02 = results.get("02")
    exp03 = results.get("03")
    exp04 = results.get("04")

    agg1 = exp01[1] if exp01 is not None else None
    agg2 = exp02[1] if exp02 is not None else None
    agg3 = exp03[1] if exp03 is not None else None
    agg4 = exp04[1] if exp04 is not None else None

    # ── Fig 1: RMSE heatmap ────────────────────────────────────────────────────
    if agg1 is not None:
        try:
            _plot_rmse_heatmap(agg1, fig_dir)
        except Exception as e:
            print(f"  [warn] RMSE heatmap failed: {e}")

    # ── Fig 2: Demand curves — Habit DGP (all 3 goods) ────────────────────────
    if agg2 is not None:
        try:
            _plot_demand_curves_by_good(agg2, fig_dir, dgp_label="Habit")
        except Exception as e:
            print(f"  [warn] Habit demand curves failed: {e}")

    # ── Fig 3: Demand curves — CES DGP (all 3 goods) ──────────────────────────
    if agg1 is not None and agg1.get("curves_ces_full_mean"):
        # Build a fake agg dict matching the exp02 interface so we can reuse
        # _plot_demand_curves_by_good
        _ces_agg = {
            "curves_all_mean": agg1["curves_ces_full_mean"],
            "curves_all_se":   agg1.get("curves_ces_full_se", {}),
            "n_runs":          agg1.get("n_runs", 1),
        }
        try:
            _plot_demand_curves_by_good(_ces_agg, fig_dir, dgp_label="CES")
        except Exception as e:
            print(f"  [warn] CES demand curves failed: {e}")

    # ── Fig 4: Cross-price elasticity heatmaps — CES DGP ─────────────────────
    if agg1 is not None and agg1.get("cross_elast_mean"):
        try:
            _plot_elasticity_heatmaps(agg1, fig_dir)
        except Exception as e:
            print(f"  [warn] Elasticity heatmaps failed: {e}")

    # ── Fig 5: Profile KL for δ ───────────────────────────────────────────────
    if agg2 is not None:
        try:
            _plot_profile_kl(agg2, agg3, fig_dir)
        except Exception as e:
            print(f"  [warn] Profile KL failed: {e}")

    # ── Fig 6: CF endogeneity robustness ──────────────────────────────────────
    if agg4 is not None:
        try:
            _plot_cf_endogeneity(agg4, fig_dir)
        except Exception as e:
            print(f"  [warn] CF endogeneity plot failed: {e}")

    # ── Fig 7: Habit RMSE bar chart ────────────────────────────────────────────
    if agg2 is not None:
        try:
            _plot_habit_rmse_bar(agg2, fig_dir)
        except Exception as e:
            print(f"  [warn] Habit RMSE bar chart failed: {e}")

    # ── Fig 9: Training convergence curves ────────────────────────────────────
    try:
        _plot_convergence(agg1, agg2, fig_dir)
    except Exception as e:
        print(f"  [warn] Convergence plots failed: {e}")

    print("─" * 68)
    print(f"  Paper figures saved to: {os.path.abspath(fig_dir)}/")
    print("─" * 68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # Build config
    cfg = dict(BASE_CFG)
    if args.fast:
        cfg.update(FAST_CFG)
        cfg["FAST_DELTA_GRID"] = np.array([0.3, 0.5, 0.7, 0.9])
        cfg["model_cache_dir"] = "results/neural_demand/simulations/models/fast"
        print("[fast mode] Using reduced N_OBS / EPOCHS / N_RUNS / DELTA_GRID")
    else:
        cfg["model_cache_dir"] = "results/neural_demand/simulations/models/full"

    # If --load is NOT set, force retraining (ignore existing cache but save new models)
    cfg["force_retrain"] = not args.load

    if args.epochs is not None:
        cfg["EPOCHS"] = args.epochs
        print(f"[epochs override] Using EPOCHS={args.epochs}")

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
    print("  Neural Demand — Simulation Experiments")
    print(f"  Device : {DEVICE}")
    print(f"  N_RUNS : {cfg['N_RUNS']}   N_OBS : {cfg['N_OBS']}   EPOCHS : {cfg['EPOCHS']}")
    print(f"  Running: {', '.join(exps_to_run)}")
    print("#" * 68 + "\n")

    # Precompute all seed-level training runs upfront; experiment run() functions
    # consume these via cfg to avoid retraining.
    seeds = [42 + ri * 15 for ri in range(cfg["N_RUNS"])]
    if exps_to_run:
        print("[Precompute] Building simulation seed cache...")
    if "01" in exps_to_run:
        from experiments.simulation.exp01_dgp_recovery import run_one_seed as _s01
        cfg["_precomputed_sim_exp01"] = [
            _s01(seed, cfg, verbose=(i == len(seeds) - 1))
            for i, seed in enumerate(seeds)
        ]
    if "02" in exps_to_run:
        from experiments.simulation.exp02_habit_advantage import run_one_seed as _s02
        cfg["_precomputed_sim_exp02"] = [
            _s02(seed, cfg, verbose=(i == len(seeds) - 1))
            for i, seed in enumerate(seeds)
        ]
    if "03" in exps_to_run:
        from experiments.simulation.exp03_delta_identification import run_one_seed as _s03
        id_cfg = dict(cfg)
        id_cfg["DELTA_GRID"] = np.linspace(0.1, 0.95, 30)
        if "FAST_DELTA_GRID" in cfg:
            id_cfg["DELTA_GRID"] = cfg["FAST_DELTA_GRID"]
        cfg["_precomputed_sim_exp03"] = [
            _s03(seed, id_cfg, verbose=(i == len(seeds) - 1))
            for i, seed in enumerate(seeds)
        ]
    if "04" in exps_to_run:
        from experiments.simulation.exp04_cf_endogeneity import run_one_seed as _s04
        cfg["_precomputed_sim_exp04"] = [
            _s04(seed, cfg, verbose=(i == len(seeds) - 1))
            for i, seed in enumerate(seeds)
        ]

    t_total = time.time()
    results  = {}
    for key in exps_to_run:
        name, fn = EXPERIMENTS[key]
        t0 = time.time()
        try:
            results[key] = fn(cfg)
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] Experiment {key} ({name}) failed:")
            traceback.print_exc()
            results[key] = None
        elapsed = time.time() - t0
        status  = "OK" if results[key] is not None else "FAILED"
        print(f"\n[{key}] {name}: {status} in {elapsed:.0f}s")

    # ── Paper-style summary figures (combining all experiment results) ───────
    try:
        make_paper_figures(results, cfg)
    except Exception as exc:
        import traceback
        print("\n[warn] Paper figure generation failed:")
        traceback.print_exc()

    print("\n" + "#" * 68)
    print(f"  All done in {time.time()-t_total:.0f}s")
    print(f"  Outputs → {cfg['out_dir']}")
    print("#" * 68)
    return results


if __name__ == "__main__":
    main()
