"""
experiments/neural_demand/dominicks/exp05_delta_identification.py
==================================================================
Section 5.5 — Non-Identification of δ on Dominick's Analgesics.

Profiles the held-out KL divergence over a grid of frozen δ values to
characterise the identified set on real data.

Produces
--------
results/neural_demand/dominicks/
  table_dom_delta_identification.csv / .tex
  fig_dom_delta_profile_kl.{pdf,png}
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

from experiments.dominicks.utils import fit_mdp_delta_grid_dom
from experiments.dominicks.utils import BAND

warnings.filterwarnings("ignore")

TEAL   = "#00897B"
ORANGE = "#FB8C00"
RED    = "#E53935"


# ─────────────────────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int, splits: dict, cfg: dict) -> dict:
    """Profile KL over δ grid for one seed on Dominick's data."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    p_tr  = splits["p_tr"];  p_te  = splits["p_te"]
    w_tr  = splits["w_tr"];  w_te  = splits["w_te"]
    y_tr  = splits["y_tr"];  y_te  = splits["y_te"]
    ls_tr = splits["ls_tr"]; ls_te = splits["ls_te"]
    s_tr  = splits["s_tr"];  s_te  = splits["s_te"]

    delta_grid = cfg["delta_grid_identification"]

    sw = fit_mdp_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg,
        delta_grid=delta_grid,
        store_ids_tr=s_tr, store_ids_val=s_te,
        hidden=cfg["mdp_e2e_hidden"],
        pred_spec="mdp",
        pred_model_key="mdp",
        tag=f"dom-delta-id-s{seed}",
    )

    return dict(
        delta_grid=delta_grid,
        kl_grid=sw["kl_grid"],
        se_grid=sw["se_grid"],
        delta_hat=sw["delta_hat"],
        id_set=sw["id_set"],
        id_mask=sw["id_mask"],
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray([x for x in arr if not np.isnan(float(x))], float)
    return float(np.std(a, ddof=1) / np.sqrt(max(len(a), 1))) if len(a) > 1 else 0.0


def aggregate(all_results: list) -> dict:
    n  = len(all_results)
    dg = all_results[0]["delta_grid"]

    kl_stack = np.stack([r["kl_grid"] for r in all_results], 0)
    se_stack = np.stack([r["se_grid"] for r in all_results], 0)

    delta_hats = [r["delta_hat"] for r in all_results]
    id_lo      = [r["id_set"][0] for r in all_results]
    id_hi      = [r["id_set"][1] for r in all_results]
    id_widths  = [hi - lo for lo, hi in zip(id_lo, id_hi)]

    return dict(
        delta_grid=dg,
        kl_mean=kl_stack.mean(0),
        kl_se=kl_stack.std(0, ddof=max(1, n-1)) / np.sqrt(n),
        delta_hat_mean=float(np.nanmean(delta_hats)),
        delta_hat_se=_se(delta_hats),
        delta_hat_all=delta_hats,
        id_set_lo_mean=float(np.nanmean(id_lo)),
        id_set_hi_mean=float(np.nanmean(id_hi)),
        id_width_mean=float(np.nanmean(id_widths)),
        id_width_se=_se(id_widths),
        n_runs=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(agg: dict, cfg: dict) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    n_runs  = agg["n_runs"]
    se_note = f"  ({n_runs} seeds, ±1 SE)" if n_runs > 1 else ""

    dg = agg["delta_grid"]
    mu = agg["kl_mean"]
    se = agg["kl_se"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dg, mu, color=TEAL, lw=2.5, marker="o", ms=5, label="Val. KL (mean)")
    if n_runs > 1:
        ax.fill_between(dg, mu - se, mu + se, color=TEAL, alpha=BAND)
    ax.axvline(agg["delta_hat_mean"], color=ORANGE, ls=":", lw=2,
               label=f"δ̂ = {agg['delta_hat_mean']:.2f} (mean)")
    lo, hi = agg["id_set_lo_mean"], agg["id_set_hi_mean"]
    ax.axvspan(lo, hi, color=TEAL, alpha=0.08,
               label=f"IS [{lo:.2f}, {hi:.2f}]")
    ax.set_xlabel("Habit-decay parameter δ", fontsize=13)
    ax.set_ylabel("Test-set KL divergence", fontsize=13)
    # ax.set_title(f"Profile KL — δ on Dominick's Data{se_note}",
    #              fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_dir}/fig_dom_delta_profile_kl.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig_dom_delta_profile_kl.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def make_tables(agg: dict, cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    n_runs  = agg["n_runs"]
    os.makedirs(out_dir, exist_ok=True)

    dg = agg["delta_grid"]
    pd.DataFrame({"delta": dg,
                  "kl_mean": agg["kl_mean"],
                  "kl_se":   agg["kl_se"]}).round(8).to_csv(
        f"{out_dir}/table_dom_delta_identification.csv", index=False)
    print(f"  Saved: {out_dir}/table_dom_delta_identification.csv")

    summary = {
        "δ̂ (mean ± SE)":     f"{agg['delta_hat_mean']:.3f} ± {agg['delta_hat_se']:.3f}",
        "IS lo (mean)":       f"{agg['id_set_lo_mean']:.3f}",
        "IS hi (mean)":       f"{agg['id_set_hi_mean']:.3f}",
        "IS width (mean±SE)": f"{agg['id_width_mean']:.3f} ± {agg['id_width_se']:.3f}",
    }
    lines = [
        r"% ============================================================",
        r"% Neural Demand — Dominick's δ Identification (auto-generated)",
        r"% ============================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Non-Identification of $\delta$ --- Dominick's Analgesics"
        rf"  ({n_runs} seeds)}}",
        r"  \label{tab:dom_delta_identification}",
        r"  \begin{tabular}{lc}",
        r"    \toprule",
        r"    \textbf{Quantity} & \textbf{Value} \\",
        r"    \midrule",
    ]
    for k, v in summary.items():
        lines.append(f"    {k} & {v} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    tex_path = f"{out_dir}/table_dom_delta_identification.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(splits: dict, cfg: dict) -> tuple:
    N_RUNS = cfg.get("N_RUNS", 1)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Dominick's Exp 05: δ Identification")
    print("=" * 68)

    pre = cfg.get("_precomputed_dom_exp05")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed exp05_delta_identification runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = run_once(seed, splits, cfg)
            all_results.append(r)
            print(f"    Done in {time.time()-t0:.0f}s  "
                  f"δ̂={r['delta_hat']:.2f}  "
                  f"IS=[{r['id_set'][0]:.2f}, {r['id_set'][1]:.2f}]")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    make_tables(agg, cfg)

    print(f"\n── Dominick's δ Identification Summary ────────────────────────────")
    print(f"  δ̂ (mean ± SE) = {agg['delta_hat_mean']:.3f} ± {agg['delta_hat_se']:.3f}")
    print(f"  IS            = [{agg['id_set_lo_mean']:.3f}, {agg['id_set_hi_mean']:.3f}]"
          f"  width = {agg['id_width_mean']:.3f}\n")

    return all_results, agg
