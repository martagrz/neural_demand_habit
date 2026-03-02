"""
experiments/neural_demand/dominicks/exp04_demand_curves.py
===========================================================
Section 5.4 — Demand Curves on Dominick's Analgesics.

Plots budget shares as a function of each good's own price while holding
all other prices and income fixed at their test-set means.
Also generates demand decomposition: static neural demand vs. habit-augmented,
highlighting the habit contribution.

Produces
--------
results/neural_demand/dominicks/
  fig_dom_demand_curves_g{g}.{pdf,png}         — per-good demand curves
  fig_dom_demand_decomposition.{pdf,png}        — habit decomposition
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
    StaticND,
    HabitND,
        _train,
    cf_first_stage,
    compute_xbar_e2e,
    feat_shared, feat_good_specific, feat_orth,
    run_lirl,
)
from experiments.dominicks.data import G, GOODS
from experiments.dominicks.utils import mdp_price_cond_habit
from experiments.dominicks.data import G as _G
from experiments.dominicks.utils import (
    predict,
    fit_nd_delta_grid_dom,
    STYLE,
    BAND,
    demand_curve_plot,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int, splits: dict, cfg: dict) -> dict:
    """Compute demand curves for one seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    p_tr  = splits["p_tr"];  p_te  = splits["p_te"]
    w_tr  = splits["w_tr"];  w_te  = splits["w_te"]
    y_tr  = splits["y_tr"];  y_te  = splits["y_te"]
    xb_tr = splits["xb_tr"]; xb_te = splits["xb_te"]
    qp_tr = splits["qp_tr"]; qp_te = splits["qp_te"]
    ls_tr = splits["ls_tr"]; ls_te = splits["ls_te"]
    s_tr  = splits["s_tr"];  s_te  = splits["s_te"]
    Z_tr  = splits["Z_tr"]
    mw_tr = splits["mw_tr"]

    N_GR  = len(splits["pgr"])
    pgr_all = splits["pgr_all"]
    tpx_all = splits["tpx_all"]
    fy    = splits["fy"]
    dev   = cfg["device"]

    # ── Train ──────────────────────────────────────────────────────────────────
    aids_m   = LAAIDS().fit(p_tr, w_tr, y_tr)
    blp_m    = BLPLogitIV().fit(p_tr, mw_tr, Z_tr)
    quaids_m = QUAIDS().fit(p_tr, w_tr, y_tr)
    series_m = SeriesDemand().fit(p_tr, w_tr, y_tr)
    th_sh   = run_lirl(feat_shared,        p_tr, y_tr, w_tr, cfg)
    th_gs   = run_lirl(feat_good_specific, p_tr, y_tr, w_tr, cfg)
    th_orth = run_lirl(feat_orth,          p_tr, y_tr, w_tr, cfg)

    nirl_m, _ = _train(StaticND(cfg["nirl_hidden"]),
                       p_tr, y_tr, w_tr, "nirl", cfg,
                       tag=f"StaticND s={seed}")

    sw = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=False,
        store_ids_tr=s_tr, store_ids_val=s_te,
        hidden=cfg["mdp_e2e_hidden"],
        tag=f"ND-Habit s={seed}")
    nd_hab_m  = sw["best_model"]
    delta_hat = sw["delta_hat"]

    d_t     = torch.tensor(float(delta_hat), dtype=torch.float32, device=dev)
    ls_te_t = torch.tensor(ls_te, dtype=torch.float32, device=dev)
    with torch.no_grad():
        xb_e2e_te = compute_xbar_e2e(d_t, ls_te_t, store_ids=s_te).cpu().numpy()

    # ── CF first-stage + CF models ─────────────────────────────────────────────
    v_hat_tr, _ = cf_first_stage(
        np.log(np.maximum(p_tr, 1e-8)), Z_tr)
    nirl_cf_m, _ = _train(
        StaticND(cfg["nirl_hidden"], n_cf=_G),
        p_tr, y_tr, w_tr, "nirl", cfg,
        v_hat_tr=v_hat_tr,
        tag=f"StaticND-CF s={seed}")
    mdp_hidden = cfg.get("mdp_hidden", cfg["nirl_hidden"])
    mdp_cf_m, _ = _train(
        HabitND(mdp_hidden, n_cf=_G),
        p_tr, y_tr, w_tr, "mdp", cfg,
        xb_prev_tr=xb_tr, q_prev_tr=qp_tr,
        v_hat_tr=v_hat_tr,
        tag=f"ND-Habit-CF s={seed}")

    KW = dict(
        aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
        ff=feat_shared, theta=th_sh,
        nirl=nirl_m,
        nirl_cf=nirl_cf_m,
        mdp_cf=mdp_cf_m,
        mdp=nd_hab_m,
    )

    def _kw_lirl(nm):
        if nm == "Linear Demand (GoodSpec)":
            return {"ff": feat_good_specific, "theta": th_gs}
        if nm == "Linear Demand (Orth)":
            return {"ff": feat_orth, "theta": th_orth}
        return {}

    # ── Demand curves for each good ────────────────────────────────────────────
    curves_per_good = {}   # {good_idx: {model_name: share_array(N_GR,)}}
    for g in range(G):
        tpx_g = tpx_all[g]
        pgr_g = pgr_all[g]
        xbr_g, qpr_g   = mdp_price_cond_habit(pgr_g, g, p_te, xb_e2e_te, qp_te)
        # For CF habit model use precomputed EWMA xbar (from splits)
        xbr_cf, qpr_cf  = mdp_price_cond_habit(pgr_g, g, p_te, xb_te, qp_te)

        curves_g = {}
        for nm, xb_kw in [
            ("LA-AIDS",                     {}),
            ("BLP (IV)",                    {}),
            ("QUAIDS",                      {}),
            ("Series Est.",                 {}),
            ("Linear Demand (Shared)",                {}),
            ("Linear Demand (GoodSpec)",              {}),
            ("Linear Demand (Orth)",                  {}),
            ("Neural Demand (static)",      {}),
            ("Neural Demand (habit)",       {"xb_prev": xbr_g, "q_prev": qpr_g}),
            ("Neural Demand (CF)",          {}),
            ("Neural Demand (habit, CF)",   {"xb_prev": xbr_cf, "q_prev": qpr_cf}),
        ]:
            try:
                wp = predict(nm, tpx_g, fy, cfg, **xb_kw, **{**KW, **_kw_lirl(nm)})
                curves_g[nm] = wp[:, 0]   # budget share of good 0
            except Exception as exc:
                print(f"    Warning: curve {nm} good {g} failed — {exc}")
                curves_g[nm] = np.full(N_GR, np.nan)
        curves_per_good[g] = curves_g

    # ── Habit decomposition at shock_good ─────────────────────────────────────
    sg     = cfg["shock_good"]
    tpx_sg = tpx_all[sg]
    pgr_sg = pgr_all[sg]
    xbr_sg,  qpr_sg  = mdp_price_cond_habit(pgr_sg, sg, p_te, xb_e2e_te, qp_te)
    xbr_sg2, qpr_sg2 = mdp_price_cond_habit(pgr_sg, sg, p_te, xb_te, qp_te)

    # Static neural demand uses the same prices but zero habit stock
    w_static    = predict("Neural Demand (static)",     tpx_sg, fy, cfg, **KW)[:, 0]
    w_static_cf = predict("Neural Demand (CF)",     tpx_sg, fy, cfg, **KW)[:, 0]
    w_habit     = predict("Neural Demand (habit)",  tpx_sg, fy, cfg,
                          xb_prev=xbr_sg, q_prev=qpr_sg, **KW)[:, 0]
    w_habit_cf  = predict("Neural Demand (habit, CF)",  tpx_sg, fy, cfg,
                          xb_prev=xbr_sg2, q_prev=qpr_sg2, **KW)[:, 0]
    habit_contribution    = w_habit    - w_static
    cf_contribution       = w_static_cf - w_static
    habit_cf_contribution = w_habit_cf  - w_static

    return dict(
        curves_per_good=curves_per_good,
        pgr_all=pgr_all,
        w_static=w_static, w_static_cf=w_static_cf,
        w_habit=w_habit, w_habit_cf=w_habit_cf,
        habit_contribution=habit_contribution,
        cf_contribution=cf_contribution,
        habit_cf_contribution=habit_cf_contribution,
        pgr_sg=pgr_sg,
        shock_good=sg,
        delta_hat=delta_hat,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(all_results: list) -> dict:
    n   = len(all_results)
    _G  = len(all_results[0]["curves_per_good"])
    nms = list(all_results[0]["curves_per_good"][0].keys())

    curves_mean = {}
    curves_se   = {}
    for g in range(_G):
        curves_mean[g] = {nm: np.mean([r["curves_per_good"][g][nm]
                                       for r in all_results], 0)
                          for nm in nms}
        curves_se[g]   = {nm: (np.std([r["curves_per_good"][g][nm]
                                       for r in all_results], 0, ddof=max(1, n-1))
                               / np.sqrt(n))
                          for nm in nms}

    def _arr_mean(key):
        return np.mean([r[key] for r in all_results], 0)

    def _arr_se(key):
        return (np.std([r[key] for r in all_results], 0, ddof=max(1, n-1))
                / np.sqrt(n))

    return dict(
        curves_mean=curves_mean, curves_se=curves_se,
        pgr_all=all_results[0]["pgr_all"],
        w_static_mean=_arr_mean("w_static"),
        w_static_cf_mean=_arr_mean("w_static_cf"),
        w_habit_mean=_arr_mean("w_habit"),
        w_habit_cf_mean=_arr_mean("w_habit_cf"),
        habit_contribution_mean=_arr_mean("habit_contribution"),
        cf_contribution_mean=_arr_mean("cf_contribution"),
        habit_cf_contribution_mean=_arr_mean("habit_cf_contribution"),
        habit_contribution_se=_arr_se("habit_contribution"),
        cf_contribution_se=_arr_se("cf_contribution"),
        habit_cf_contribution_se=_arr_se("habit_cf_contribution"),
        pgr_sg=all_results[0]["pgr_sg"],
        shock_good=all_results[0]["shock_good"],
        n_runs=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(agg: dict, cfg: dict) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    n_runs  = agg["n_runs"]
    se_note = f"  ({n_runs} runs, ±1 SE)" if n_runs > 1 else ""

    # ── Per-good demand curves ─────────────────────────────────────────────────
    for g in range(G):
        pgr = agg["pgr_all"][g]
        fig, ax = plt.subplots(figsize=(10, 6))
        demand_curve_plot(
            agg["curves_mean"][g], pgr, shock_good_idx=g,
            ax=ax, n_runs=n_runs, ses=agg["curves_se"][g],
            title=f"Demand Curves — {GOODS[g]} price varies{se_note}",
        )
        ax.set_ylabel(f"Good-0 (ASP) budget share", fontsize=12)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"{fig_dir}/fig_dom_demand_curves_g{g}.{ext}",
                        dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_dir}/fig_dom_demand_curves_g{g}.pdf/png")

    # ── Habit decomposition ───────────────────────────────────────────────────
    sg    = agg["shock_good"]
    pgr   = agg["pgr_sg"]
    figD, axes_d = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: demand curves with all models
    ax1 = axes_d[0]
    for nm in ["Neural Demand (static)", "Neural Demand (habit)",
               "Neural Demand (CF)", "Neural Demand (habit, CF)",
               "LA-AIDS"]:
        mu  = agg["curves_mean"][sg].get(nm)
        if mu is None: continue
        sty = STYLE.get(nm, {})
        ax1.plot(pgr, mu, label=nm, **sty)
    ax1.set_xlabel(f"{GOODS[sg]} price", fontsize=12)
    ax1.set_ylabel("Good-0 budget share", fontsize=12)
    # ax1.set_title(f"All Models{se_note}", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)

    # Panel 2: habit contribution (Habit − Static)
    ax2 = axes_d[1]
    hc_mu = agg["habit_contribution_mean"]
    hc_se = agg["habit_contribution_se"]
    ax2.plot(pgr, hc_mu, color="#00897B", lw=2.5, label="ND+Habit − ND (static)")
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    if n_runs > 1:
        ax2.fill_between(pgr, hc_mu - hc_se, hc_mu + hc_se, color="#00897B", alpha=BAND)
    cf_mu  = agg["cf_contribution_mean"]
    cf_se  = agg["cf_contribution_se"]
    ax2.plot(pgr, cf_mu, color="#283593", lw=2.0, ls="--",
             label="ND(CF) − ND (static)")
    if n_runs > 1:
        ax2.fill_between(pgr, cf_mu - cf_se, cf_mu + cf_se, color="#283593", alpha=BAND)
    ax2.set_xlabel(f"{GOODS[sg]} price", fontsize=12)
    ax2.set_ylabel("Δ budget share", fontsize=12)
    # ax2.set_title(f"Habit & CF Contribution{se_note}", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: CF decomposition — static vs habit vs habit+CF
    ax3 = axes_d[2]
    for nm, key in [("Neural Demand (static)", "w_static_mean"),
                    ("Neural Demand (CF)",         "w_static_cf_mean"),
                    ("Neural Demand (habit, CF)",  "w_habit_cf_mean")]:
        mu  = agg[key]
        sty = STYLE.get(nm, {})
        ax3.plot(pgr, mu, label=nm, **sty)
    ax3.set_xlabel(f"{GOODS[sg]} price", fontsize=12)
    ax3.set_ylabel("Good-0 budget share", fontsize=12)
    # ax3.set_title(f"CF Decomposition{se_note}", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(True, alpha=0.3)

    # figD.suptitle(f"Demand Decomposition — {GOODS[sg]}",
    #               fontsize=12, fontweight="bold")
    figD.tight_layout()
    for ext in ("pdf", "png"):
        figD.savefig(f"{fig_dir}/fig_dom_demand_decomposition.{ext}",
                     dpi=150, bbox_inches="tight")
    plt.close(figD)
    print(f"  Saved: {fig_dir}/fig_dom_demand_decomposition.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(splits: dict, cfg: dict) -> tuple:
    N_RUNS = cfg.get("N_RUNS", 1)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Dominick's Exp 04: Demand Curves")
    print("=" * 68)

    pre = cfg.get("_precomputed_dom_exp04")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed exp04_demand_curves runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = run_once(seed, splits, cfg)
            all_results.append(r)
            hcb = r["habit_contribution"]
            print(f"    Done in {time.time()-t0:.0f}s  "
                  f"max |habit contrib|={np.abs(hcb).max():.4f}  "
                  f"δ̂={r['delta_hat']:.2f}")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    return all_results, agg
