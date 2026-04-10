"""
experiments/neural_demand/dominicks/exp02_elasticities.py
==========================================================
Section 5.2 — Price Elasticities on Dominick's Analgesics.

Computes own-price and cross-price elasticity matrices at the test-set mean
price vector for all Neural Demand paper models.

Produces
--------
results/neural_demand/dominicks/
  table_dom_own_elasticities.csv / .tex
  table_dom_cross_elasticities.csv / .tex
  fig_dom_elasticities.{pdf,png}
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

from src.models.dominicks import (
    LAAIDS, BLPLogitIV, QUAIDS, SeriesDemand,
    StaticND,
    StaticND_FE,
    HabitND,
            WindowND,
    _train,
    build_window_features,
    cf_first_stage,
    compute_xbar_e2e,
    feat_good_specific, feat_orth, feat_shared,
    run_lirl,
    train_window_irl,
)
from experiments.dominicks.data import G, GOODS
from experiments.dominicks.data import G as _G
from experiments.dominicks.utils import (
    own_elasticity,
    elasticity_matrix,
    fit_nd_delta_grid_dom,
    ALL_MODEL_NAMES,
    STYLE,
    make_elasticity_table,
    BAND,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int, splits: dict, cfg: dict) -> dict:
    """Compute elasticities for one seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    p_tr  = splits["p_tr"];  p_te  = splits["p_te"]
    w_tr  = splits["w_tr"];  w_te  = splits["w_te"]
    mw_tr = splits["mw_tr"]
    y_tr  = splits["y_tr"];  y_te  = splits["y_te"]
    xb_tr = splits["xb_tr"]; xb_te = splits["xb_te"]
    qp_tr = splits["qp_tr"]; qp_te = splits["qp_te"]
    ls_tr = splits["ls_tr"]; ls_te = splits["ls_te"]
    s_tr  = splits["s_tr"];  s_te  = splits["s_te"]
    Z_tr  = splits["Z_tr"]
    dev   = cfg["device"]

    N_STORES      = splits["N_STORES"]
    STORE_EMB_DIM = splits["STORE_EMB_DIM"]
    s_tr_idx      = splits["s_tr_idx"]
    s_te_idx      = splits["s_te_idx"]
    s_te_mode_idx = splits["s_te_mode_idx"]

    p_mn  = splits["p_mn"]
    y_mn  = float(splits["y_mn"])
    xb_mn = splits["xb_mn"]
    qp_mn = splits["qp_mn"]

    # ── Train ──────────────────────────────────────────────────────────────────
    aids_m   = LAAIDS().fit(p_tr, w_tr, y_tr)
    blp_m    = BLPLogitIV().fit(p_tr, mw_tr, Z_tr, y_tr)
    quaids_m = QUAIDS().fit(p_tr, w_tr, y_tr)
    series_m = SeriesDemand().fit(p_tr, w_tr, y_tr)

    th_sh   = run_lirl(feat_shared,        p_tr, y_tr, w_tr, cfg)
    th_gs   = run_lirl(feat_good_specific, p_tr, y_tr, w_tr, cfg)
    th_orth = run_lirl(feat_orth,          p_tr, y_tr, w_tr, cfg)

    nirl_m, _ = _train(StaticND(cfg["nirl_hidden"]),
                       p_tr, y_tr, w_tr, "nirl", cfg,
                       tag=f"StaticND s={seed}")

    # ── Window IRL ─────────────────────────────────────────────────────────────
    # Needs lagged (log p, log q) features; we build them from the sequential panel.
    WIRL_W = int(cfg.get("wirl_window", 4))
    lp_tr  = np.log(np.maximum(p_tr, 1e-8))
    ly_tr  = np.log(np.maximum(y_tr, 1e-8))
    q_tr   = w_tr * y_tr[:, None] / np.maximum(p_tr, 1e-8)
    lq_tr  = np.log(np.maximum(q_tr, 1e-6))
    wf_tr  = build_window_features(lp_tr, ly_tr, lq_tr, window=WIRL_W, store_ids=s_tr)

    wirl_m, _ = train_window_irl(
        WindowND(n_goods=G, hidden_dim=cfg["nirl_hidden"], window=WIRL_W),
        wf_tr, w_tr,
        epochs=cfg.get("nirl_epochs", 10000),
        lr=cfg.get("nirl_lr", 1e-4),
        batch_size=cfg.get("nirl_batch", 256),
        lam_mono=cfg.get("nirl_lam_mono", 0.2),
        lam_slut=cfg.get("nirl_lam_slut", 0.05),
        slut_start_frac=cfg.get("nirl_slut_start", 0.3),
        device=dev,
        verbose=bool(cfg.get("verbose", False)),
        tag=f"WindowND s={seed}",
        cache_dir=cfg.get("model_cache_dir"),
        force_retrain=bool(cfg.get("force_retrain", False)),
    )
    wirl_lp_mean = lp_tr.mean(0)
    wirl_lq_mean = lq_tr.mean(0)

    sw = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=False,
        store_ids_tr=s_tr, store_ids_val=s_te,
        hidden=cfg["mdp_e2e_hidden"],
        tag=f"ND-Habit s={seed}")
    nd_hab_m  = sw["best_model"]
    delta_hat = sw["delta_hat"]

    d_t = torch.tensor(float(delta_hat), dtype=torch.float32, device=dev)
    ls_te_t = torch.tensor(ls_te, dtype=torch.float32, device=dev)
    with torch.no_grad():
        xb_e2e_te = compute_xbar_e2e(d_t, ls_te_t, store_ids=s_te).cpu().numpy()

    # ── Store-FE variants ──────────────────────────────────────────────────────
    nirl_fe_m, _ = _train(
        StaticND_FE(cfg["nirl_hidden"], n_stores=N_STORES, emb_dim=STORE_EMB_DIM),
        p_tr, y_tr, w_tr, "nirl", cfg,
        store_idx_tr=s_tr_idx,
        tag=f"StaticND-FE s={seed}",
    )

    sw_fe = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=True,
        n_stores=N_STORES, store_emb_dim=STORE_EMB_DIM,
        store_ids_tr=s_tr, store_ids_val=s_te,
        store_idx_tr=s_tr_idx, store_idx_val=s_te_idx,
        hidden=cfg["mdp_e2e_hidden"],
        tag=f"ND-Habit-FE s={seed}",
    )
    nd_hab_fe_m = sw_fe["best_model"]

    # ── CF first-stage + CF models ─────────────────────────────────────────────
    v_hat_tr, cf_rsq = cf_first_stage(
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

    # KW bundles
    KW_SHARED = dict(
        aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
        ff=feat_shared, theta=th_sh,
        # Window model + fallback history (used for single-point elasticity eval)
        wirl=wirl_m,
        wirl_log_p_hist=wirl_lp_mean,
        wirl_log_q_hist=wirl_lq_mean,
        wirl_window=WIRL_W,
        nirl=nirl_m,
        nirl_fe=nirl_fe_m,
        nirl_cf=nirl_cf_m,
        mdp_cf=mdp_cf_m,
        mdp=nd_hab_m,
        mdp_fe=nd_hab_fe_m,
    )

    def _xb(nm):
        if "CF" in nm and "habit" in nm:
            return xb_mn          # precomputed EWMA for HabitND-CF
        return xb_mn if "habit" in nm else None

    def _kw(nm):
        if nm == "Linear Demand (GoodSpec)":
            return {"ff": feat_good_specific, "theta": th_gs}
        if nm == "Linear Demand (Orth)":
            return {"ff": feat_orth, "theta": th_orth}
        return {}

    # ── Own-price elasticities ─────────────────────────────────────────────────
    own_eps = {}
    for nm in ALL_MODEL_NAMES:
        try:
            eps = own_elasticity(nm, p_mn, y_mn, cfg,
                                 xb_prev0=_xb(nm), q_prev0=qp_mn,
                                 s_te_mode_idx=s_te_mode_idx,
                                 **{**KW_SHARED, **_kw(nm)})
            own_eps[nm] = eps
        except Exception as exc:
            print(f"    Warning: own-elast {nm} failed — {exc}")
            own_eps[nm] = np.full(G, np.nan)

    # ── Cross-price elasticity matrices ───────────────────────────────────────
    cross_eps = {}
    for nm in ["Logit-IV", "LA-AIDS",
               "Neural Demand (static)", "Neural Demand (habit)",
               "Neural Demand (CF)", "Neural Demand (habit, CF)"]:
        try:
            mat = elasticity_matrix(nm, p_mn, y_mn, cfg,
                                    xb_prev0=_xb(nm), q_prev0=qp_mn,
                                    **{**KW_SHARED, **_kw(nm)})
            cross_eps[nm] = mat
        except Exception as exc:
            print(f"    Warning: cross-elast {nm} failed — {exc}")
            cross_eps[nm] = np.full((G, G), np.nan)

    return dict(
        own_eps=own_eps,
        cross_eps=cross_eps,
        p_mn=p_mn,
        delta_hat=delta_hat,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray([x for x in arr if not np.isnan(float(x))], float)
    return float(np.std(a, ddof=1) / np.sqrt(max(len(a), 1))) if len(a) > 1 else 0.0


def aggregate(all_results: list) -> dict:
    n = len(all_results)
    model_names = list(all_results[0]["own_eps"].keys())

    own_agg = {}
    for nm in model_names:
        stack = np.stack([r["own_eps"][nm] for r in all_results], 0)
        own_agg[nm] = {"mean": stack.mean(0), "se": stack.std(0, ddof=max(1, n-1)) / np.sqrt(n)}

    cross_agg = {}
    for nm in all_results[0]["cross_eps"].keys():
        stack = np.stack([r["cross_eps"][nm] for r in all_results], 0)
        cross_agg[nm] = {"mean": stack.mean(0), "se": stack.std(0, ddof=max(1, n-1)) / np.sqrt(n)}

    return dict(own_agg=own_agg, cross_agg=cross_agg, n_runs=n)


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(agg: dict, cfg: dict) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    n_runs  = agg["n_runs"]
    se_note = f"  ({n_runs} runs, ±1 SE)" if n_runs > 1 else ""

    model_names = list(agg["own_agg"].keys())
    colors = [STYLE.get(nm, {}).get("color", "#888") for nm in model_names]

    fig, axes = plt.subplots(1, G, figsize=(5 * G, 5), sharey=False)
    for g, (ax, good) in enumerate(zip(axes, GOODS)):
        vals = [agg["own_agg"][nm]["mean"][g] for nm in model_names]
        errs = [agg["own_agg"][nm]["se"][g]   for nm in model_names]
        x = np.arange(len(model_names))
        ax.bar(x, vals, yerr=errs if n_runs > 1 else None, capsize=5,
               color=colors, edgecolor="k", alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Own-price elasticity", fontsize=11)
        # ax.set_title(f"{good} ($g={g}$)", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    # fig.suptitle(f"Own-Price Elasticities — Dominick's{se_note}",
                #  fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_dir}/fig_dom_elasticities.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig_dom_elasticities.pdf/png")

    # Cross-elasticity heatmaps
    for nm, d in agg["cross_agg"].items():
        figH, axH = plt.subplots(figsize=(6, 5))
        mat = d["mean"]
        im  = axH.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2)
        plt.colorbar(im, ax=axH, label="Elasticity")
        axH.set_xticks(range(G)); axH.set_yticks(range(G))
        axH.set_xticklabels(GOODS); axH.set_yticklabels(GOODS)
        for i in range(G):
            for j in range(G):
                axH.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=10)
        safe_nm = nm.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        axH.set_title(f"Cross-Price Elasticities — {nm}", fontsize=11, fontweight="bold")
        figH.tight_layout()
        for ext in ("pdf", "png"):
            figH.savefig(f"{fig_dir}/fig_dom_cross_elast_{safe_nm}.{ext}",
                         dpi=150, bbox_inches="tight")
        plt.close(figH)
        print(f"  Saved: {fig_dir}/fig_dom_cross_elast_{safe_nm}.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def make_tables(agg: dict, cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    n_runs  = agg["n_runs"]
    os.makedirs(out_dir, exist_ok=True)

    # Own-elasticity table
    rows = []
    for nm, d in agg["own_agg"].items():
        rows.append({
            "Model":    nm,
            "eps_0":    d["mean"][0], "eps_0_se": d["se"][0],
            "eps_1":    d["mean"][1], "eps_1_se": d["se"][1],
            "eps_2":    d["mean"][2], "eps_2_se": d["se"][2],
        })
    make_elasticity_table(
        rows, out_dir=out_dir,
        label="table_dom_own_elasticities",
        caption=(r"Own-Price Elasticities at Test-Mean Prices --- "
                 r"Dominick's Analgesics."),
        n_runs=n_runs,
    )

    # Cross-elasticity tables
    for nm, d in agg["cross_agg"].items():
        safe_nm = nm.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        mat_df  = pd.DataFrame(d["mean"].round(4), index=GOODS, columns=GOODS)
        mat_df.to_csv(f"{out_dir}/table_dom_cross_elast_{safe_nm}.csv")
        print(f"  Saved: {out_dir}/table_dom_cross_elast_{safe_nm}.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(splits: dict, cfg: dict) -> tuple:
    N_RUNS = cfg.get("N_RUNS", 1)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Dominick's Exp 02: Elasticities")
    print("=" * 68)

    pre = cfg.get("_precomputed_dom_exp02")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed Exp02 runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = run_once(seed, splits, cfg)
            all_results.append(r)
            nd  = r["own_eps"].get("Neural Demand (static)", np.full(G, np.nan))
            nh  = r["own_eps"].get("Neural Demand (habit)",  np.full(G, np.nan))
            print(f"    Done in {time.time()-t0:.0f}s  "
                  f"NDS eps={np.round(nd, 3)}  ND+Habit eps={np.round(nh, 3)}")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    make_tables(agg, cfg)
    return all_results, agg
