"""
experiments/neural_demand/simulation/exp02_habit_advantage.py
==============================================================
Section 3 — Habit-State Augmentation Advantage.

Compares ALL paper models on data generated from HabitFormationConsumer.
The key claims are:

1. Habit-augmented model reduces out-of-sample RMSE by ~90% relative to both
   LA-AIDS and the static neural model.
2. The static neural model provides no improvement over LA-AIDS on habit data,
   confirming the gain is structural rather than architectural.
3. The profile KL loss is flat across the identified set for δ.
4. CF-corrected models further reduce bias when prices are endogenous.

Models compared
---------------
  LA-AIDS, QUAIDS, Series Estm., Linear Demand (Shared), Linear Demand (GoodSpec),
  Linear Demand (Orth), Var. Mixture, Neural Demand (static),
  Neural Demand (habit), Neural Demand (CF), Neural Demand (habit, CF)

Produces
--------
results/neural_demand/simulations/
  table_habit_advantage.csv / .tex
  fig_habit_advantage.{pdf,png}      (RMSE and KL bar charts)
  fig_habit_demand_curves.{pdf,png}  (demand curves good-0 vs p1)
  fig_habit_profile_kl.{pdf,png}     (profile KL across δ grid)
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
    AIDSBench,
    BLPBench,
    QUAIDS,
    SeriesDemand,
    HabitFormationConsumer,
    StaticND,
    HabitND,
        compute_xbar_e2e,
    cf_first_stage,
    features_shared,
    features_good_specific,
    features_orthogonalised,
    run_linear_irl,
    train_neural_irl,
    train_mdp_e2e,
)
from experiments.simulation.utils import (
    P_GRID,
    AVG_Y,
    BAND,
    STYLE,
    predict_shares,
    get_metrics,
    kl_div,
    compute_compensating_variation,
    fit_neural_demand_delta_grid,
)

warnings.filterwarnings("ignore")


# All ordered model specs (display-name → predict_shares spec key)
MODEL_SPECS = [
    ("LA-AIDS",                      "aids"),
    ("Logit-IV",                     "blp"),
    ("QUAIDS",                       "quaids"),
    ("Series Estm.",                 "series"),
    ("Linear Demand (Shared)",                 "linear-demand-shared"),
    ("Linear Demand (GoodSpec)",               "linear-demand-goodspecific"),
    ("Linear Demand (Orth)",                   "linear-demand-orth"),
    ("Neural Demand (static)",       "neural-demand-static"),
    ("Neural Demand (habit)",        "neural-demand-habit"),
    ("Neural Demand (CF)",           "neural-demand-static-cf"),
    ("Neural Demand (habit, CF)",    "neural-demand-habit-cf"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Single-seed runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(seed: int, cfg: dict, verbose: bool = False) -> dict:
    """Execute the full habit-advantage comparison pipeline for one seed."""
    N          = cfg["N_OBS"]
    DEVICE     = cfg["DEVICE"]
    DELTA_GRID = cfg["DELTA_GRID"]
    EPOCHS     = cfg["EPOCHS"]
    TRUE_DELTA = cfg.get("TRUE_DELTA", 0.7)
    HIDDEN     = cfg.get("hidden_dim", 128)
    DELTA_HAB  = cfg.get("DELTA_HAB",
                         float(np.asarray(DELTA_GRID)[len(DELTA_GRID) // 2]))
    CACHE_DIR  = cfg.get("model_cache_dir")
    FORCE_RETRAIN = cfg.get("force_retrain", False)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Simulate training data (cost-shifter Z → slightly endogenous prices) ──
    Z      = np.random.uniform(1, 5, (N, 3))
    p_pre  = np.clip(Z + np.random.normal(0, 0.1, (N, 3)), 1e-3, None)
    income = np.random.uniform(1200, 2000, N)

    # post-shock: good 1 becomes 20% more expensive
    p_post = p_pre.copy()
    p_post[:, 1] *= 1.2

    habit_consumer = HabitFormationConsumer()
    w_hab,       xbar_tr = habit_consumer.solve_demand(p_pre,  income, return_xbar=True)
    w_hab_shock, _       = habit_consumer.solve_demand(p_post, income, return_xbar=True)

    # Log-quantity sequences for EWMA habit stock computation
    q_tr  = w_hab * income[:, None] / np.maximum(p_pre, 1e-8)
    lq_tr = np.log(np.maximum(q_tr, 1e-6))

    # ── Validation split ───────────────────────────────────────────────────────
    rng_val = np.random.default_rng(seed + 7777)
    N_val   = max(N // 5, 100)
    p_val   = np.clip(rng_val.uniform(1, 5, (N_val, 3))
                      + rng_val.normal(0, 0.1, (N_val, 3)), 1e-3, None)
    y_val   = rng_val.uniform(1200, 2000, N_val)
    hc_val  = HabitFormationConsumer()
    w_val, _ = hc_val.solve_demand(p_val, y_val, return_xbar=True)
    q_val   = w_val * y_val[:, None] / np.maximum(p_val, 1e-8)
    lq_val  = np.log(np.maximum(q_val, 1e-6))

    # ── Static benchmarks ──────────────────────────────────────────────────────
    aids_hab   = AIDSBench();    aids_hab.fit(p_pre, w_hab, income)
    # BLP: all 3 goods are inside goods; add a fixed 1% outside option
    # (no genuine outside good in simulation — mirrors main_multiple_runs.py).
    _blp_out = 0.01
    mw_hab   = np.column_stack([w_hab * (1 - _blp_out),
                                 np.full(len(w_hab), _blp_out)])
    blp_hab  = BLPBench().fit(p_pre, mw_hab, Z, income)
    quaids_hab = QUAIDS();       quaids_hab.fit(p_pre, w_hab, income)
    series_hab = SeriesDemand(); series_hab.fit(p_pre, w_hab, income)

    # ── Linear IRL (3 feature variants) ───────────────────────────────────────
    F_sh = features_shared(p_pre, income)
    F_gs = features_good_specific(p_pre, income)
    F_or = features_orthogonalised(p_pre, income)
    theta_sh = run_linear_irl(F_sh, w_hab, lr=0.05, epochs=3000, l2=1e-4)
    theta_gs = run_linear_irl(F_gs, w_hab, lr=0.05, epochs=3000, l2=1e-4)
    theta_or = run_linear_irl(F_or, w_hab, lr=0.05, epochs=3000, l2=1e-4)

    # ── Neural Demand (static) ─────────────────────────────────────────────────
    nds_static = StaticND(n_goods=3, hidden_dim=HIDDEN)
    nds_static, hist_static = train_neural_irl(
        nds_static, p_pre, income, w_hab,
        epochs=EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
        device=DEVICE, verbose=verbose,
        tag=f"nds-static-habadv-s{seed}",
        cache_dir=CACHE_DIR,
        force_retrain=FORCE_RETRAIN)

    # ── CF first-stage residuals ───────────────────────────────────────────────
    v_hat_tr, _ = cf_first_stage(np.log(np.maximum(p_pre, 1e-8)), Z)

    # ── Neural Demand (CF) ─────────────────────────────────────────────────────
    nds_cf_m = StaticND(n_goods=3, hidden_dim=HIDDEN, n_cf=3)
    hist_cf = []
    try:
        nds_cf_m, hist_cf = train_neural_irl(
            nds_cf_m, p_pre, income, w_hab,
            epochs=EPOCHS, lr=5e-4, batch_size=256,
            lam_mono=0.2, lam_slut=0.05, slut_start_frac=0.3,
            v_hat_data=v_hat_tr,
            device=DEVICE, verbose=False,
            tag=f"nds-cf-habadv-s{seed}",
            cache_dir=CACHE_DIR,
            force_retrain=FORCE_RETRAIN)
    except Exception as exc:
        if verbose:
            print(f"    [ND-static-CF fit failed: {exc}]")
        nds_cf_m = None

    # ── Neural Demand (habit) — profile-criterion δ sweep ─────────────────────
    sweep = fit_neural_demand_delta_grid(
        p_pre, income, w_hab, lq_tr,
        p_val, y_val, w_val, lq_val,
        delta_grid=DELTA_GRID,
        epochs=EPOCHS, lr=5e-4, batch_size=256,
        lam_mono=0.3, lam_slut=0.1,
        hidden_dim=HIDDEN,
        device=DEVICE,
        tag=f"nd-habit-s{seed}",
        cache_dir=CACHE_DIR,
        force_retrain=FORCE_RETRAIN,
    )
    nds_hab   = sweep["best_model"]
    delta_hat = sweep["delta_hat"]
    kl_grid   = sweep["kl_grid"]
    se_grid   = sweep["se_grid"]
    id_set    = sweep["id_set"]

    # Compute xbar on test (post-shock) prices using selected δ
    q_shock  = w_hab_shock * income[:, None] / np.maximum(p_post, 1e-8)
    lq_shock = np.log(np.maximum(q_shock, 1e-6))
    with torch.no_grad():
        d_t      = torch.tensor(float(delta_hat), dtype=torch.float32, device=DEVICE)
        lq_sh_t  = torch.tensor(lq_shock, dtype=torch.float32, device=DEVICE)
        xbar_hat = compute_xbar_e2e(d_t, lq_sh_t, store_ids=None).cpu().numpy()

    # ── Neural Demand (habit, CF) — fixed DELTA_HAB ───────────────────────────
    xb_ewma = np.zeros_like(w_hab)
    xb_ewma[0] = np.log(np.maximum(w_hab[0], 1e-8))
    for t in range(1, N):
        xb_ewma[t] = (DELTA_HAB * xb_ewma[t - 1]
                      + (1.0 - DELTA_HAB) * np.log(np.maximum(w_hab[t - 1], 1e-8)))
    q_prev_tr = np.roll(lq_tr, 1, axis=0); q_prev_tr[0] = lq_tr[0]

    nds_hab_cf_m = HabitND(n_goods=3, hidden_dim=HIDDEN,
                                delta_init=DELTA_HAB, n_cf=3)
    hist_hab_cf = []
    try:
        nds_hab_cf_m, hist_hab_cf = train_neural_irl(
            nds_hab_cf_m, p_pre, income, w_hab,
            epochs=EPOCHS, lr=5e-4, batch_size=256,
            lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
            xb_prev_data=np.exp(xb_ewma),
            q_prev_data=np.exp(q_prev_tr),
            v_hat_data=v_hat_tr,
            device=DEVICE, verbose=False,
            tag=f"nds-hab-cf-habadv-s{seed}",
            cache_dir=CACHE_DIR,
            force_retrain=FORCE_RETRAIN)
    except Exception as exc:
        if verbose:
            print(f"    [ND+Habit-CF fit failed: {exc}]")
        nds_hab_cf_m = None

    # Post-shock EWMA for habit-CF evaluation (v_hat=0 → structural)
    xb_shock_ewma = np.zeros_like(w_hab_shock)
    xb_shock_ewma[0] = np.log(np.maximum(w_hab_shock[0], 1e-8))
    for t in range(1, len(w_hab_shock)):
        xb_shock_ewma[t] = (DELTA_HAB * xb_shock_ewma[t - 1]
                            + (1.0 - DELTA_HAB)
                            * np.log(np.maximum(w_hab_shock[t - 1], 1e-8)))
    q_prev_shock = np.roll(lq_shock, 1, axis=0); q_prev_shock[0] = lq_shock[0]

    # ── Shared KW bundle ───────────────────────────────────────────────────────
    KW = dict(
        aids=aids_hab, blp=blp_hab, quaids=quaids_hab, series=series_hab,
        theta_sh=theta_sh, theta_gs=theta_gs, theta_or=theta_or,
        nds=nds_static,
        nds_hab=nds_hab,   xbar_hab=xbar_hat,
        nds_cf=nds_cf_m,
        nds_hab_cf=nds_hab_cf_m,
        device=DEVICE,
    )

    # ── Metrics on post-shock test data ────────────────────────────────────────
    rmse       = {}
    kl_scores  = {}
    for nm, sp in MODEL_SPECS:
        xb_kw = {}
        if sp == "neural-demand-habit":
            xb_kw = dict(xbar_hab=xbar_hat)
        elif sp == "neural-demand-habit-cf":
            xb_kw = dict(xbar_hab=xb_shock_ewma, q_prev_hab=q_prev_shock)
        try:
            wp = predict_shares(sp, p_post, income, **{**KW, **xb_kw})
            rmse[nm]      = float(np.sqrt(np.mean((wp - w_hab_shock) ** 2)))
            kl_scores[nm] = kl_div(wp, w_hab_shock)
        except Exception:
            rmse[nm]      = np.nan
            kl_scores[nm] = np.nan

    base_rmse  = rmse.get("LA-AIDS", np.nan)
    reductions = {k: 100.0 * (base_rmse - v) / max(abs(base_rmse), 1e-12)
                  for k, v in rmse.items()}

    # ── Compensating Variation ─────────────────────────────────────────────────
    avg_p   = p_pre.mean(0)
    p0_welf = avg_p / np.array([1.0, 1.2, 1.0])
    p1_welf = avg_p
    xb_mean = xbar_tr.mean(0)

    cv = {}
    for nm, sp in MODEL_SPECS:
        xb_kw = {}
        if sp == "neural-demand-habit":
            xb_kw = dict(xbar_pt=xb_mean)
        try:
            cv[nm] = compute_compensating_variation(sp, p0_welf, p1_welf, AVG_Y,
                                                    **{**KW, **xb_kw})
        except Exception:
            cv[nm] = np.nan

    # ── Demand curves (varying p1, good-0 share) ───────────────────────────────
    test_p  = np.tile(p_pre.mean(0), (len(P_GRID), 1))
    test_p[:, 1] = P_GRID
    fixed_y = np.full(len(P_GRID), AVG_Y)

    # xbar for habit model on demand curve sweep (from HabitND δ̂)
    q_sweep  = (np.tile(w_hab.mean(0), (len(P_GRID), 1))
                * AVG_Y / np.maximum(test_p, 1e-8))
    lq_sweep = np.log(np.maximum(q_sweep, 1e-6))
    with torch.no_grad():
        d_t2    = torch.tensor(float(delta_hat), dtype=torch.float32, device=DEVICE)
        lq_sw_t = torch.tensor(lq_sweep, dtype=torch.float32, device=DEVICE)
        xb_sw   = compute_xbar_e2e(d_t2, lq_sw_t, store_ids=None).cpu().numpy()

    # xbar for habit-CF sweep (EWMA at DELTA_HAB)
    xb_sw_ewma = np.zeros_like(xb_sw)
    for t in range(len(P_GRID)):
        if t == 0:
            xb_sw_ewma[t] = np.log(np.maximum(w_hab.mean(0), 1e-8))
        else:
            xb_sw_ewma[t] = (DELTA_HAB * xb_sw_ewma[t - 1]
                             + (1.0 - DELTA_HAB)
                             * np.log(np.maximum(w_hab.mean(0), 1e-8)))
    q_prev_sw = np.roll(lq_sweep, 1, axis=0); q_prev_sw[0] = lq_sweep[0]

    curves = {"Truth": habit_consumer.solve_demand(test_p, fixed_y)[:, 0]}
    for nm, sp in MODEL_SPECS:
        xb_kw = {}
        if sp == "neural-demand-habit":
            xb_kw = dict(xbar_hab=xb_sw)
        elif sp == "neural-demand-habit-cf":
            xb_kw = dict(xbar_hab=xb_sw_ewma, q_prev_hab=q_prev_sw)
        try:
            curves[nm] = predict_shares(sp, test_p, fixed_y, **{**KW, **xb_kw})[:, 0]
        except Exception:
            pass

    # ── All-goods demand curves for paper figures (shape: (80, 3)) ────────────
    curves_all = {"Truth": habit_consumer.solve_demand(test_p, fixed_y)}
    for nm, sp in MODEL_SPECS:
        _xb_kw = {}
        if sp == "neural-demand-habit":
            _xb_kw = dict(xbar_hab=xb_sw)
        elif sp == "neural-demand-habit-cf":
            _xb_kw = dict(xbar_hab=xb_sw_ewma, q_prev_hab=q_prev_sw)
        try:
            curves_all[nm] = predict_shares(
                sp, test_p, fixed_y, **{**KW, **_xb_kw})  # (80, 3)
        except Exception:
            pass

    return dict(
        rmse=rmse, kl_scores=kl_scores, reductions=reductions,
        cv=cv,
        curves=curves,
        curves_all=curves_all,
        delta_hat=delta_hat,
        delta_true=TRUE_DELTA,
        delta_in_id_set=bool(id_set[0] <= TRUE_DELTA <= id_set[1]),
        id_set=id_set,
        kl_grid=kl_grid,
        se_grid=se_grid,
        delta_grid=np.asarray(DELTA_GRID, dtype=float),
        hist_static=hist_static,
        hist_habit=sweep["best_hist"],
        hist_cf=hist_cf,
        hist_hab_cf=hist_hab_cf,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)], float)
    if len(a) < 2:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


def _agg_training_hist(hist_list):
    """Average training histories (list of list-of-dicts) across seeds.

    Returns dict with keys ``epochs``, ``kl_mean``, ``kl_se``.
    """
    valid = [h for h in hist_list if h and len(h) > 0]
    if not valid:
        return {"epochs": np.array([]), "kl_mean": np.array([]), "kl_se": np.array([])}
    min_len = min(len(h) for h in valid)
    epochs  = np.array([e["epoch"] for e in valid[0][:min_len]])
    kl_mat  = np.array([[e["kl"] for e in h[:min_len]] for h in valid])
    se_vals = (kl_mat.std(axis=0, ddof=1) / np.sqrt(len(valid))
               if len(valid) > 1 else np.zeros(min_len))
    return {
        "epochs":  epochs,
        "kl_mean": kl_mat.mean(axis=0),
        "kl_se":   se_vals,
    }


def aggregate(all_results: list) -> dict:
    n          = len(all_results)
    r0         = all_results[0]

    def _agg(key):
        return {nm: {"mean": float(np.nanmean([r[key].get(nm, np.nan) for r in all_results])),
                     "se":   _se([r[key].get(nm, np.nan) for r in all_results])}
                for nm in all_results[0][key].keys()}

    rmse_agg = _agg("rmse")
    kl_agg   = _agg("kl_scores")
    red_agg  = _agg("reductions")
    cv_agg   = _agg("cv")

    all_curve_keys = list(dict.fromkeys(
        k for r in all_results for k in r["curves"].keys()
    ))
    curves_mean = {k: np.mean(
        [r["curves"][k] for r in all_results if k in r["curves"]], 0)
        for k in all_curve_keys}
    curves_se   = {k: (np.array([r["curves"][k] for r in all_results
                                 if k in r["curves"]]).std(0, ddof=min(1, n - 1))
                       / np.sqrt(n)) for k in all_curve_keys}

    # ── All-goods demand curves ────────────────────────────────────────────────
    _all_keys = list(dict.fromkeys(
        k for r in all_results for k in r.get("curves_all", {}).keys()
    ))
    curves_all_mean = {}
    curves_all_se   = {}
    for k in _all_keys:
        arrs = [r["curves_all"][k] for r in all_results
                if k in r.get("curves_all", {})]
        if arrs:
            curves_all_mean[k] = np.mean(arrs, axis=0)
            curves_all_se[k]   = (np.array(arrs).std(axis=0, ddof=max(1, n - 1))
                                  / np.sqrt(n))

    kl_stack = np.stack([r["kl_grid"] for r in all_results], 0)
    se_stack = np.stack([r["se_grid"] for r in all_results], 0)

    delta_hats = [r["delta_hat"] for r in all_results]
    in_id_frac = float(np.mean([r["delta_in_id_set"] for r in all_results]))
    id_set_lo  = float(np.nanmean([r["id_set"][0] for r in all_results]))
    id_set_hi  = float(np.nanmean([r["id_set"][1] for r in all_results]))

    # ── Aggregate training convergence histories ───────────────────────────────
    train_conv = {
        "Neural Demand (static)":    _agg_training_hist(
            [r.get("hist_static", []) for r in all_results]),
        "Neural Demand (habit)":     _agg_training_hist(
            [r.get("hist_habit",  []) for r in all_results]),
        "Neural Demand (CF)":        _agg_training_hist(
            [r.get("hist_cf",     []) for r in all_results]),
        "Neural Demand (habit, CF)": _agg_training_hist(
            [r.get("hist_hab_cf", []) for r in all_results]),
    }

    return dict(
        rmse_agg=rmse_agg, kl_agg=kl_agg, red_agg=red_agg, cv_agg=cv_agg,
        curves_mean=curves_mean, curves_se=curves_se,
        curves_all_mean=curves_all_mean, curves_all_se=curves_all_se,
        kl_prof_mean=kl_stack.mean(0),
        kl_prof_se=kl_stack.std(0, ddof=max(1, n - 1)) / np.sqrt(n),
        delta_grid=r0["delta_grid"],
        delta_hat_mean=float(np.nanmean(delta_hats)),
        delta_hat_se=_se(delta_hats),
        true_delta=r0["delta_true"],
        in_id_frac=in_id_frac,
        id_set_lo=id_set_lo,
        id_set_hi=id_set_hi,
        train_conv=train_conv,
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
    se_note = f"  ({N_RUNS} runs, ±1 SE)" if N_RUNS > 1 else ""

    model_names = list(agg["rmse_agg"].keys())
    rmse_means  = [agg["rmse_agg"][nm]["mean"] for nm in model_names]
    rmse_ses    = [agg["rmse_agg"][nm]["se"]   for nm in model_names]
    kl_means    = [agg["kl_agg"][nm]["mean"]   for nm in model_names]
    kl_ses      = [agg["kl_agg"][nm]["se"]     for nm in model_names]

    bar_colors = [STYLE.get(nm, {}).get("color", "#888") for nm in model_names]

    # ── Bar chart: RMSE and KL ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(model_names))
    for ax, means, ses, ylabel, title in zip(
        axes,
        [rmse_means, kl_means],
        [rmse_ses,   kl_ses],
        ["Out-of-Sample RMSE", "KL Divergence KL(truth‖pred)"],
        ["Post-Shock RMSE — Habit DGP", "Post-Shock KL — Habit DGP"],
    ):
        valid_means = [m if not np.isnan(m) else 0.0 for m in means]
        valid_ses   = [s if not np.isnan(s) else 0.0 for s in ses]
        ax.bar(x, valid_means,
               yerr=valid_ses if N_RUNS > 1 else None,
               capsize=5, color=bar_colors, edgecolor="k", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        # ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # fig.suptitle(f"Neural Demand — Habit Advantage{se_note}",
                #  fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{fig_dir}/fig_habit_advantage.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_dir}/fig_habit_advantage.pdf/png")

    # ── Demand curves ──────────────────────────────────────────────────────────
    figC, axC = plt.subplots(figsize=(12, 6))
    for lbl, sty in [("Truth", dict(color="k", ls="-", lw=3.0))]  \
            + [(nm, STYLE.get(nm, dict(color="#888", ls="--", lw=1.5)))
               for nm, _ in MODEL_SPECS]:
        if lbl not in agg["curves_mean"]:
            continue
        mu  = agg["curves_mean"][lbl]
        sig = agg["curves_se"].get(lbl, np.zeros_like(mu))
        axC.plot(P_GRID, mu, label=lbl, **sty)
        if N_RUNS > 1:
            axC.fill_between(P_GRID, mu - sig, mu + sig,
                             color=sty["color"], alpha=BAND)
    axC.set_xlabel(r"Good-1 price $p_1$", fontsize=13)
    axC.set_ylabel(r"Good-0 budget share $w_0$", fontsize=13)
    # axC.set_title(f"Demand Curves — Habit DGP{se_note}",
    #               fontsize=12, fontweight="bold")
    axC.legend(fontsize=9, ncol=2, loc="best")
    axC.grid(True, alpha=0.3)
    figC.tight_layout()
    for ext in ("pdf", "png"):
        figC.savefig(f"{fig_dir}/fig_habit_demand_curves.{ext}", dpi=150,
                     bbox_inches="tight")
    plt.close(figC)
    print(f"  Saved: {fig_dir}/fig_habit_demand_curves.pdf/png")

    # ── Profile KL (δ vs validation KL, averaged over seeds) ─────────────────
    figD, axD = plt.subplots(figsize=(9, 5))
    mu_kl = agg["kl_prof_mean"]
    se_kl = agg["kl_prof_se"]
    dg    = agg["delta_grid"]
    TEAL  = "#00897B"
    axD.plot(dg, mu_kl, color=TEAL, lw=2.5, marker="o", ms=5, label="Val. KL")
    if N_RUNS > 1:
        axD.fill_between(dg, mu_kl - se_kl, mu_kl + se_kl,
                         color=TEAL, alpha=BAND)
    axD.axvline(agg["true_delta"], color="#E53935", ls="--", lw=2,
                label=f"True δ={agg['true_delta']:.1f}")
    axD.axvline(agg["delta_hat_mean"], color="#FB8C00", ls=":", lw=2,
                label=f"δ̂={agg['delta_hat_mean']:.2f} (mean)")
    axD.set_xlabel("Habit-decay parameter δ", fontsize=13)
    axD.set_ylabel("Validation KL divergence", fontsize=13)
    # axD.set_title(f"Profile KL — δ Identification{se_note}",
    #               fontsize=12, fontweight="bold")
    axD.legend(fontsize=11)
    axD.grid(True, alpha=0.3)
    figD.tight_layout()
    for ext in ("pdf", "png"):
        figD.savefig(f"{fig_dir}/fig_habit_profile_kl.{ext}", dpi=150,
                     bbox_inches="tight")
    plt.close(figD)
    print(f"  Saved: {fig_dir}/fig_habit_profile_kl.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def make_tables(agg: dict, cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    N_RUNS  = agg["n_runs"]
    os.makedirs(out_dir, exist_ok=True)

    # ── CSV ────────────────────────────────────────────────────────────────────
    rows = []
    for nm in agg["rmse_agg"]:
        rows.append({
            "Model":          nm,
            "RMSE_mean":      agg["rmse_agg"][nm]["mean"],
            "RMSE_se":        agg["rmse_agg"][nm]["se"],
            "KL_mean":        agg["kl_agg"].get(nm, {}).get("mean", np.nan),
            "KL_se":          agg["kl_agg"].get(nm, {}).get("se",   np.nan),
            "RMSE_reduction": agg["red_agg"].get(nm, {}).get("mean", np.nan),
            "CV_mean":        agg["cv_agg"].get(nm, {}).get("mean", np.nan),
            "CV_se":          agg["cv_agg"].get(nm, {}).get("se",   np.nan),
        })
    pd.DataFrame(rows).round(6).to_csv(
        f"{out_dir}/table_habit_advantage.csv", index=False)
    print(f"  Saved: {out_dir}/table_habit_advantage.csv")

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    def _c(m, s, d=5):
        if np.isnan(m):
            return "{---}"
        if N_RUNS > 1:
            return f"${m:.{d}f} \\pm {s:.{d}f}$"
        return f"${m:.{d}f}$"

    lines = [
        r"% ============================================================",
        r"% Neural Demand — Habit Advantage Table (auto-generated)",
        f"% N_RUNS = {N_RUNS}",
        r"% ============================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Habit-State Augmentation Advantage --- Simulation"
        rf"  ({N_RUNS} runs; mean $\pm$ SE)}}",
        r"  \label{tab:sim_habit_advantage}",
        r"  \begin{threeparttable}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{RMSE} & \textbf{KL Div.}"
        r"    & \textbf{RMSE Red.\ (\%)} & \textbf{CV} \\",
        r"    \midrule",
    ]
    for row in rows:
        nm  = row["Model"]
        red = (f"{row['RMSE_reduction']:.1f}\\%"
               if not np.isnan(row["RMSE_reduction"]) else "baseline")
        lines.append(
            f"    {nm} & {_c(row['RMSE_mean'], row['RMSE_se'])} "
            f"& {_c(row['KL_mean'], row['KL_se'])} "
            f"& {red} "
            f"& {_c(row['CV_mean'], row['CV_se'], d=4)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}\small",
        rf"    \item True DGP: \texttt{{HabitFormationConsumer}} with"
        rf"    $\delta_{{true}}={agg['true_delta']:.1f}$."
        rf"    $\hat{{\delta}} = {agg['delta_hat_mean']:.3f} \pm"
        rf"    {agg['delta_hat_se']:.3f}$ (mean $\pm$ SE over {N_RUNS} runs)."
        rf"    True $\delta$ in identified set: {100*agg['in_id_frac']:.0f}\% of seeds.",
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}", "",
        r"% ============================================================",
        r"% Figure environments",
        r"% ============================================================", "",
    ]
    for fn, cap, lbl in [
        ("fig_habit_advantage",
         "Habit-State Augmentation Advantage.  RMSE (left) and KL divergence (right) "
         "on post-shock test data.  Shaded bands = ±1 SE across runs.",
         "fig:sim_habit_advantage"),
        ("fig_habit_demand_curves",
         "Demand Curves under Habit DGP.  Good-0 budget share as a function of "
         "good-1 price.  Habit-augmented model best tracks the true curve.",
         "fig:sim_habit_curves"),
        ("fig_habit_profile_kl",
         "Profile KL criterion for $\\delta$ identification.  "
         "The flat region spans the identified set.",
         "fig:sim_habit_profile_kl"),
    ]:
        lines += [
            r"\begin{figure}[htbp]", r"  \centering",
            f"  \\includegraphics[width=0.85\\linewidth]{{{cfg['fig_dir']}/{fn}.pdf}}",
            f"  \\caption{{{cap}}}", f"  \\label{{{lbl}}}",
            r"\end{figure}", "",
        ]

    tex_path = f"{out_dir}/table_habit_advantage.tex"
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved: {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> tuple:
    """Run all seeds, aggregate, produce outputs.

    Parameters
    ----------
    cfg : dict with keys:
        N_RUNS      int   number of independent seeds
        N_OBS       int   observations per seed
        EPOCHS      int   training epochs
        DELTA_GRID  array grid of δ values for profile sweep
        DELTA_HAB   float fixed δ for habit-CF model (default: grid midpoint)
        DEVICE      str   torch device
        TRUE_DELTA  float true δ used in HabitFormationConsumer
        hidden_dim  int   MLP hidden size (default 128)
        out_dir     str   path for tables
        fig_dir     str   path for figures
    """
    N_RUNS = cfg["N_RUNS"]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand  —  Simulation Exp 02: Habit Advantage")
    print("=" * 68)

    pre = cfg.get("_precomputed_sim_exp02")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed Exp02 runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS}  seed={seed}")
            r = run_one_seed(seed, cfg, verbose=(ri == N_RUNS - 1))
            all_results.append(r)
            s_rmse = r["rmse"].get("Neural Demand (static)", np.nan)
            h_rmse = r["rmse"].get("Neural Demand (habit)",  np.nan)
            print(f"    Done in {time.time()-t0:.0f}s  "
                  f"δ̂={r['delta_hat']:.2f} (true {r['delta_true']:.1f})  "
                  f"static_RMSE={s_rmse:.5f}  habit_RMSE={h_rmse:.5f}")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    make_tables(agg, cfg)

    # Summary
    print("\n── Habit Advantage Summary ─────────────────────────────────────────")
    for nm in agg["rmse_agg"]:
        m = agg["rmse_agg"][nm]["mean"]
        s = agg["rmse_agg"][nm]["se"]
        r = agg["red_agg"].get(nm, {}).get("mean", 0.0)
        print(f"  {nm:40s}  RMSE={m:.5f}±{s:.5f}  red={r:+.1f}%")
    print(f"\n  δ̂ = {agg['delta_hat_mean']:.3f} ± {agg['delta_hat_se']:.3f}  "
          f"(true {agg['true_delta']:.1f})")
    print(f"  True δ in identified set: {100*agg['in_id_frac']:.0f}% of seeds\n")

    return all_results, agg
