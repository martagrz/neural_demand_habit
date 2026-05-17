"""
experiments/neural_demand/simulation/exp01_dgp_recovery.py
===========================================================
Section 3 — Simulation Study: Recovery of demand, elasticities, and
compensating variation across five data-generating processes:
  CES, Quasilinear, Leontief, Stone–Geary, Habit Formation.

Models compared
---------------
  LA-AIDS, QUAIDS, Series Estm., Linear Demand (Shared), Linear Demand (GoodSpec),
  Linear Demand (Orth), Neural Demand (static),
  Neural Demand (habit), Neural Demand (CF), Neural Demand (habit, CF)

For each DGP and each model we compute:
  • Out-of-sample RMSE and MAE (budget shares)
  • Own-price quantity elasticities at mean prices
  • Compensating variation from a 20% price-2 shock

Produces
--------
results/neural_demand/simulations/
  table_dgp_recovery.csv
  table_dgp_recovery.tex      (LaTeX booktabs table)
  table_elasticities.csv
  table_elasticities.tex
  table_welfare.csv
  table_welfare.tex
  fig_demand_curves_CES.{pdf,png}   (demand curves under CES DGP)
  fig_dgp_robustness.{pdf,png}      (bar chart of RMSE across DGPs)
  fig_observed_vs_predicted.{pdf,png}
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import warnings

from src.models.simulation import (
    AIDSBench, BLPBench, QUAIDS, SeriesDemand,
    CESConsumer, QuasilinearConsumer, LeontiefConsumer,
    StoneGearyConsumer, HabitFormationConsumer,
    StaticND, HabitND,
        compute_xbar_e2e,
    cf_first_stage, cf_first_stage_full,
    features_shared, features_good_specific, features_orthogonalised,
    run_linear_irl,
    train_neural_irl,
    train_mdp_e2e,
)
from experiments.simulation.utils import (
    P_GRID, AVG_Y, BAND, STYLE,
    predict_shares, get_metrics, kl_div,
    compute_own_elasticities, compute_cross_elasticity_matrix,
    compute_compensating_variation,
    fit_neural_demand_delta_grid,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL SPECS  (display name, predict_shares spec key)
# ─────────────────────────────────────────────────────────────────────────────

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
#  SINGLE-SEED FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(seed: int, cfg: dict, verbose: bool = False) -> dict:
    """Full pipeline for one RNG seed."""
    N          = cfg["N_OBS"]
    DEVICE     = cfg["DEVICE"]
    DELTA_GRID = cfg["DELTA_GRID"]
    EPOCHS     = cfg["EPOCHS"]
    HIDDEN     = cfg.get("hidden_dim", 128)
    CACHE_DIR  = cfg.get("model_cache_dir")
    FORCE_RETRAIN = cfg.get("force_retrain", False)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Shared price / income draws ────────────────────────────────────────────
    # Instrument (cost-shifter) generates slightly endogenous prices
    Z      = np.random.uniform(1, 5, (N, 3))
    p_pre  = np.clip(Z + np.random.normal(0, 0.1, (N, 3)), 1e-3, None)
    income = np.random.uniform(1200, 2000, N)

    p_post  = p_pre.copy(); p_post[:, 1] *= 1.2      # 20% shock on good 1

    avg_p   = p_post.mean(0)
    p0_welf = avg_p / np.array([1.0, 1.2, 1.0])   # pre-shock
    p1_welf = avg_p                                  # post-shock

    # Validation set (for habit-model δ sweep)
    rng_val = np.random.default_rng(seed + 9999)
    N_val   = max(N // 5, 100)
    p_val   = np.clip(rng_val.uniform(1, 5, (N_val, 3))
                      + rng_val.normal(0, 0.1, (N_val, 3)), 1e-3, None)
    y_val   = rng_val.uniform(1200, 2000, N_val)

    # ── DGP catalogue ─────────────────────────────────────────────────────────
    DGPs = {
        "CES":         CESConsumer(),
        "Quasilinear": QuasilinearConsumer(),
        "Leontief":    LeontiefConsumer(),
        "Stone–Geary": StoneGearyConsumer(),
        "Habit":       HabitFormationConsumer(),
        "Endogenous CES": CESConsumer(),  # Placeholder, handled specially in loop
    }

    acc_rows          = []
    elast_rows        = []
    welf_rows         = []
    welf_by_shock_rows = []
    curves_ces        = {}
    scatter_data      = {}
    curves_ces_full   = {}
    cross_elast_ces   = {}
    train_hists       = {}     # {dgp_name: {model_name: hist}}

    dgp_filter = cfg.get("dgp_filter", None)
    for dgp_name, consumer in DGPs.items():
        if dgp_filter is not None and dgp_name != dgp_filter:
            continue
        if verbose:
            print(f"  DGP: {dgp_name}")

        # ── Data Generation ──────────────────────────────────────────────────
        if dgp_name == "Endogenous CES":
            # ── Endogenous CES with log-linear (multiplicative) price endogeneity ──
            #
            # Price equation:  log(p_j) = log(Z_j) + xi_j + nu_j
            #                  i.e.  p_j = Z_j * exp(xi_j + nu_j)
            #
            # This is the natural Hausman IV structure: log-prices in other stores
            # are the instruments, so the CF first stage log(p) ~ log(Z) is exactly
            # specified.  Logit-IV uses raw p (not log p) in its second stage, so it
            # faces a double misspecification:
            #   (a) Linear first stage in levels is wrong for a log-linear price DGP.
            #   (b) Linear second stage log(s/s0) ~ p̂ is wrong because CES shares
            #       depend on log(p), not p; this error compounds at high sigma.
            #
            # rho_ces = 0.7 (sigma ≈ 3.33) gives a strongly curved denominator so
            # that the logit's linear-in-prices second stage is visibly misspecified
            # over the wider price support used here (Z ~ U[0.5, 8]).

            Z_endo = np.random.uniform(0.5, 8.0, (N, 3))   # wider than shared Z
            xi = np.random.normal(0, 0.3, (N, 3))           # endogenous taste shock
            nu = np.random.normal(0, 0.05, (N, 3))          # idiosyncratic supply shock

            p_endo = np.clip(Z_endo * np.exp(xi + nu), 0.05, None)

            base_alpha = np.array([0.4, 0.4, 0.2])
            rho_ces    = 0.7                         # sigma ≈ 3.33: more nonlinear
            sigma_ces  = 1.0 / (1.0 - rho_ces)

            alpha_obs = base_alpha[None, :] * np.exp(xi)
            num       = alpha_obs ** sigma_ces * p_endo ** (1.0 - sigma_ces)
            w_endo    = num / num.sum(axis=1, keepdims=True)

            curr_p_pre   = p_endo
            curr_w_train = w_endo
            curr_Z       = Z_endo

            # Structural demand at the shared p_post prices (xi = 0, base_alpha).
            # All models are evaluated at the same p_post for comparability.
            num_post    = base_alpha[None, :] ** sigma_ces * p_post ** (1.0 - sigma_ces)
            curr_w_post = num_post / num_post.sum(axis=1, keepdims=True)
            
        else:
            # Standard DGPs use shared exogenous prices
            try:
                curr_p_pre   = p_pre
                curr_w_train = consumer.solve_demand(p_pre, income)
                curr_w_post  = consumer.solve_demand(p_post, income)
                curr_Z       = Z
            except Exception as e:
                print(f"    [skip DGP={dgp_name}: {e}]"); continue

        val_consumer = type(consumer)()
        try:
            w_val = val_consumer.solve_demand(p_val, y_val)
        except Exception:
            w_val = np.full((N_val, 3), 1/3)
        # Override validation shares for Endogenous CES: use the local sigma_ces
        # (rho=0.7) and base_alpha, not the default CESConsumer's parameters.
        if dgp_name == "Endogenous CES":
            _num_v = base_alpha[None, :] ** sigma_ces * p_val ** (1.0 - sigma_ces)
            w_val  = _num_v / _num_v.sum(axis=1, keepdims=True)

        # ── Static benchmarks ─────────────────────────────────────────────────
        aids_m   = AIDSBench(); aids_m.fit(curr_p_pre, curr_w_train, income)
        # BLP: all 3 goods are inside goods; add a fixed 1% outside option
        # (no genuine outside good in simulation — mirrors main_multiple_runs.py).
        _blp_out = 0.01
        mw_train = np.column_stack([curr_w_train * (1 - _blp_out),
                                    np.full(N, _blp_out)])
        blp_m    = BLPBench().fit(curr_p_pre, mw_train, curr_Z, income)
        quaids_m = QUAIDS();    quaids_m.fit(curr_p_pre, curr_w_train, income)
        series_m = SeriesDemand(); series_m.fit(curr_p_pre, curr_w_train, income)

        # ── Linear IRL (3 feature variants) ──────────────────────────────────
        F_sh = features_shared(curr_p_pre, income)
        F_gs = features_good_specific(curr_p_pre, income)
        F_or = features_orthogonalised(curr_p_pre, income)
        theta_sh = run_linear_irl(F_sh, curr_w_train, lr=0.05, epochs=3000, l2=1e-4)
        theta_gs = run_linear_irl(F_gs, curr_w_train, lr=0.05, epochs=3000, l2=1e-4)
        theta_or = run_linear_irl(F_or, curr_w_train, lr=0.05, epochs=3000, l2=1e-4)

        # Adjust LR for Endogenous CES (harder landscape)
        LR = 5e-3 if dgp_name == "Endogenous CES" else 5e-4

        # ── Neural Demand (static) ────────────────────────────────────────────
        nds_m = StaticND(n_goods=3, hidden_dim=HIDDEN)
        nds_m, hist_nds_static = train_neural_irl(
            nds_m, curr_p_pre, income, curr_w_train,
            epochs=EPOCHS, lr=LR, batch_size=256,
            lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
            device=DEVICE, verbose=verbose,
            tag=f"nds-static-{dgp_name}-s{seed}",
            cache_dir=CACHE_DIR,
            force_retrain=FORCE_RETRAIN)

        # ── Neural Demand (CF) ────────────────────────────────────────────────
        # Log-log first stage: log(p_j) ~ log(Z_j).  This is the natural spec
        # for Hausman IVs and is correctly specified for the multiplicative
        # Endogenous CES DGP where log(p) = log(Z) + xi + nu.  For all other
        # DGPs prices are exogenous so the residuals are negligible regardless.
        v_hat_tr, _ = cf_first_stage(np.log(np.maximum(curr_p_pre, 1e-8)),
                                     np.log(np.maximum(curr_Z, 1e-8)))
        nds_cf_m = StaticND(n_goods=3, hidden_dim=HIDDEN, n_cf=3)
        nds_cf_m, hist_nds_cf = train_neural_irl(
            nds_cf_m, curr_p_pre, income, curr_w_train,
            epochs=EPOCHS, lr=LR, batch_size=256,
            lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
            v_hat_data=v_hat_tr,
            device=DEVICE, verbose=False,
            tag=f"nds-cf-{dgp_name}-s{seed}",
            cache_dir=CACHE_DIR,
            force_retrain=FORCE_RETRAIN)

        # ── Neural Demand (habit) — δ sweep ──────────────────────────────────
        q_tr  = curr_w_train * income[:, None] / np.maximum(curr_p_pre, 1e-8)
        lq_tr = np.log(np.maximum(q_tr, 1e-6))
        q_v   = w_val * y_val[:, None] / np.maximum(p_val, 1e-8)
        lq_v  = np.log(np.maximum(q_v, 1e-6))

        sweep = None
        try:
            sweep = fit_neural_demand_delta_grid(
                curr_p_pre, income, curr_w_train, lq_tr,
                p_val, y_val, w_val, lq_v,
                delta_grid=DELTA_GRID,
                epochs=EPOCHS, lr=LR, batch_size=256,
                lam_mono=0.3, lam_slut=0.1,
                hidden_dim=HIDDEN, device=DEVICE,
                tag=f"nd-hab-{dgp_name}-s{seed}",
                cache_dir=CACHE_DIR,
                force_retrain=FORCE_RETRAIN,
            )
            nds_hab_m = sweep["best_model"]
            delta_hat = sweep["delta_hat"]
        except Exception as exc:
            if verbose: print(f"    [ND+Habit fit failed: {exc}]")
            nds_hab_m = None; delta_hat = float(DELTA_GRID[len(DELTA_GRID)//2])

        # Compute xbar for test (post-shock) set AND mean training xbar for
        # demand-curve evaluation (habit model should be evaluated at a
        # representative in-distribution habit stock, not at zeros).
        q_post = curr_w_post * income[:, None] / np.maximum(p_post, 1e-8)
        lq_post = np.log(np.maximum(q_post, 1e-6))
        if nds_hab_m is not None:
            with torch.no_grad():
                d_t  = torch.tensor(float(delta_hat), dtype=torch.float32, device=DEVICE)
                lq_t = torch.tensor(lq_post, dtype=torch.float32, device=DEVICE)
                xb_post = compute_xbar_e2e(d_t, lq_t, store_ids=None).cpu().numpy()
                # Mean training habit stock — used to fix xbar when sweeping prices
                lq_t_tr = torch.tensor(lq_tr, dtype=torch.float32, device=DEVICE)
                xb_tr_arr = compute_xbar_e2e(d_t, lq_t_tr, store_ids=None).cpu().numpy()
            xb_tr_mean = xb_tr_arr.mean(0)   # (G,) — representative training-time log_xbar
        else:
            xb_post    = np.zeros_like(curr_w_post)
            xb_tr_mean = np.zeros(curr_w_train.shape[1])

        # ── Neural Demand (habit, CF) ─────────────────────────────────────────
        # Use fixed delta (midpoint of grid) for the CF habit model.
        # xb_ewma uses log-shares (not log-quantities) for exp01.  In exp01 the
        # DGPs are non-habit (CES, E-CES, Leontief …) so the xbar is irrelevant
        # noise regardless.  Log-shares are bounded (≈ −2 to 0) and — unlike
        # log-quantities — don't amplify the E-CES price endogeneity by sigma≈3.
        # Using log-quantities blows up E-CES training.  Exp02 (Habit DGP) uses
        # log-quantities throughout since habit is real there.
        DELTA_HAB = cfg.get("DELTA_HAB", float(DELTA_GRID[len(DELTA_GRID)//2]))
        log_w_tr = np.log(np.maximum(curr_w_train, 1e-8))
        xb_ewma = np.empty_like(log_w_tr)
        xb_ewma[0] = log_w_tr[0]
        for t in range(1, N):
            xb_ewma[t] = (DELTA_HAB * xb_ewma[t-1]
                          + (1.0 - DELTA_HAB) * log_w_tr[t-1])
        q_prev_tr = np.roll(lq_tr, 1, axis=0); q_prev_tr[0] = lq_tr[0]

        nds_hab_cf_m = HabitND(n_goods=3, hidden_dim=HIDDEN,
                                    delta_init=DELTA_HAB, n_cf=3)
        hist_nds_hab_cf = []
        try:
            nds_hab_cf_m, hist_nds_hab_cf = train_neural_irl(
                nds_hab_cf_m, curr_p_pre, income, curr_w_train,
                epochs=EPOCHS, lr=LR, batch_size=256,
                lam_mono=0.3, lam_slut=0.1, slut_start_frac=0.25,
                xb_prev_data=np.exp(xb_ewma),
                q_prev_data=np.exp(q_prev_tr),
                v_hat_data=v_hat_tr,
                device=DEVICE, verbose=False,
                tag=f"nds-hab-cf-{dgp_name}-s{seed}",
                cache_dir=CACHE_DIR,
                force_retrain=FORCE_RETRAIN)
        except Exception as exc:
            if verbose: print(f"    [ND+Habit-CF fit failed: {exc}]")
            nds_hab_cf_m = None

        # Post-shock EWMA for habit-CF evaluation: log-shares, consistent with training.
        log_w_post = np.log(np.maximum(curr_w_post, 1e-8))
        xb_post_ewma = np.empty_like(log_w_post)
        xb_post_ewma[0] = log_w_post[0]
        for t in range(1, len(log_w_post)):
            xb_post_ewma[t] = (DELTA_HAB * xb_post_ewma[t-1]
                               + (1.0 - DELTA_HAB) * log_w_post[t-1])
        q_prev_post = np.roll(lq_post, 1, axis=0); q_prev_post[0] = lq_post[0]

        # ── Shared KW bundle ─────────────────────────────────────────────────
        KW = dict(
            aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
            theta_sh=theta_sh, theta_gs=theta_gs, theta_or=theta_or,
            nds=nds_m, nds_hab=nds_hab_m,
            nds_cf=nds_cf_m, nds_hab_cf=nds_hab_cf_m,
            consumer=consumer, device=DEVICE,
        )
        # habit-stock kwargs — passed explicitly per call site to avoid
        # duplicate-keyword errors when overriding with evaluation-time zeros
        _HAB_KW      = dict(xbar_hab=xb_post, q_prev_hab=q_prev_post)
        _HAB_EWMA_KW = dict(xbar_hab=xb_post_ewma, q_prev_hab=q_prev_post)

        # ── Accuracy ─────────────────────────────────────────────────────────
        for nm, sp in MODEL_SPECS:
            xb_kw = {}
            if sp == "neural-demand-habit":
                xb_kw = _HAB_KW
            elif sp == "neural-demand-habit-cf":
                xb_kw = _HAB_EWMA_KW
            try:
                m = get_metrics(sp, p_post, income, curr_w_post, **xb_kw, **KW)
                acc_rows.append({"DGP": dgp_name, "Model": nm,
                                 "RMSE": m["RMSE"], "MAE": m["MAE"]})
            except Exception:
                acc_rows.append({"DGP": dgp_name, "Model": nm,
                                 "RMSE": np.nan, "MAE": np.nan})
        acc_rows.append({"DGP": dgp_name, "Model": "Ground Truth",
                         "RMSE": 0.0, "MAE": 0.0})

        # ── Elasticities ─────────────────────────────────────────────────────
        p_eval = p_post.mean(0)
        for nm, sp in MODEL_SPECS:
            xb_kw = {}
            if sp == "neural-demand-habit":
                xb_kw = dict(xbar_pt=xb_post.mean(0))
            elif sp == "neural-demand-habit-cf":
                xb_kw = dict(xbar_pt=xb_post_ewma.mean(0))
            try:
                eps = compute_own_elasticities(sp, p_eval, AVG_Y, **xb_kw, **KW)
                elast_rows.append({"DGP": dgp_name, "Model": nm,
                                   "eps0": eps[0], "eps1": eps[1], "eps2": eps[2]})
            except Exception:
                elast_rows.append({"DGP": dgp_name, "Model": nm,
                                   "eps0": np.nan, "eps1": np.nan, "eps2": np.nan})
        try:
            eps_true = compute_own_elasticities("truth", p_eval, AVG_Y, **KW)
        except Exception:
            eps_true = [np.nan] * 3
        elast_rows.append({"DGP": dgp_name, "Model": "Ground Truth",
                           "eps0": eps_true[0], "eps1": eps_true[1],
                           "eps2": eps_true[2]})

        # ── Welfare ──────────────────────────────────────────────────────────
        try:
            cv_true = compute_compensating_variation(
                "truth", p0_welf, p1_welf, AVG_Y, **KW)
        except Exception:
            cv_true = np.nan

        for nm, sp in MODEL_SPECS:
            xb_kw = {}
            if sp == "neural-demand-habit":
                xb_kw = dict(xbar_pt=xb_post.mean(0))
            elif sp == "neural-demand-habit-cf":
                xb_kw = dict(xbar_pt=xb_post_ewma.mean(0))
            try:
                cv = compute_compensating_variation(sp, p0_welf, p1_welf, AVG_Y,
                                                    **xb_kw, **KW)
                welf_rows.append({"DGP": dgp_name, "Model": nm,
                                  "CV": cv,
                                  "CV_err_pct":
                                      100 * abs(cv - cv_true) / max(abs(cv_true), 1e-9)
                                      if not np.isnan(cv_true) else np.nan})
            except Exception:
                welf_rows.append({"DGP": dgp_name, "Model": nm,
                                  "CV": np.nan, "CV_err_pct": np.nan})
        welf_rows.append({"DGP": dgp_name, "Model": "Ground Truth",
                          "CV": cv_true, "CV_err_pct": 0.0})

        # ── Welfare by shock (three 10% single-good shocks, CES only) ────────
        if dgp_name == "CES":
            _SHOCK_PCT = 0.10
            _avg_p_base = p_pre.mean(0)
            _cv_true_by_g = {}
            for _g in range(3):
                _sf = np.ones(3); _sf[_g] = 1.0 + _SHOCK_PCT
                try:
                    _cv_true_by_g[_g] = compute_compensating_variation(
                        "truth", _avg_p_base, _avg_p_base * _sf, AVG_Y, **KW)
                except Exception:
                    _cv_true_by_g[_g] = np.nan

            for nm, sp in MODEL_SPECS:
                _xb_kw = {}
                if sp == "neural-demand-habit":
                    _xb_kw = dict(xbar_pt=xb_post.mean(0))
                elif sp == "neural-demand-habit-cf":
                    _xb_kw = dict(xbar_pt=xb_post_ewma.mean(0))
                _row = {"DGP": dgp_name, "Model": nm}
                for _g in range(3):
                    _sf = np.ones(3); _sf[_g] = 1.0 + _SHOCK_PCT
                    try:
                        _cv_g = compute_compensating_variation(
                            sp, _avg_p_base, _avg_p_base * _sf, AVG_Y,
                            **_xb_kw, **KW)
                        _ct_g = _cv_true_by_g.get(_g, np.nan)
                        _row[f"CV_g{_g}"] = _cv_g
                        _row[f"CV_err_g{_g}"] = (
                            100 * abs(_cv_g - _ct_g) / max(abs(_ct_g), 1e-9)
                            if not np.isnan(_ct_g) else np.nan)
                    except Exception:
                        _row[f"CV_g{_g}"] = np.nan
                        _row[f"CV_err_g{_g}"] = np.nan
                welf_by_shock_rows.append(_row)

            _gt_row = {"DGP": dgp_name, "Model": "Ground Truth"}
            for _g in range(3):
                _gt_row[f"CV_g{_g}"] = _cv_true_by_g.get(_g, np.nan)
                _gt_row[f"CV_err_g{_g}"] = 0.0
            welf_by_shock_rows.append(_gt_row)

        # ── Demand curves (CES only) ──────────────────────────────────────────
        if dgp_name == "CES":
            test_p  = np.tile(p_pre.mean(0), (80, 1)); test_p[:, 1] = P_GRID
            fixed_y = np.full(80, AVG_Y)
            curves_ces["Truth"]                        = consumer.solve_demand(test_p, fixed_y)[:, 0]
            curves_ces["LA-AIDS"]                      = predict_shares("aids",        test_p, fixed_y, **KW)[:, 0]
            curves_ces["Logit-IV"]                     = predict_shares("blp",         test_p, fixed_y, **KW)[:, 0]
            curves_ces["QUAIDS"]                       = predict_shares("quaids",      test_p, fixed_y, **KW)[:, 0]
            curves_ces["Series Estm."]                 = predict_shares("series",      test_p, fixed_y, **KW)[:, 0]
            curves_ces["Linear Demand (Shared)"]                 = predict_shares("linear-demand-shared", test_p, fixed_y, **KW)[:, 0]
            curves_ces["Linear Demand (GoodSpec)"]               = predict_shares("linear-demand-goodspecific", test_p, fixed_y, **KW)[:, 0]
            curves_ces["Neural Demand (static)"]       = predict_shares("neural-demand-static",   test_p, fixed_y, **KW)[:, 0]
            if nds_hab_m is not None:
                # Hold the habit stock at its mean training value so we isolate
                # the pure price effect.  Using zeros gives log_xbar = 0, i.e.
                # xbar = exp(0) = 1, which is outside the training distribution
                # and causes the model to produce near-flat demand curves.
                xb_cs = np.tile(xb_tr_mean, (80, 1))   # (80, G)
                curves_ces["Neural Demand (habit)"]    = predict_shares(
                    "neural-demand-habit", test_p, fixed_y,
                    **{**KW, "xbar_hab": xb_cs, "q_prev_hab": xb_cs})[:, 0]
            curves_ces["Neural Demand (CF)"]           = predict_shares("neural-demand-static-cf", test_p, fixed_y, **KW)[:, 0]

            wp_nds = predict_shares("neural-demand-static", p_post, income, **KW)
            scatter_data = {"observed": curr_w_post[:, 0], "predicted_nds": wp_nds[:, 0],
                            "predicted_aids": aids_m.predict(p_post, income)[:, 0]}

            # ── All-goods demand curves for paper figures (shape: (80, 3)) ────────
            # Mean training xbar keeps habit-model evaluation in-distribution
            _xb_cs = np.tile(xb_tr_mean, (80, 1))   # (80, G)
            _curve_specs = [
                ("Truth",                    "truth",       {}),
                ("LA-AIDS",                  "aids",        {}),
                ("Logit-IV",                 "blp",         {}),
                ("QUAIDS",                   "quaids",      {}),
                ("Series Estm.",             "series",      {}),
                ("Linear Demand (Orth)",               "linear-demand-orth",   {}),
                ("Neural Demand (static)",   "neural-demand-static",   {}),
            ]
            if nds_hab_m is not None:
                _curve_specs.append(("Neural Demand (habit)", "neural-demand-habit", {"xbar_hab": _xb_cs}))
            curves_ces_full = {}
            for _lbl, _sp, _xb_kw in _curve_specs:
                try:
                    curves_ces_full[_lbl] = predict_shares(
                        _sp, test_p, fixed_y, **{**KW, **_xb_kw})  # (80, 3)
                except Exception:
                    pass

            # ── Cross-price elasticity matrices for paper figures (3 × 3) ────────
            _p_eval = p_pre.mean(0)
            _elast_specs = [
                ("Ground Truth",             "truth",     {}),
                ("LA-AIDS",                  "aids",      {}),
                ("Logit-IV",                 "blp",       {}),
                ("QUAIDS",                   "quaids",    {}),
                ("Neural Demand (static)",   "neural-demand-static", {}),
            ]
            cross_elast_ces = {}
            for _nm, _sp, _xb_kw in _elast_specs:
                try:
                    cross_elast_ces[_nm] = compute_cross_elasticity_matrix(
                        _sp, _p_eval, AVG_Y, **{**KW, **_xb_kw})  # (3, 3)
                except Exception:
                    cross_elast_ces[_nm] = np.full((3, 3), np.nan)

        # ── Training convergence histories for this DGP ───────────────────────
        _h_hab = sweep["best_hist"] if sweep is not None else []
        train_hists[dgp_name] = {
            "Neural Demand (static)":    hist_nds_static,
            "Neural Demand (CF)":        hist_nds_cf,
            "Neural Demand (habit)":     _h_hab,
            "Neural Demand (habit, CF)": hist_nds_hab_cf,
        }

    return dict(
        acc=pd.DataFrame(acc_rows),
        elast=pd.DataFrame(elast_rows),
        welf=pd.DataFrame(welf_rows),
        welf_by_shock=pd.DataFrame(welf_by_shock_rows),
        curves_ces=curves_ces,
        scatter_data=scatter_data,
        curves_ces_full=curves_ces_full,
        cross_elast_ces=cross_elast_ces,
        train_hists=train_hists,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray(arr, float)
    return np.nanstd(a, ddof=1) / np.sqrt(np.sum(~np.isnan(a)))


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
    n = len(all_results)

    def _agg_df(df_list, val_cols):
        idx = [c for c in df_list[0].columns if c not in val_cols]
        combined = pd.concat(df_list, ignore_index=True)
        mean_df  = combined.groupby(idx, sort=False)[val_cols].mean().reset_index()
        se_df    = combined.groupby(idx, sort=False)[val_cols].sem().reset_index()
        se_df.columns = idx + [f"{c}_se" for c in val_cols]
        return mean_df.merge(se_df, on=idx)

    acc_agg   = _agg_df([r["acc"]   for r in all_results], ["RMSE", "MAE"])
    elast_agg = _agg_df([r["elast"] for r in all_results], ["eps0", "eps1", "eps2"])
    welf_agg  = _agg_df([r["welf"]  for r in all_results], ["CV", "CV_err_pct"])

    _wbs_list = [r["welf_by_shock"] for r in all_results
                 if not r["welf_by_shock"].empty]
    if _wbs_list:
        _wbs_val_cols = ["CV_g0", "CV_g1", "CV_g2",
                         "CV_err_g0", "CV_err_g1", "CV_err_g2"]
        welf_by_shock_agg = _agg_df(_wbs_list, _wbs_val_cols)
    else:
        welf_by_shock_agg = pd.DataFrame()

    curve_keys  = list(all_results[0]["curves_ces"].keys())
    curves_mean = {k: np.mean([r["curves_ces"][k] for r in all_results
                               if k in r["curves_ces"]], 0)
                   for k in curve_keys}
    curves_se   = {k: np.array([r["curves_ces"][k] for r in all_results
                                if k in r["curves_ces"]]).std(0, ddof=1)
                      / np.sqrt(n) for k in curve_keys}

    # ── All-goods demand curves (shape per key: (80, 3)) ──────────────────────
    _full_keys = list(dict.fromkeys(
        k for r in all_results for k in r.get("curves_ces_full", {}).keys()
    ))
    curves_ces_full_mean = {}
    curves_ces_full_se   = {}
    for k in _full_keys:
        arrs = [r["curves_ces_full"][k] for r in all_results
                if k in r.get("curves_ces_full", {})]
        if arrs:
            curves_ces_full_mean[k] = np.mean(arrs, axis=0)
            curves_ces_full_se[k]   = (np.array(arrs).std(axis=0, ddof=1)
                                       / np.sqrt(len(arrs)))

    # ── Cross-elasticity matrices ──────────────────────────────────────────────
    _elast_keys = list(dict.fromkeys(
        k for r in all_results for k in r.get("cross_elast_ces", {}).keys()
    ))
    cross_elast_mean = {}
    cross_elast_se   = {}
    for k in _elast_keys:
        arrs = [r["cross_elast_ces"][k] for r in all_results
                if k in r.get("cross_elast_ces", {})]
        if arrs:
            cross_elast_mean[k] = np.mean(arrs, axis=0)
            cross_elast_se[k]   = (np.array(arrs).std(axis=0, ddof=1)
                                   / np.sqrt(len(arrs)))

    # ── Aggregate training convergence histories per DGP + model ──────────────
    _dgp_names = list(dict.fromkeys(
        dgp for r in all_results for dgp in r.get("train_hists", {}).keys()
    ))
    _nd_models = [
        "Neural Demand (static)",
        "Neural Demand (CF)",
        "Neural Demand (habit)",
        "Neural Demand (habit, CF)",
    ]
    train_conv = {}   # {dgp_name: {model_name: aggregated_hist}}
    for dgp in _dgp_names:
        train_conv[dgp] = {}
        for mn in _nd_models:
            hist_list = [r["train_hists"][dgp].get(mn, [])
                         for r in all_results
                         if dgp in r.get("train_hists", {})]
            train_conv[dgp][mn] = _agg_training_hist(hist_list)

    return dict(
        acc_agg=acc_agg, elast_agg=elast_agg, welf_agg=welf_agg,
        welf_by_shock_agg=welf_by_shock_agg,
        curves_mean=curves_mean, curves_se=curves_se,
        curves_ces_full_mean=curves_ces_full_mean,
        curves_ces_full_se=curves_ces_full_se,
        cross_elast_mean=cross_elast_mean,
        cross_elast_se=cross_elast_se,
        train_conv=train_conv,
        last=all_results[-1], n_runs=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(agg: dict, cfg: dict) -> None:
    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    N_RUNS = agg["n_runs"]

    curves_mean = agg["curves_mean"]
    curves_se   = agg["curves_se"]

    # ── Fig A: Demand curves under CES DGP ────────────────────────────────────
    figA, axA = plt.subplots(figsize=(11, 6))
    for lbl, sty in STYLE.items():
        if lbl not in curves_mean:
            continue
        mu    = curves_mean[lbl]
        sigma = curves_se.get(lbl, np.zeros_like(mu))
        axA.plot(P_GRID, mu, label=lbl, **sty)
        axA.fill_between(P_GRID, mu - sigma, mu + sigma,
                         color=sty["color"], alpha=BAND)
    axA.set_xlabel("Good-1 price $p_1$", fontsize=13)
    axA.set_ylabel("Good-0 budget share $w_0$", fontsize=13)
    axA.legend(fontsize=9, ncol=2, loc="best")
    axA.grid(True, alpha=0.3)
    se_note = f"  (shaded = ±1 SE, {N_RUNS} runs)" if N_RUNS > 1 else ""
    # axA.set_title(f"Demand Curves — CES DGP{se_note}", fontsize=12, fontweight="bold")
    figA.tight_layout()
    for ext in ("pdf", "png"):
        figA.savefig(f"{fig_dir}/fig_demand_curves_CES.{ext}", dpi=150, bbox_inches="tight")
    plt.close(figA)
    print(f"  Saved: {fig_dir}/fig_demand_curves_CES")

    # ── Fig B: DGP robustness bar chart (exclude Habit DGP) ──────────────────
    acc_agg = agg["acc_agg"]
    dgp_names   = [d for d in acc_agg["DGP"].unique() if d != "Habit"]
    model_names = [m for m in acc_agg["Model"].unique() if m != "Ground Truth"]
    colors_bar  = [STYLE.get(m, {}).get("color", "#888") for m in model_names]

    figB, axB = plt.subplots(figsize=(14, 5))
    x     = np.arange(len(dgp_names))
    width = 0.8 / max(len(model_names), 1)
    for ci, (mn, col) in enumerate(zip(model_names, colors_bar)):
        means = [float(acc_agg.loc[(acc_agg["DGP"] == d) & (acc_agg["Model"] == mn), "RMSE"].values[0])
                 if len(acc_agg.loc[(acc_agg["DGP"] == d) & (acc_agg["Model"] == mn)]) else np.nan
                 for d in dgp_names]
        ses   = [float(acc_agg.loc[(acc_agg["DGP"] == d) & (acc_agg["Model"] == mn), "RMSE_se"].values[0])
                 if len(acc_agg.loc[(acc_agg["DGP"] == d) & (acc_agg["Model"] == mn)]) else np.nan
                 for d in dgp_names]
        axB.bar(x + ci * width, means, width, yerr=ses if N_RUNS > 1 else None,
                capsize=3, color=col, label=mn, alpha=0.85)
    axB.set_xticks(x + (len(model_names) - 1) / 2 * width)
    axB.set_xticklabels(dgp_names, fontsize=10)
    axB.set_ylabel("Post-shock RMSE", fontsize=12)
    # axB.set_title("DGP Robustness — Post-shock RMSE", fontsize=12, fontweight="bold")
    axB.legend(fontsize=8, ncol=2); axB.grid(axis="y", alpha=0.3)
    figB.tight_layout()
    for ext in ("pdf", "png"):
        figB.savefig(f"{fig_dir}/fig_dgp_robustness.{ext}", dpi=150, bbox_inches="tight")
    plt.close(figB)
    print(f"  Saved: {fig_dir}/fig_dgp_robustness")

    # ── Fig C: Observed vs predicted (CES, last run) ──────────────────────────
    scat = agg["last"]["scatter_data"]
    if scat:
        fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 5))
        for ax, pred_key, model_nm, col in [
            (axes_s[0], "predicted_nds",  "Neural Demand (static)", "#1E88E5"),
            (axes_s[1], "predicted_aids", "LA-AIDS",                "#E53935"),
        ]:
            if pred_key not in scat: continue
            obs = scat["observed"]; pred = scat[pred_key]
            valid = ~(np.isnan(obs) | np.isnan(pred))
            ax.scatter(obs[valid], pred[valid], alpha=0.3, s=8, color=col, rasterized=True)
            lo = 0.0; hi = max(float(obs[valid].max()), float(pred[valid].max())) * 1.05
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            rmse = float(np.sqrt(np.mean((obs[valid] - pred[valid]) ** 2)))
            # ax.set_title(f"{model_nm}\nRMSE={rmse:.5f}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Observed $w_0$", fontsize=10)
            ax.set_ylabel("Predicted $w_0$", fontsize=10)
            ax.grid(True, alpha=0.3)
        # fig_s.suptitle("Observed vs Predicted — CES DGP (last run)", fontsize=11, fontweight="bold")
        fig_s.tight_layout()
        for ext in ("pdf", "png"):
            fig_s.savefig(f"{fig_dir}/fig_observed_vs_predicted.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        print(f"  Saved: {fig_dir}/fig_observed_vs_predicted")


# ─────────────────────────────────────────────────────────────────────────────
#  LATEX TABLES
# ─────────────────────────────────────────────────────────────────────────────

def _cell(m, s, d=5):
    if np.isnan(m): return "{---}"
    return f"${m:.{d}f} \\pm {s:.{d}f}$"


def make_tables(agg: dict, cfg: dict) -> None:
    out_dir = cfg["out_dir"]
    fig_dir = cfg["fig_dir"]
    N_RUNS  = agg["n_runs"]
    os.makedirs(out_dir, exist_ok=True)

    acc_agg   = agg["acc_agg"]
    elast_agg = agg["elast_agg"]
    welf_agg  = agg["welf_agg"]

    # ── CSV ──────────────────────────────────────────────────────────────────
    acc_agg.round(6).to_csv(f"{out_dir}/table_dgp_recovery.csv", index=False)
    elast_agg.round(4).to_csv(f"{out_dir}/table_elasticities.csv", index=False)
    welf_agg.round(6).to_csv(f"{out_dir}/table_welfare.csv", index=False)
    print(f"  Saved CSVs to {out_dir}/")

    dgp_names   = list(acc_agg["DGP"].unique())
    model_names = list(acc_agg["Model"].unique())

    tex_lines = [
        r"% ============================================================",
        r"% Neural Demand Paper — Simulation Tables (auto-generated)",
        f"% N_RUNS = {N_RUNS}",
        r"% ============================================================", "",
    ]

    # ── RMSE table ───────────────────────────────────────────────────────────
    col_spec = "l" + "c" * len(dgp_names)
    tex_lines += [
        r"\begin{table}[htbp]", r"  \centering",
        rf"  \caption{{Post-Shock RMSE by DGP and Model (mean $\pm$ SE, {N_RUNS} runs)}}",
        r"  \label{tab:sim_dgp_recovery}",
        r"  \begin{threeparttable}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        r"    \textbf{Model} & "
        + " & ".join(f"\\textbf{{{d}}}" for d in dgp_names) + r" \\",
        r"    \midrule",
    ]
    for mn in model_names:
        cells = []
        for dg in dgp_names:
            sub = acc_agg.loc[(acc_agg["DGP"] == dg) & (acc_agg["Model"] == mn)]
            cells.append(_cell(float(sub["RMSE"].iloc[0]), float(sub["RMSE_se"].iloc[0]))
                         if len(sub) else "{---}")
        tex_lines.append(f"    {mn} & " + " & ".join(cells) + r" \\")
    tex_lines += [
        r"    \bottomrule", r"  \end{tabular}",
        r"  \begin{tablenotes}\small",
        rf"    \item Post-shock: 20\% increase in price of good 1.",
        r"  \end{tablenotes}", r"  \end{threeparttable}", r"\end{table}", "",
    ]

    # ── Elasticities table (CES DGP) ─────────────────────────────────────────
    tex_lines += [
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{Own-Price Quantity Elasticities — Simulation (CES DGP)}",
        r"  \label{tab:sim_elasticities}",
        r"  \begin{threeparttable}",
        r"  \begin{tabular}{lccc}", r"    \toprule",
        r"    \textbf{Model} & $\hat{\varepsilon}_{00}$ & $\hat{\varepsilon}_{11}$ & $\hat{\varepsilon}_{22}$ \\",
        r"    \midrule",
    ]
    ces_elast = elast_agg[elast_agg["DGP"] == "CES"]
    for mn in ces_elast["Model"].unique():
        sub = ces_elast[ces_elast["Model"] == mn]
        if not len(sub): continue
        row_cells = [_cell(float(sub[ec].iloc[0]), float(sub[se].iloc[0]), d=3)
                     for ec, se in [("eps0","eps0_se"),("eps1","eps1_se"),("eps2","eps2_se")]]
        tex_lines.append(f"    {mn} & " + " & ".join(row_cells) + r" \\")
    tex_lines += [r"    \bottomrule", r"  \end{tabular}", r"  \end{threeparttable}",
                  r"\end{table}", ""]

    # ── Welfare table (CES DGP) ───────────────────────────────────────────────
    tex_lines += [
        r"\begin{table}[htbp]", r"  \centering",
        r"  \caption{Compensating Variation — Simulation (CES DGP, 20\% shock)}",
        r"  \label{tab:sim_welfare}",
        r"  \begin{threeparttable}",
        r"  \begin{tabular}{lcc}", r"    \toprule",
        r"    \textbf{Model} & \textbf{CV} & \textbf{Error vs Truth (\%)} \\",
        r"    \midrule",
    ]
    ces_welf = welf_agg[welf_agg["DGP"] == "CES"]
    for mn in ces_welf["Model"].unique():
        sub = ces_welf[ces_welf["Model"] == mn]
        if not len(sub): continue
        cv_str  = _cell(float(sub["CV"].iloc[0]), float(sub["CV_se"].iloc[0]), d=4)
        err_str = (f"{float(sub['CV_err_pct'].iloc[0]):.1f}\\%"
                   if mn != "Ground Truth" else "---")
        tex_lines.append(f"    {mn} & {cv_str} & {err_str} \\\\")
    tex_lines += [r"    \bottomrule", r"  \end{tabular}", r"  \end{threeparttable}",
                  r"\end{table}", ""]

    # ── Welfare by shock table (CES DGP, three 10% single-good shocks) ───────
    welf_by_shock_agg = agg.get("welf_by_shock_agg", pd.DataFrame())
    if not welf_by_shock_agg.empty:
        welf_by_shock_agg.round(6).to_csv(
            f"{out_dir}/table_welfare_by_shock.csv", index=False)

        _MODEL_GROUPS = [
            ("Benchmark", ["Ground Truth"]),
            ("Traditional Models", ["LA-AIDS", "Logit-IV", "QUAIDS"]),
            ("Semi-parametric", ["Series Estm."]),
            ("Neural Demand", [
                "Neural Demand (static)",
                "Neural Demand (habit)",
                "Neural Demand (CF)",
                "Neural Demand (habit, CF)",
            ]),
        ]
        _MODEL_DISPLAY_WELFARE = {
            "Ground Truth":              "Ground Truth",
            "LA-AIDS":                   "LA-AIDS",
            "Logit-IV":                  "Logit-IV",
            "QUAIDS":                    "QUAIDS",
            "Series Estm.":              "Series Estm.",
            "Neural Demand (static)":    "Static",
            "Neural Demand (habit)":     "Habit",
            "Neural Demand (CF)":        "Static, CF",
            "Neural Demand (habit, CF)": "Habit, CF",
        }

        ces_wbs = welf_by_shock_agg[welf_by_shock_agg["DGP"] == "CES"]

        def _cv_cell(val, se, d=2):
            if np.isnan(val): return "---"
            return f"${val:.{d}f}$"

        def _se_cell(se, d=2):
            if np.isnan(se): return ""
            return f"$({se:.{d}f})$"

        wbs_lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Compensating Variation --- Simulation (CES DGP)}",
            r"  \label{tab:sim_welfare_updated}",
            r"  \begin{tabular}{llllcc}",
            r"    \toprule",
            r"    & \multicolumn{4}{c}{\textbf{Compensating Variation (CV)}} & \textbf{Error vs.} \\",
            r"    \cmidrule(lr){2-5}",
            r"    \textbf{Model} & \textbf{Shock $g_0$} & \textbf{Shock $g_1$} & \textbf{Shock $g_2$} & \textbf{Average} & \textbf{Truth (\%)} \\",
            r"    \midrule",
        ]

        for group_label, group_models in _MODEL_GROUPS:
            wbs_lines.append(
                rf"    \multicolumn{{2}}{{l}}{{\textit{{{group_label}}}}} & & & & \\"
            )
            for mn in group_models:
                sub = ces_wbs[ces_wbs["Model"] == mn]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                disp = _MODEL_DISPLAY_WELFARE.get(mn, mn)

                cv0  = float(row.get("CV_g0",    np.nan))
                cv1  = float(row.get("CV_g1",    np.nan))
                cv2  = float(row.get("CV_g2",    np.nan))
                se0  = float(row.get("CV_g0_se", np.nan))
                se1  = float(row.get("CV_g1_se", np.nan))
                se2  = float(row.get("CV_g2_se", np.nan))

                # Average CV and its SE (SE of mean of 3 shocks)
                cv_vals = [v for v in [cv0, cv1, cv2] if not np.isnan(v)]
                cv_avg  = np.mean(cv_vals) if cv_vals else np.nan
                se_vals = [s for s in [se0, se1, se2] if not np.isnan(s)]
                se_avg  = (np.sqrt(sum(s**2 for s in se_vals)) / len(se_vals)
                           if se_vals else np.nan)

                # Average error = mean of per-shock pct errors
                err_vals = [float(row.get(f"CV_err_g{g}", np.nan)) for g in range(3)]
                err_valid = [e for e in err_vals if not np.isnan(e)]
                err_avg   = np.mean(err_valid) if err_valid else np.nan

                err_str = ("---" if mn == "Ground Truth"
                           else (f"{err_avg:.1f}\\%" if not np.isnan(err_avg) else "---"))

                wbs_lines.append(
                    f"    {disp} & "
                    f"{_cv_cell(cv0, se0)} & "
                    f"{_cv_cell(cv1, se1)} & "
                    f"{_cv_cell(cv2, se2)} & "
                    f"{_cv_cell(cv_avg, se_avg)} & "
                    f"{err_str} \\\\"
                )
                wbs_lines.append(
                    f"                 & "
                    f"{_se_cell(se0)} & "
                    f"{_se_cell(se1)} & "
                    f"{_se_cell(se2)} & "
                    f"{_se_cell(se_avg)} & \\\\"
                )
            wbs_lines.append(r"    \addlinespace")

        # Remove trailing \addlinespace
        if wbs_lines and wbs_lines[-1] == r"    \addlinespace":
            wbs_lines.pop()

        wbs_lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{tablenotes}",
            r"    Standard errors are reported in parentheses below the estimates."
            r" Simulations based on CES DGP with three 10\% one-good shocks.",
            r"  \end{tablenotes}",
            r"\end{table}",
        ]

        # Append to the main sim_tables.tex bundle as well
        tex_lines += wbs_lines + [""]

        wbs_tex_path = f"{out_dir}/table_welfare_by_shock.tex"
        with open(wbs_tex_path, "w") as _f:
            _f.write(
                "% ============================================================\n"
                "% Neural Demand — Welfare by Shock (auto-generated)\n"
                f"% N_RUNS = {N_RUNS}\n"
                "% ============================================================\n\n"
            )
            _f.write("\n".join(wbs_lines) + "\n")
        print(f"  Saved by-shock welfare table to {wbs_tex_path}")

    # ── Figure environments ───────────────────────────────────────────────────
    for fn, cap, lbl in [
        ("fig_demand_curves_CES",
         "Demand Curves — CES DGP.  Shaded bands = ±1 SE across runs.",
         "fig:sim_demand_ces"),
        ("fig_dgp_robustness",
         "DGP Robustness: Out-of-Sample RMSE across Five DGPs.",
         "fig:sim_dgp_robustness"),
        ("fig_observed_vs_predicted",
         r"Observed vs.\ Predicted Budget Shares — CES DGP.",
         "fig:sim_obs_pred"),
    ]:
        tex_lines += [
            r"\begin{figure}[htbp]", r"  \centering",
            f"  \\includegraphics[width=0.85\\linewidth]{{{fig_dir}/{fn}.pdf}}",
            f"  \\caption{{{cap}}}", f"  \\label{{{lbl}}}",
            r"\end{figure}", "",
        ]

    tex_path = f"{out_dir}/sim_tables.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines))
    print(f"  Saved LaTeX to {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> tuple:
    """Orchestrate N_RUNS seeds, aggregate, produce figures and tables."""
    N_RUNS = cfg["N_RUNS"]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["fig_dir"], exist_ok=True)

    print("=" * 68)
    print("  Neural Demand — Simulation Exp 01: DGP Recovery")
    print("=" * 68)

    pre = cfg.get("_precomputed_sim_exp01")
    if pre is not None:
        all_results = list(pre)
        print(f"  Using precomputed Exp01 runs ({len(all_results)} seeds)")
    else:
        all_results = []
        for ri in range(N_RUNS):
            seed = 42 + ri * 15
            t0   = time.time()
            print(f"  Run {ri+1}/{N_RUNS} (seed={seed})")
            r = run_one_seed(seed, cfg, verbose=(ri == N_RUNS - 1))
            all_results.append(r)
            print(f"    Done in {time.time()-t0:.0f}s")

    agg = aggregate(all_results)
    make_figures(agg, cfg)
    make_tables(agg, cfg)
    return all_results, agg
