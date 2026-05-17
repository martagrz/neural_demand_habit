"""
experiments/neural_demand/simulation/utils.py
==============================================
Shared prediction and evaluation utilities for Neural Demand simulation
experiments.  Mirrors experiments/simulation/utils.py but with naming
conventions aligned to the Neural Demand paper:

  - "Neural Demand (static)"   ↔  StaticND
  - "Neural Demand (habit)"    ↔  HabitND with fixed δ
  - "LA-AIDS", "QUAIDS", "Series Estm." remain unchanged
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.simulation import (
    BLPBench,
    HabitND,
    compute_xbar_e2e,
    train_neural_irl,
    cf_first_stage,
    features_shared,
    features_good_specific,
    features_orthogonalised,
    predict_linear_irl,
)

# ─────────────────────────────────────────────────────────────────────────────
#  FIXED EVALUATION GRID
# ─────────────────────────────────────────────────────────────────────────────

P_GRID = np.linspace(1, 10, 80)
AVG_Y  = 1600.0
BAND   = 0.15   # alpha for ±1 SE bands

# ─────────────────────────────────────────────────────────────────────────────
#  PAPER MODEL NAMES
# ─────────────────────────────────────────────────────────────────────────────

LIRL_MODELS   = ["Linear Demand (Shared)", "Linear Demand (GoodSpec)", "Linear Demand (Orth)"]
STATIC_MODELS = ["LA-AIDS", "QUAIDS", "Series Estm."] + LIRL_MODELS + ["Neural Demand (static)"]
HABIT_MODELS  = STATIC_MODELS + ["Neural Demand (habit)"]
CF_MODELS     = HABIT_MODELS + ["Neural Demand (CF)", "Neural Demand (habit, CF)"]
ALL_MODELS    = CF_MODELS

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "Truth":                        dict(color="k",        ls="-",  lw=3.0),
    "LA-AIDS":                      dict(color="#E53935",  ls="--", lw=2.0),
    "Logit-IV":                     dict(color="#9C27B0",  ls="--", lw=2.0),
    "QUAIDS":                       dict(color="#43A047",  ls="-.", lw=2.0),
    "Series Estm.":                 dict(color="#FB8C00",  ls=":",  lw=2.0),
    "Linear Demand (Shared)":                 dict(color="#039BE5",  ls=":",  lw=1.5),
    "Linear Demand (GoodSpec)":               dict(color="#00ACC1",  ls=":",  lw=1.5),
    "Linear Demand (Orth)":                   dict(color="#006064",  ls=":",  lw=1.5),
    "Neural Demand (static)":       dict(color="#1E88E5",  ls="-",  lw=2.5),
    "Neural Demand (habit)":        dict(color="#00897B",  ls="-",  lw=2.5),
    "Neural Demand (CF)":           dict(color="#283593",  ls="--", lw=2.0),
    "Neural Demand (habit, CF)":    dict(color="#1B5E20",  ls="--", lw=2.0),
}


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_shares(spec, p, y, *,
                   aids=None, blp=None, quaids=None, series=None,
                   nds=None,        # Neural Demand (static)            — StaticND
                   nds_hab=None,    # Neural Demand (habit)              — HabitND
                   nds_cf=None,     # Neural Demand (CF)                 — StaticND(n_cf=G)
                   nds_hab_cf=None, # Neural Demand (habit, CF)          — HabitND(n_cf=G)
                   xbar_hab=None,   # habit stock (N, G) for habit models
                   q_prev_hab=None, # previous log-quantities (N, G) for nds_hab_cf
                   v_hat=None,      # CF residuals (N, G); zeros at eval/welfare time
                   # Linear IRL
                   theta_sh=None,   # Linear Demand (Shared) coefficients
                   theta_gs=None,   # Linear Demand (GoodSpec) coefficients
                   theta_or=None,   # Linear Demand (Orth) coefficients
                   consumer=None,
                   device="cpu"):
    """Predict budget shares for one of the paper's demand model specs.

    Parameters
    ----------
    spec         : one of "truth", "aids", "quaids", "series",
                   "lirl-shared", "lirl-gs", "lirl-orth",
                   "nd-static", "nd-habit",
                   "nd-static-cf", "nd-habit-cf"
    p            : (N, G) prices
    y            : (N,) or scalar income
    aids         : fitted LAAIDS / AIDSBench
    quaids       : fitted QUAIDS
    series       : fitted SeriesDemand
    theta_sh     : Linear Demand (Shared) coefficient vector
    theta_gs     : Linear Demand (GoodSpec) coefficient vector
    theta_or     : Linear Demand (Orth) coefficient vector
    nds          : trained StaticND (static Neural Demand)
    nds_hab      : trained HabitND (habit-augmented Neural Demand)
    nds_cf       : trained StaticND(n_cf=G)  (static + CF endogeneity correction)
    nds_hab_cf   : trained HabitND(n_cf=G) (habit + CF; uses pre-computed xbar)
    xbar_hab     : (N, G) habit stock for all habit models; zeros if None
    q_prev_hab   : (N, G) previous log-quantities for nds_hab_cf (HabitND);
                   if None uses xbar_hab as fallback
    v_hat        : (N, G) CF first-stage residuals; pass zeros (or None) at
                   evaluation / welfare / counterfactual time to recover
                   the structural demand net of endogeneity bias
    consumer     : true consumer for "truth" spec
    device       : torch device string
    """
    # Normalize to demand-model spec names while keeping legacy aliases.
    spec_alias = {
        "linear-demand-shared": "lirl-shared",
        "linear-demand-goodspecific": "lirl-gs",
        "linear-demand-orth": "lirl-orth",
        "neural-demand-static": "nd-static",
        "neural-demand-habit": "nd-habit",
        "neural-demand-static-cf": "nd-static-cf",
        "neural-demand-habit-cf": "nd-habit-cf",
    }
    spec = spec_alias.get(spec, spec)

    if np.ndim(y) == 0:
        y = np.full(len(p), float(y))

    N = len(p)
    G = p.shape[1]

    if spec == "truth":
        return consumer.solve_demand(p, y)
    if spec == "aids":
        return aids.predict(p, y)
    if spec == "blp":
        # All 3 goods are inside goods; predict() returns (N, 4) =
        # [s_0, s_1, s_2, s_outside].  Drop the outside-option column so the
        # result is (N, 3), matching w_true — mirrors main_multiple_runs.py.
        y_arr = np.full(len(p), float(y)) if np.ndim(y) == 0 else np.asarray(y)
        return blp.predict(p, y_arr)[:, :3]   # (N, 3)
    if spec == "quaids":
        return quaids.predict(p, y)
    if spec == "series":
        return series.predict(p, y)
    if spec == "lirl-shared":
        F = features_shared(p, y)
        return predict_linear_irl(F, theta_sh)
    if spec == "lirl-gs":
        F = features_good_specific(p, y)
        return predict_linear_irl(F, theta_gs)
    if spec == "lirl-orth":
        F = features_orthogonalised(p, y)
        return predict_linear_irl(F, theta_or)
    if spec == "nd-static":
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8),
                                        dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(np.maximum(y, 1e-8),
                                        dtype=torch.float32, device=device)).unsqueeze(1)
            return nds(lp, ly).cpu().numpy()
    if spec == "nd-habit":
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8),
                                        dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(np.maximum(y, 1e-8),
                                        dtype=torch.float32, device=device)).unsqueeze(1)
            if xbar_hab is None:
                xbar_hab = np.zeros((N, G))
            xb = torch.tensor(xbar_hab, dtype=torch.float32, device=device)

            qp = (torch.tensor(q_prev_hab, dtype=torch.float32, device=device)
                  if q_prev_hab is not None else xb.clone())
            return nds_hab(lp, ly, xb, qp).cpu().numpy()
    if spec == "nd-static-cf":
        # StaticND(n_cf=G): pass v_hat at training time; zeros at eval/welfare time
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8),
                                        dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(np.maximum(y, 1e-8),
                                        dtype=torch.float32, device=device)).unsqueeze(1)
            vh = (torch.tensor(v_hat, dtype=torch.float32, device=device)
                  if v_hat is not None else torch.zeros(N, G, device=device))
            return nds_cf(lp, ly, v_hat=vh).cpu().numpy()
    if spec == "nd-habit-cf":
        # HabitND(n_cf=G): takes (lp, ly, log_xb_prev, log_q_prev, v_hat)
        # xbar_hab should be in log-quantity space (EWMA of log-quantities)
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8),
                                        dtype=torch.float32, device=device))
            ly = torch.log(torch.tensor(np.maximum(y, 1e-8),
                                        dtype=torch.float32, device=device)).unsqueeze(1)
            if xbar_hab is None:
                xbar_hab = np.zeros((N, G))
            xb = torch.tensor(xbar_hab, dtype=torch.float32, device=device)
            qp = (torch.tensor(q_prev_hab, dtype=torch.float32, device=device)
                  if q_prev_hab is not None else xb.clone())
            vh = (torch.tensor(v_hat, dtype=torch.float32, device=device)
                  if v_hat is not None else torch.zeros(N, G, device=device))
            return nds_hab_cf(lp, ly, xb, qp, v_hat=vh).cpu().numpy()
    raise ValueError(f"Unknown spec: {spec!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def get_metrics(spec, p, y, w_true, **kw):
    """Return dict with RMSE and MAE for one spec.

    All habit-stock kwargs (xbar_hab, q_prev_hab, v_hat, …) are forwarded
    directly through **kw to predict_shares — do NOT pass xbar_hab both as
    a named positional arg and inside **kw or Python will raise a
    "multiple values" TypeError (which was previously swallowed silently by
    the caller's try/except, giving RMSE = NaN for all habit models).
    """
    wp = predict_shares(spec, p, y, **kw)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(w_true, wp))),
        "MAE":  float(mean_absolute_error(w_true, wp)),
    }


def kl_div(w_pred, w_true):
    """Mean KL divergence KL(w_true ‖ w_pred)."""
    wp = np.clip(w_pred, 1e-8, 1.0)
    wt = np.clip(w_true, 1e-8, 1.0)
    return float(np.mean(np.sum(wt * np.log(wt / wp), axis=1)))


def compute_own_elasticities(spec, p_pt, y_pt, h=1e-4, xbar_pt=None, **kw):
    """Own-price quantity elasticities at point (p_pt, y_pt).

    Returns list of length G: eps[i] = d log q_i / d log p_i
    evaluated as (d log w_i / d log p_i) - 1  (Slutsky equation approx).
    """
    xb_1d = xbar_pt.reshape(1, -1) if xbar_pt is not None else None
    w0 = predict_shares(spec, p_pt.reshape(1, -1), np.array([y_pt]),
                        xbar_hab=xb_1d, **kw)[0]
    G  = len(w0)
    eps = []
    for i in range(G):
        p1    = p_pt.copy().reshape(1, -1)
        p1[0, i] *= (1 + h)
        w1    = predict_shares(spec, p1, np.array([y_pt]),
                               xbar_hab=xb_1d, **kw)[0]
        eps.append(((w1[i] - w0[i]) / max(w0[i], 1e-9)) / h - 1)
    return np.array(eps)


def compute_cross_elasticity_matrix(spec, p_pt, y_pt, h=1e-4, xbar_pt=None, **kw):
    """(G, G) cross-price elasticity matrix eps[i, j] = d log q_j / d log p_i."""
    xb_1d = xbar_pt.reshape(1, -1) if xbar_pt is not None else None
    w0    = predict_shares(spec, p_pt.reshape(1, -1), np.array([y_pt]),
                           xbar_hab=xb_1d, **kw)[0]
    G     = len(w0)
    eps   = np.zeros((G, G))
    for i in range(G):
        p1 = p_pt.copy().reshape(1, -1)
        p1[0, i] *= (1 + h)
        w1 = predict_shares(spec, p1, np.array([y_pt]),
                            xbar_hab=xb_1d, **kw)[0]
        for j in range(G):
            eps[i, j] = ((w1[j] - w0[j]) / max(w0[j], 1e-9)) / h - (1 if i == j else 0)
    return eps


def compute_compensating_variation(spec, p0, p1, y, steps=100,
                                   xbar_pt=None, q_prev_pt=None, **kw):
    """CV via Riemann integral of Hicksian demand along price path p0→p1.

    Parameters
    ----------
    xbar_pt   : (G,) array — initial habit stock passed as ``xbar_hab`` to
                predict_shares.  Both Habit and Habit+CF use log-quantities.
    q_prev_pt : (G,) array — initial previous-quantity state passed as
                ``q_prev_hab`` to predict_shares (Habit+CF model only).
                When None, predict_shares falls back to ``xb.clone()``.
    """
    xb_1d = xbar_pt.reshape(1, -1) if xbar_pt is not None else None
    qp_1d = q_prev_pt.reshape(1, -1) if q_prev_pt is not None else None
    path  = np.linspace(p0, p1, steps)
    dp    = (p1 - p0) / steps
    loss  = 0.0
    for t in range(steps):
        w = predict_shares(spec, path[t:t+1], np.array([y]),
                           xbar_hab=xb_1d, q_prev_hab=qp_1d, **kw)[0]
        loss -= (w * y / path[t]) @ dp
    return loss


# ─────────────────────────────────────────────────────────────────────────────
#  PROFILE-CRITERION δ SWEEP
#  For each δ in the grid, train a frozen-δ neural demand model and evaluate
#  hold-out KL divergence.  δ̂ = argmin KL_val.
# ─────────────────────────────────────────────────────────────────────────────

def fit_neural_demand_delta_grid(
    p_tr,  y_tr,  w_tr,  lq_tr,
    p_val, y_val, w_val, lq_val,
    delta_grid,
    epochs        = 500,
    lr            = 5e-4,
    batch_size    = 256,
    lam_mono      = 0.3,
    lam_slut      = 0.1,
    hidden_dim    = 256,
    n_goods       = 3,
    se_multiplier = 2.0,
    device        = "cpu",
    tag           = "nd-habit-delta",
    cache_dir     = None,
    force_retrain = False,
):
    """Train one fixed-δ Neural Demand (habit) model per grid point.

    Returns
    -------
    dict with:
        best_model : HabitND at δ̂
        delta_hat  : float — selected δ
        kl_grid    : (K,) mean validation KL at each δ
        se_grid    : (K,) SE of per-obs KL
        id_set     : (lo, hi) — identified set
        id_mask    : (K,) bool
        all_models : {float(δ): model}
        all_hists  : {float(δ): hist}
    """
    delta_grid = np.asarray(delta_grid, dtype=float)
    K          = len(delta_grid)
    kl_grid    = np.zeros(K)
    se_grid    = np.zeros(K)
    all_models = {}
    all_hists  = {}

    lq_tr_t  = torch.tensor(lq_tr, dtype=torch.float32, device=device)
    lq_val_t = torch.tensor(lq_val, dtype=torch.float32, device=device)
    q_prev_tr = np.roll(lq_tr, 1, axis=0); q_prev_tr[0] = lq_tr[0]
    q_prev_val = np.roll(lq_val, 1, axis=0); q_prev_val[0] = lq_val[0]
    def _lag_xbar(xb):
        """Align xbar with HabitND input contract: pass xbar_{t-1} at row t."""
        out = np.empty_like(xb)
        out[0] = xb[0]
        out[1:] = xb[:-1]
        return out

    for k, d in enumerate(delta_grid):
        model = HabitND(n_goods=n_goods, hidden_dim=hidden_dim,
                             delta_init=float(d))
        d_t = torch.tensor(float(d), dtype=torch.float32, device=device)
        with torch.no_grad():
            xb_tr_raw  = compute_xbar_e2e(d_t, lq_tr_t,  store_ids=None).cpu().numpy()
            xb_val_raw = compute_xbar_e2e(d_t, lq_val_t, store_ids=None).cpu().numpy()
        xb_tr = _lag_xbar(xb_tr_raw)
        xb_val = _lag_xbar(xb_val_raw)

        model, hist = train_neural_irl(
            model, p_tr, y_tr, w_tr,
            epochs=epochs, lr=lr, batch_size=batch_size,
            lam_mono=lam_mono, lam_slut=lam_slut, slut_start_frac=0.25,
            xb_prev_data=np.exp(xb_tr),
            q_prev_data=np.exp(q_prev_tr),
            device=device,
            tag=f"{tag}-d{d:.2f}",
            cache_dir=cache_dir,
            force_retrain=force_retrain,
        )

        # validation KL
        wp = predict_shares("neural-demand-habit", p_val, y_val,
                            nds_hab=model, xbar_hab=xb_val, q_prev_hab=q_prev_val,
                            device=device)
        wt = np.clip(w_val, 1e-8, 1.0)
        wp = np.clip(wp,    1e-8, 1.0)
        kl_obs    = np.sum(wt * np.log(wt / wp), axis=1)
        kl_grid[k] = float(kl_obs.mean())
        se_grid[k] = float(kl_obs.std(ddof=min(1, len(kl_obs) - 1))
                           / np.sqrt(max(len(kl_obs), 1)))
        all_models[float(d)] = model
        all_hists[float(d)]  = hist
        print(f"  {tag} δ={d:.2f}  val-KL={kl_grid[k]:.6f} ± {se_grid[k]:.6f}")

    best_k     = int(np.argmin(kl_grid))
    delta_hat  = float(delta_grid[best_k])
    best_model = all_models[delta_hat]
    best_hist  = all_hists[delta_hat]

    threshold = kl_grid[best_k] + se_multiplier * se_grid[best_k]
    id_mask   = kl_grid <= threshold
    id_deltas = delta_grid[id_mask]
    id_set    = (float(id_deltas.min()), float(id_deltas.max())) if id_mask.any() \
                else (delta_hat, delta_hat)

    print(f"  → δ̂ = {delta_hat:.2f}  IS=[{id_set[0]:.2f}, {id_set[1]:.2f}]  "
          f"(KL_min={kl_grid[best_k]:.6f})")

    return dict(
        best_model=best_model,
        best_hist=best_hist,
        delta_hat=delta_hat,
        kl_grid=kl_grid,
        se_grid=se_grid,
        id_set=id_set,
        id_mask=id_mask,
        all_models=all_models,
        all_hists=all_hists,
    )
