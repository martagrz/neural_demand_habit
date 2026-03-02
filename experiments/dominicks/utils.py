"""
experiments/dominicks/utils.py
================================
Section 7 of dominicks_multiple_runs.py.

Prediction, evaluation, and welfare utilities for the Dominick's experiments.
All functions accept a config dict (cfg) rather than using globals.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from experiments.dominicks.data import G


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def pred(spec: str, p, y,
         cfg: dict,
         xb_prev=None, q_prev=None,
         store_idx=None,
         s_te_mode_idx: int = 0,
         **kw):
    """Unified prediction helper for all Dominick's model specs.

    For MDP models pass *both* xb_prev (log-normalised habit stock) and
    q_prev (log-normalised previous quantities).
    For store-FE models pass *store_idx* as an integer array.
    When store_idx is None for an FE model the modal test-store index is used.
    """
    dev = cfg['device']
    spec_alias = {
        # New demand-model naming
        "linear-demand": "lirl",
        "neural-demand-static": "nirl",
        "neural-demand-window": "window-irl",
        "neural-demand-habit": "mdp",
        "neural-demand-habit-e2e": "mdp-e2e",
        "neural-demand-static-fe": "nirl-fe",
        "neural-demand-habit-fe": "mdp-fe",
        "neural-demand-habit-e2e-fe": "mdp-e2e-fe",
        "neural-demand-static-cf": "nirl-cf",
        "neural-demand-habit-cf": "mdp-cf",
        "neural-demand-habit-fe-cf": "mdp-fe-cf",
        # Legacy aliases kept for backward compatibility
        "window-irl": "window-irl",
        "lirl": "lirl",
        "nirl": "nirl",
        "mdp": "mdp",
        "mdp-e2e": "mdp-e2e",
        "nirl-fe": "nirl-fe",
        "mdp-fe": "mdp-fe",
        "mdp-e2e-fe": "mdp-e2e-fe",
        "nirl-cf": "nirl-cf",
        "mdp-cf": "mdp-cf",
        "mdp-fe-cf": "mdp-fe-cf",
    }
    spec = spec_alias.get(spec, spec)

    if spec == 'aids':   return kw['aids'].predict(p, y)
    if spec == 'blp':
        return kw['blp'].predict(p)[:, :G]        # inside market shares (N, G); drop outside-option col
    if spec == 'quaids': return kw['quaids'].predict(p, y)
    if spec == 'series': return kw['series'].predict(p, y)

    if spec == 'window-irl':
        _wm   = kw['wirl']
        # Default to zeros if not provided (e.g. for evaluation where we might not have history)
        # But evaluation should use the same features as training.
        # The caller (exp01_main_runs.py) passes wirl_log_p_hist etc.
        # If they are missing, we can't build features correctly.
        # However, for prediction on test set, we need test set history.
        # The current implementation of 'window-irl' in exp01_main_runs.py
        # builds features using build_window_features on the fly for training,
        # but for prediction it relies on this helper.
        
        # We need to rebuild features for the input p, y.
        # But 'pred' only gets p, y (current). It doesn't get history.
        # WindowND needs history.
        # If this is called from 'get_metrics' or 'kl_div', we are iterating over test set.
        # We can't easily reconstruct history from just p, y rows unless we know the sequence.
        
        # HACK: If we are just predicting for a few points (demand curves), use mean history.
        # If we are predicting for the full test set (RMSE), we should have passed
        # the pre-built feature matrix or the full sequence.
        
        # Check if 'wirl_features' is passed in kw
        if 'wirl_features' in kw:
             feats = kw['wirl_features']
             # If feats is (N, in_dim), and p is (N, G), we can just use feats.
             # But p might be a grid (demand curve).
             if len(p) != len(feats):
                 # This happens during demand curve generation (p is grid, feats is test set).
                 # We must use mean history from training (or test) and current p.
                 pass # Fall through to mean history logic below
             else:
                 with torch.no_grad():
                     xt = torch.tensor(feats, dtype=torch.float32, device=dev)
                     return _wm(xt).cpu().numpy()

        # Fallback: use mean history (for demand curves or if features missing)
        _lp_h = kw.get('wirl_log_p_hist', np.zeros(G))
        _lq_h = kw.get('wirl_log_q_hist', np.zeros(G))
        _ww   = kw.get('wirl_window', 4)
        lp_cur = np.log(np.maximum(p, 1e-8))
        ly_cur = np.log(np.maximum(y, 1e-8))
        in_dim = G + 1 + _ww * 2 * G
        feats  = np.zeros((len(p), in_dim), dtype=np.float32)
        # Repeat mean history for the window
        hist_f = np.concatenate([_lp_h, _lq_h] * _ww)
        for _i in range(len(p)):
            feats[_i] = np.concatenate([lp_cur[_i], [ly_cur[_i]], hist_f])
        with torch.no_grad():
            xt = torch.tensor(feats, dtype=torch.float32, device=dev)
            return _wm(xt).cpu().numpy()

    if spec == 'lirl':
        from src.models.dominicks import pred_lirl
        return pred_lirl(kw['ff'], kw['theta'], p, y)

    if spec == 'nirl':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            return kw['nirl'](lp, ly).cpu().numpy()

    if spec == 'mdp':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly  = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                   .unsqueeze(1).to(dev))
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            return kw['mdp'](lp, ly, xbp, qp).cpu().numpy()

    if spec == 'mdp-e2e':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            xb = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            return kw['mdp_e2e'](lp, ly, xb).cpu().numpy()

    # ── Control-function (CF) variants ────────────────────────────────────────
    if spec == 'nirl-cf':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            _m = kw['nirl_cf']
            vh = torch.zeros(len(p), _m.n_cf, dtype=torch.float32).to(dev)
            return _m(lp, ly, v_hat=vh).cpu().numpy()

    if spec == 'mdp-cf':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly  = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                   .unsqueeze(1).to(dev))
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            _m  = kw['mdp_cf']
            vh  = torch.zeros(len(p), _m.n_cf, dtype=torch.float32).to(dev)
            return _m(lp, ly, xbp, qp, v_hat=vh).cpu().numpy()

    if spec == 'mdp-fe-cf':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly  = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                   .unsqueeze(1).to(dev))
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            _si = (store_idx if store_idx is not None
                   else np.full(len(p), s_te_mode_idx, dtype=np.int64))
            si  = torch.tensor(_si, dtype=torch.long).to(dev)
            _m  = kw['mdp_fe_cf']
            vh  = torch.zeros(len(p), _m.n_cf, dtype=torch.float32).to(dev)
            return _m(lp, ly, xbp, qp, si, v_hat=vh).cpu().numpy()

    # ── Store-FE variants ─────────────────────────────────────────────────────
    if spec == 'nirl-fe':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            _si = (store_idx if store_idx is not None
                   else np.full(len(p), s_te_mode_idx, dtype=np.int64))
            si = torch.tensor(_si, dtype=torch.long).to(dev)
            return kw['nirl_fe'](lp, ly, si).cpu().numpy()

    if spec == 'mdp-fe':
        with torch.no_grad():
            lp  = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly  = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                   .unsqueeze(1).to(dev))
            xbp = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            qp  = torch.tensor(q_prev,  dtype=torch.float32).to(dev)
            _si = (store_idx if store_idx is not None
                   else np.full(len(p), s_te_mode_idx, dtype=np.int64))
            si = torch.tensor(_si, dtype=torch.long).to(dev)
            return kw['mdp_fe'](lp, ly, xbp, qp, si).cpu().numpy()

    if spec == 'mdp-e2e-fe':
        with torch.no_grad():
            lp = torch.log(torch.tensor(np.maximum(p, 1e-8), dtype=torch.float32)).to(dev)
            ly = (torch.log(torch.tensor(np.maximum(y, 1e-8), dtype=torch.float32))
                  .unsqueeze(1).to(dev))
            xb = torch.tensor(xb_prev, dtype=torch.float32).to(dev)
            _si = (store_idx if store_idx is not None
                   else np.full(len(p), s_te_mode_idx, dtype=np.int64))
            si = torch.tensor(_si, dtype=torch.long).to(dev)
            return kw['mdp_e2e_fe'](lp, ly, xb, si).cpu().numpy()

    raise ValueError(f"Unknown spec: {spec!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  ELASTICITIES
# ─────────────────────────────────────────────────────────────────────────────

def own_elasticity(spec, p0, y0, cfg, xb_prev0=None, q_prev0=None, h=1e-4, **kw):
    w0  = pred(spec, p0[None], np.array([y0]), cfg,
               xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
               q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
    eps = []
    for i in range(G):
        p1 = p0.copy()[None]; p1[0, i] *= 1 + h
        w1 = pred(spec, p1, np.array([y0]), cfg,
                  xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
                  q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
        eps.append(((w1[i] - w0[i]) / w0[i]) / h - 1)
    return np.array(eps)


def full_elasticity_matrix(spec, p0, y0, cfg, xb_prev0=None, q_prev0=None,
                            h=1e-4, **kw):
    """Return (G, G) matrix of price elasticities.

    eps[i, j] = d log(w_j) / d log(p_i)
    """
    _kw_xb = dict(
        xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
        q_prev =q_prev0[None]  if q_prev0  is not None else None,
    )
    w0  = pred(spec, p0[None], np.array([y0]), cfg, **_kw_xb, **kw)[0]
    eps = np.zeros((G, G))
    for i in range(G):
        p1 = p0.copy()[None]; p1[0, i] *= (1 + h)
        w1 = pred(spec, p1, np.array([y0]), cfg, **_kw_xb, **kw)[0]
        for j in range(G):
            eps[i, j] = ((w1[j] - w0[j]) / max(w0[j], 1e-9)) / h
            if i == j:
                eps[i, j] -= 1.0
    return eps


# ─────────────────────────────────────────────────────────────────────────────
#  WELFARE (COMPENSATING VARIATION)
# ─────────────────────────────────────────────────────────────────────────────

def comp_var(spec, p0, p1, y, cfg, xb_prev0=None, q_prev0=None, **kw):
    cv_steps = cfg['cv_steps']
    path = np.linspace(p0, p1, cv_steps)
    dp   = (p1 - p0) / cv_steps
    cv   = 0.0
    for t in range(cv_steps):
        w   = pred(spec, path[t:t+1], np.array([y]), cfg,
                   xb_prev=xb_prev0[None] if xb_prev0 is not None else None,
                   q_prev =q_prev0[None]  if q_prev0  is not None else None, **kw)[0]
        cv -= (w * y / path[t]) @ dp
    return cv


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def get_metrics(spec, p, y, w_true, cfg, xb_prev=None, q_prev=None, **kw):
    wp = pred(spec, p, y, cfg, xb_prev=xb_prev, q_prev=q_prev, **kw)
    return {'RMSE': np.sqrt(mean_squared_error(w_true, wp)),
            'MAE':  mean_absolute_error(w_true, wp)}


def kl_div(spec, p, y, w_true, cfg, xb_prev=None, q_prev=None, **kw):
    wp = np.clip(pred(spec, p, y, cfg, xb_prev=xb_prev,
                      q_prev=q_prev, **kw), 1e-8, 1.0)
    wt = np.clip(w_true, 1e-8, 1.0)
    return float(np.mean(np.sum(wt * np.log(wt / wp), 1)))


# ─────────────────────────────────────────────────────────────────────────────
#  HABIT-STOCK CONDITIONAL ON PRICE GRID (for smooth demand curves)
# ─────────────────────────────────────────────────────────────────────────────

def mdp_price_cond_habit(p_grid, shock_g: int, p_te, xb_te, qp_te,
                          bandwidth=None):
    """Gaussian-kernel-weighted average habit stock conditional on price grid.

    Returns (xbr, qpr) each of shape (N_grid, G), suitable for _pred kwargs.
    """
    if bandwidth is None:
        bandwidth = 0.45 * np.std(p_te[:, shock_g])
        bandwidth = max(bandwidth, 1e-3)

    xbr = np.zeros((len(p_grid), G))
    qpr = np.zeros((len(p_grid), G))
    for k, pg in enumerate(p_grid):
        dists   = np.abs(p_te[:, shock_g] - pg)
        weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
        weights = weights / (weights.sum() + 1e-12)
        xbr[k]  = (xb_te * weights[:, None]).sum(0)
        qpr[k]  = (qp_te * weights[:, None]).sum(0)
    return xbr, qpr


# ─────────────────────────────────────────────────────────────────────────────
#  FROZEN-δ GRID SWEEP (Dominick's version)
# ─────────────────────────────────────────────────────────────────────────────

MDP_DELTA_GRID_DOM = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def fit_mdp_delta_grid_dom(
    p_tr, y_tr, w_tr, ls_tr,
    p_val, y_val, w_val, ls_val,
    cfg: dict,
    v_hat_tr=None,
    delta_grid=None,
    store_ids_tr=None,
    store_ids_val=None,
    store_idx_tr=None,
    store_idx_val=None,
    hidden=256,
    model_class=None,
    pred_spec=None,
    pred_model_key=None,
    extra_model_kw=None,
    se_multiplier=2.0,
    tag="dom-delta-sweep",
):
    """Train one frozen-δ model per grid point on Dominick's data.

    δ̂ = argmin test-KL; identified set uses ``se_multiplier × SE`` rule.
    """
    from src.models.dominicks import (
        HabitND,
        HabitND_FE,
        _train,
        compute_xbar_e2e,
    )

    if delta_grid is None:
        delta_grid = MDP_DELTA_GRID_DOM
    if model_class is None:
        model_class = HabitND_FE if 'n_stores' in (extra_model_kw or {}) else HabitND
    if extra_model_kw is None:
        extra_model_kw = {}

    if pred_spec is None:
        pred_spec = 'mdp-fe' if 'n_stores' in extra_model_kw else 'mdp'
    if pred_model_key is None:
        pred_model_key = ('mdp_fe' if 'n_stores' in extra_model_kw else 'mdp')

    delta_grid = np.asarray(delta_grid, dtype=float)
    K          = len(delta_grid)
    kl_grid    = np.zeros(K)
    se_grid    = np.zeros(K)
    all_models = {}
    all_hists  = {}
    dev        = cfg['device']
    verbose    = bool(cfg.get("verbose", False))

    ls_tr_t = torch.tensor(ls_tr, dtype=torch.float32, device=dev)
    ls_val_t = torch.tensor(ls_val, dtype=torch.float32, device=dev)

    def _lag_with_store(arr, store_ids=None):
        out = np.roll(arr, 1, axis=0)
        out[0] = arr[0]
        if store_ids is not None:
            for i in range(1, len(arr)):
                if store_ids[i] != store_ids[i - 1]:
                    out[i] = arr[i]
        return out
    def _lag_xbar_with_store(xb, store_ids=None):
        """Align xbar with HabitND input contract: row t should carry xbar_{t-1}."""
        out = np.empty_like(xb)
        out[0] = xb[0]
        for i in range(1, len(xb)):
            if store_ids is not None and store_ids[i] != store_ids[i - 1]:
                out[i] = xb[i]
            else:
                out[i] = xb[i - 1]
        return out

    for k, d in enumerate(delta_grid):
        mkw = dict(n_goods=G, hidden_dim=hidden, delta_init=float(d))
        mkw.update(extra_model_kw)
        model = model_class(**mkw)

        d_t = torch.tensor(float(d), dtype=torch.float32, device=dev)
        with torch.no_grad():
            xb_tr_raw = compute_xbar_e2e(
                d_t, ls_tr_t, store_ids=store_ids_tr).cpu().numpy()
            xb_val_raw = compute_xbar_e2e(
                d_t, ls_val_t, store_ids=store_ids_val).cpu().numpy()
        xb_tr = _lag_xbar_with_store(xb_tr_raw, store_ids_tr)
        xb_val = _lag_xbar_with_store(xb_val_raw, store_ids_val)
        q_prev_tr = _lag_with_store(ls_tr, store_ids_tr)
        q_prev_val = _lag_with_store(ls_val, store_ids_val)

        _train_kwargs = dict(
            xb_prev_tr=xb_tr,
            q_prev_tr=q_prev_tr,
            tag=f"{tag}-d{d:.2f}",
        )
        if v_hat_tr is not None:
            _train_kwargs["v_hat_tr"] = v_hat_tr
        if store_idx_tr is not None and pred_model_key in ("mdp_fe", "mdp_fe_cf"):
            _train_kwargs["store_idx_tr"] = store_idx_tr
        model, _hist = _train(model, p_tr, y_tr, w_tr, "mdp", cfg, **_train_kwargs)

        pred_kw = {pred_model_key: model, 'xb_prev': xb_val, 'q_prev': q_prev_val}
        if store_idx_val is not None:
            pred_kw['store_idx'] = store_idx_val
        wp = np.clip(pred(pred_spec, p_val, y_val, cfg, **pred_kw), 1e-8, 1.0)
        wt = np.clip(w_val, 1e-8, 1.0)
        kl_per_obs = np.sum(wt * np.log(wt / wp), axis=1)
        kl_grid[k] = float(kl_per_obs.mean())
        se_grid[k] = float(kl_per_obs.std(ddof=min(1, len(kl_per_obs) - 1))
                           / np.sqrt(max(len(kl_per_obs), 1)))
        all_models[float(d)] = model
        all_hists[float(d)]  = _hist
        if verbose:
            print(f"    {tag} δ={d:.2f}: test-KL={kl_grid[k]:.6f} ± {se_grid[k]:.6f}")

    best_k     = int(np.argmin(kl_grid))
    delta_hat  = float(delta_grid[best_k])
    best_model = all_models[delta_hat]
    best_hist  = all_hists[delta_hat]

    threshold = kl_grid[best_k] + se_multiplier * se_grid[best_k]
    id_mask   = kl_grid <= threshold
    id_deltas = delta_grid[id_mask]
    id_set    = ((float(id_deltas.min()), float(id_deltas.max()))
                 if id_mask.any() else (delta_hat, delta_hat))

    print(f"  → {tag} δ̂={delta_hat:.2f}  IS=[{id_set[0]:.2f}, {id_set[1]:.2f}]  "
          f"(KL_min={kl_grid[best_k]:.6f})")
    return {
        'best_model': best_model,
        'best_hist':  best_hist,
        'delta_hat':  delta_hat,
        'kl_grid':    kl_grid,
        'se_grid':    se_grid,
        'id_set':     id_set,
        'id_mask':    id_mask,
        'all_models': all_models,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def dm_test_by_store(resid1, resid2, store_ids):
    """Diebold-Mariano type test blocking by store.

    1. Compute loss difference d_it = e1_it^2 - e2_it^2 for each observation.
       (Sum of squared errors across goods if multivariate).
    2. Compute mean difference d_i for each store i.
    3. Perform paired t-test on d_i values (H0: mean difference is 0).

    Returns
    -------
    t_stat : float
    p_val  : float
    mean_diff : float (positive means model 2 is better if d = e1^2 - e2^2)
    """
    if resid1.ndim > 1:
        loss1 = (resid1**2).sum(axis=1)
        loss2 = (resid2**2).sum(axis=1)
    else:
        loss1 = resid1**2
        loss2 = resid2**2

    diff = loss1 - loss2

    # Aggregate by store
    import pandas as pd
    from scipy import stats

    df = pd.DataFrame({'diff': diff, 'store': store_ids})
    store_means = df.groupby('store')['diff'].mean()

    n = len(store_means)
    mean_diff = store_means.mean()
    std_diff  = store_means.std(ddof=1)
    se_diff   = std_diff / np.sqrt(n)

    if se_diff < 1e-12:
        return 0.0, 1.0, 0.0

    t_stat = mean_diff / se_diff
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    return t_stat, p_val, mean_diff


# ====== Combined from former neural_utils.py ======
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.dominicks.data import GOODS

# Keep references to low-level utilities before paper-facing wrapper overrides.
_pred_dom = pred
_own_elast_dom = own_elasticity
_full_elast_dom = full_elasticity_matrix
_comp_var_dom = comp_var
_get_metrics_dom = get_metrics
_kl_div_dom = kl_div

MODEL_NAMES_STATIC = [
    "LA-AIDS", "BLP (IV)", "QUAIDS", "Series Est.",
    "Linear Demand (Shared)", "Linear Demand (GoodSpec)", "Linear Demand (Orth)",
    "Neural Demand (static)", "Neural Demand (window)",
]
MODEL_NAMES_HABIT = MODEL_NAMES_STATIC + ["Neural Demand (habit)"]
MODEL_NAMES_FE    = MODEL_NAMES_HABIT  + [
    "Neural Demand (FE)", "Neural Demand (habit, FE)",
]
MODEL_NAMES_CF = MODEL_NAMES_FE + [
    "Neural Demand (CF)", "Neural Demand (habit, CF)",
]
ALL_MODEL_NAMES = MODEL_NAMES_CF

# Map: paper name → internal spec used by pred()
_SPEC_MAP = {
    "LA-AIDS":                       "aids",
    "BLP (IV)":                      "blp",
    "QUAIDS":                        "quaids",
    "Series Est.":                   "series",
    "Linear Demand (Shared)":                  "linear-demand",
    "Linear Demand (GoodSpec)":                "linear-demand",
    "Linear Demand (Orth)":                    "linear-demand",
    "Neural Demand (static)":        "neural-demand-static",
    "Neural Demand (window)":        "neural-demand-window",
    "Neural Demand (habit)":         "neural-demand-habit",
    "Neural Demand (FE)":            "neural-demand-static-fe",
    "Neural Demand (habit, FE)":     "neural-demand-habit-fe",
    "Neural Demand (CF)":            "neural-demand-static-cf",
    "Neural Demand (habit, CF)":     "neural-demand-habit-cf",
}

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR / STYLE MAP
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "LA-AIDS":                    dict(color="#E53935", ls="--", lw=1.8),
    "BLP (IV)":                   dict(color="#8E24AA", ls="--", lw=1.8),
    "QUAIDS":                     dict(color="#43A047", ls="-.", lw=1.8),
    "Series Est.":                dict(color="#FB8C00", ls=":",  lw=1.8),
    "Linear Demand (Shared)":               dict(color="#039BE5", ls=":",  lw=1.5),
    "Linear Demand (GoodSpec)":             dict(color="#00ACC1", ls=":",  lw=1.5),
    "Linear Demand (Orth)":                 dict(color="#0277BD", ls=":",  lw=1.5),
    "Neural Demand (static)":     dict(color="#1E88E5", ls="-",  lw=2.5),
    "Neural Demand (window)":     dict(color="#5E35B1", ls="-",  lw=2.0),
    "Neural Demand (habit)":      dict(color="#00897B", ls="-",  lw=2.5),
    "Neural Demand (FE)":         dict(color="#1565C0", ls="-",  lw=2.0),
    "Neural Demand (habit, FE)":  dict(color="#004D40", ls="-",  lw=2.0),
    "Neural Demand (CF)":         dict(color="#283593", ls="--", lw=2.0),
    "Neural Demand (habit, CF)":  dict(color="#1B5E20", ls="--", lw=2.0),
}

# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED PREDICTION (paper names → internal specs)
# ─────────────────────────────────────────────────────────────────────────────

def predict(paper_name: str, p, y, cfg: dict,
            xb_prev=None, q_prev=None,
            store_idx=None, s_te_mode_idx: int = 0,
            **kw):
    """Predict budget shares using a paper-facing model name.

    Wraps `experiments.dominicks.utils.pred` and maps Neural Demand paper
    names to internal spec strings.

    Parameters
    ----------
    paper_name : str  — one of the keys in MODEL_NAMES_* lists
    p          : (N, G) price array
    y          : (N,) income array
    cfg        : experiment config dict (passed through to pred)
    xb_prev    : (N, G) log-share habit stock (for habit models)
    q_prev     : (N, G) log-quantity of previous period (for MDP models)
    store_idx  : (N,) int store indices (for FE models)
    s_te_mode_idx : int  — modal test-store index (fallback for FE models)
    **kw       : additional keyword args forwarded to pred
                 (aids=, blp=, quaids=, series=, nirl=, mdp=, …)
    """
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _pred_dom(spec, p, y, cfg,
                     xb_prev=xb_prev, q_prev=q_prev,
                     store_idx=store_idx,
                     s_te_mode_idx=s_te_mode_idx,
                     **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC WRAPPERS  (paper names)
# ─────────────────────────────────────────────────────────────────────────────

def metrics(paper_name: str, p, y, w_true, cfg: dict,
            xb_prev=None, q_prev=None, **kw) -> dict:
    """Return {'RMSE': …, 'MAE': …} for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _get_metrics_dom(spec, p, y, w_true, cfg,
                            xb_prev=xb_prev, q_prev=q_prev, **kw)


def kl_divergence(paper_name: str, p, y, w_true, cfg: dict,
                  xb_prev=None, q_prev=None, **kw) -> float:
    """KL(truth‖pred) for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _kl_div_dom(spec, p, y, w_true, cfg,
                       xb_prev=xb_prev, q_prev=q_prev, **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  ELASTICITY / WELFARE WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def own_elasticity(paper_name: str, p0, y0, cfg: dict,
                   xb_prev0=None, q_prev0=None, h=1e-4, **kw):
    """Own-price elasticities vector (G,) for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _own_elast_dom(spec, p0, y0, cfg,
                          xb_prev0=xb_prev0, q_prev0=q_prev0, h=h, **kw)


def elasticity_matrix(paper_name: str, p0, y0, cfg: dict,
                      xb_prev0=None, q_prev0=None, h=1e-4, **kw):
    """(G, G) price elasticity matrix for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _full_elast_dom(spec, p0, y0, cfg,
                           xb_prev0=xb_prev0, q_prev0=q_prev0, h=h, **kw)


def compensating_variation(paper_name: str, p0, p1, y, cfg: dict,
                            xb_prev0=None, q_prev0=None, **kw) -> float:
    """Compensating variation for a paper-facing model name."""
    spec = _SPEC_MAP.get(paper_name, paper_name)
    return _comp_var_dom(spec, p0, p1, y, cfg,
                         xb_prev0=xb_prev0, q_prev0=q_prev0, **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  δ SWEEP (Neural Demand paper alias)
# ─────────────────────────────────────────────────────────────────────────────

def fit_nd_delta_grid_dom(
    p_tr, y_tr, w_tr, ls_tr,
    p_val, y_val, w_val, ls_val,
    cfg: dict,
    delta_grid=None,
    store_ids_tr=None,
    store_ids_val=None,
    store_idx_tr=None,
    store_idx_val=None,
    hidden=256,
    with_fe: bool = False,
    n_stores=None,
    store_emb_dim=None,
    tag: str = "nd-habit-dom",
):
    """Frozen-δ profile sweep for Neural Demand (habit) on Dominick's data.

    This is a thin wrapper around `fit_mdp_delta_grid_dom` that chooses
    the right fixed-δ habit model class (standard vs FE) and spec string.

    Returns the same dict as `fit_mdp_delta_grid_dom`.
    """
    from src.models.dominicks import HabitND, HabitND_FE

    if with_fe:
        n_stores = n_stores if n_stores is not None else cfg.get("n_stores", 1)
        emb_dim  = store_emb_dim if store_emb_dim is not None else cfg.get("store_emb_dim", 8)
        return fit_mdp_delta_grid_dom(
            p_tr, y_tr, w_tr, ls_tr,
            p_val, y_val, w_val, ls_val,
            cfg,
            delta_grid=delta_grid,
            store_ids_tr=store_ids_tr, store_ids_val=store_ids_val,
            store_idx_tr=store_idx_tr, store_idx_val=store_idx_val,
            hidden=hidden,
            model_class=HabitND_FE,
            extra_model_kw={"n_stores": n_stores, "emb_dim": emb_dim},
            pred_spec="neural-demand-habit-fe",
            pred_model_key="mdp_fe",
            tag=tag,
        )
    return fit_mdp_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_val, y_val, w_val, ls_val,
        cfg,
        delta_grid=delta_grid,
        store_ids_tr=store_ids_tr, store_ids_val=store_ids_val,
        hidden=hidden,
        model_class=HabitND,
        pred_spec="neural-demand-habit",
        pred_model_key="mdp",
        tag=tag,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _se(arr):
    a = np.asarray([x for x in arr
                    if x is not None and not np.isnan(float(x))], float)
    if len(a) < 2:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


def aggregate_runs(all_results: list, model_names: list) -> dict:
    """Aggregate metrics over N simulation/bootstrap runs.

    Parameters
    ----------
    all_results : list of dicts from run_once (each with 'perf' sub-dict)
    model_names : list of paper-facing model names to aggregate

    Returns
    -------
    dict with {model_name: {'RMSE_mean', 'RMSE_se', 'MAE_mean', 'MAE_se',
                             'KL_mean', 'KL_se'}}
    """
    n = len(all_results)

    def _mean_se(key, nm):
        vals = []
        for r in all_results:
            v = (r.get("perf", {}).get(nm, {}).get(key)
                 or r.get(key, {}).get(nm))
            if v is not None:
                vals.append(float(v))
        if not vals:
            return np.nan, np.nan
        return float(np.nanmean(vals)), _se(vals)

    out = {}
    for nm in model_names:
        rm, rs = _mean_se("RMSE", nm)
        mm, ms = _mean_se("MAE",  nm)
        km, ks = _mean_se("KL",   nm)
        out[nm] = {"RMSE_mean": rm, "RMSE_se": rs,
                   "MAE_mean":  mm, "MAE_se":  ms,
                   "KL_mean":   km, "KL_se":   ks}
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED FIGURE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

BAND = 0.15   # alpha for ±1 SE bands


def bar_chart(means: dict, ses: dict, ylabel: str, title: str,
              ax=None, n_runs: int = 1) -> plt.Axes:
    """Generic horizontal bar chart for model comparison."""
    names  = list(means.keys())
    vals   = np.asarray([means[n] for n in names], float)
    errs   = np.asarray([ses.get(n, 0.0) for n in names], float)
    colors = [STYLE.get(nm, {}).get("color", "#888") for nm in names]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(names))
    ax.bar(x, vals, yerr=errs if n_runs > 1 else None,
           capsize=5, color=colors, edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    # ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    return ax


def demand_curve_plot(curves: dict, pgr, shock_good_idx: int,
                      ax=None, n_runs: int = 1,
                      ses: dict | None = None,
                      title: str = "Demand Curves") -> plt.Axes:
    """Plot budget shares vs. price for each model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for nm, mu in curves.items():
        sty = STYLE.get(nm, {})
        ax.plot(pgr, mu, label=nm, **sty)
        if n_runs > 1 and ses is not None and nm in ses:
            sig = ses[nm]
            ax.fill_between(pgr, mu - sig, mu + sig,
                            color=sty.get("color", "#888"), alpha=BAND)

    ax.set_xlabel(f"Good-{shock_good_idx} price", fontsize=13)
    ax.set_ylabel(f"Budget share $w_{{0}}$", fontsize=13)
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED TABLE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_performance_table(perf_agg: dict, out_dir: str, label: str,
                            caption: str, n_runs: int = 1) -> None:
    """Write CSV + LaTeX performance table.

    Parameters
    ----------
    perf_agg : dict from aggregate_runs with {model: {RMSE_mean, …}}
    out_dir  : output directory
    label    : LaTeX label stem (e.g. 'dom_main_perf')
    caption  : LaTeX caption string
    n_runs   : number of runs (for SE annotation)
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for nm, d in perf_agg.items():
        rows.append({
            "Model":     nm,
            "RMSE_mean": d.get("RMSE_mean", np.nan),
            "RMSE_se":   d.get("RMSE_se",   0.0),
            "MAE_mean":  d.get("MAE_mean",  np.nan),
            "MAE_se":    d.get("MAE_se",    0.0),
            "KL_mean":   d.get("KL_mean",   np.nan),
            "KL_se":     d.get("KL_se",     0.0),
        })
    df = pd.DataFrame(rows)
    csv_path = f"{out_dir}/{label}.csv"
    df.round(6).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    def _c(m, s, d=5):
        if np.isnan(float(m)):
            return "{---}"
        if n_runs > 1 and s > 0:
            return f"${float(m):.{d}f} \\pm {float(s):.{d}f}$"
        return f"${float(m):.{d}f}$"

    lines = [
        r"% ============================================================",
        f"% Neural Demand — {label} (auto-generated)",
        f"% N_RUNS = {n_runs}",
        r"% ============================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label}}}",
        r"  \begin{threeparttable}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{KL Div.} \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            f"    {row['Model']} & "
            f"{_c(row['RMSE_mean'], row['RMSE_se'])} & "
            f"{_c(row['MAE_mean'],  row['MAE_se'])} & "
            f"{_c(row['KL_mean'],   row['KL_se'])} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}\small",
        rf"    \item Mean $\pm$ SE over {n_runs} run(s). "
        r"RMSE and MAE computed on held-out test weeks.",
        r"  \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}", "",
    ]
    tex_path = f"{out_dir}/{label}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {tex_path}")


def make_elasticity_table(elast_rows: list, out_dir: str, label: str,
                           caption: str, n_runs: int = 1) -> None:
    """Write CSV + LaTeX table of own-price elasticities.

    Parameters
    ----------
    elast_rows : list of dicts with keys 'Model', 'eps_0', 'eps_1', 'eps_2'
                 (and optional '_se' variants)
    """
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(elast_rows).round(4).to_csv(
        f"{out_dir}/{label}.csv", index=False)
    print(f"  Saved: {out_dir}/{label}.csv")

    def _c(v, s=None, d=3):
        if v is None or np.isnan(float(v)):
            return "{---}"
        if n_runs > 1 and s is not None and not np.isnan(float(s)):
            return f"${float(v):.{d}f} \\pm {float(s):.{d}f}$"
        return f"${float(v):.{d}f}$"

    lines = [
        r"% ============================================================",
        f"% Neural Demand — {label} (auto-generated)",
        f"% N_RUNS = {n_runs}",
        r"% ============================================================", "",
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{tab:{label}}}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    \textbf{Model} & "
        r"\textbf{$\epsilon_{00}$ (ASP)} & "
        r"\textbf{$\epsilon_{11}$ (ACET)} & "
        r"\textbf{$\epsilon_{22}$ (IBU)} \\",
        r"    \midrule",
    ]
    for row in elast_rows:
        nm = row["Model"]
        e0 = _c(row.get("eps_0"), row.get("eps_0_se"))
        e1 = _c(row.get("eps_1"), row.get("eps_1_se"))
        e2 = _c(row.get("eps_2"), row.get("eps_2_se"))
        lines.append(f"    {nm} & {e0} & {e1} & {e2} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    tex_path = f"{out_dir}/{label}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {tex_path}")
