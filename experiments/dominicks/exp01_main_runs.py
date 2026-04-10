"""
experiments/dominicks/exp01_main_runs.py
=========================================
Section 9 of dominicks_multiple_runs.py.

One full training + evaluation pass for the Dominick's analgesics experiment.
Encapsulates the ``run_once(seed, splits, cfg)`` function and aggregation helpers.

All global arrays are passed via the ``splits`` dict returned by
``experiments.dominicks.data.load()``.
"""

from __future__ import annotations

import numpy as np
import torch

from src.models.dominicks import (
    LAAIDS,
    BLPLogitIV,
    QUAIDS,
    SeriesDemand,
    StaticND, StaticND_FE,
    HabitND, HabitND_FE,
    WindowND,
    _train,
    build_window_features,
    compute_xbar_e2e,
    cf_first_stage,
    train_window_irl,
    feat_good_specific,
    feat_orth,
    feat_shared,
    run_lirl,
)
from experiments.dominicks.data import G, GOODS
from experiments.dominicks.utils import (
    pred,
    own_elasticity,
    full_elasticity_matrix,
    comp_var,
    get_metrics,
    kl_div,
    mdp_price_cond_habit,
    fit_nd_delta_grid_dom,
    fit_mdp_delta_grid_dom,
    dm_test_by_store,
)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL NAME LIST  (same order as tables / figures in paper)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAMES = [
    'LA-AIDS', 'Logit-IV', 'QUAIDS', 'Series Est.', 'Neural Demand (window)',
    'Linear Demand (Shared)', 'Linear Demand (GoodSpec)', 'Linear Demand (Orth)',
    'Neural Demand (static)', 'Neural Demand (habit)',
    # Store-FE variants
    'Neural Demand (FE)', 'Neural Demand (habit, FE)',
    # Control-function (CF) variants
    'Neural Demand (CF)', 'Neural Demand (habit, CF)', 'Neural Demand (habit, FE, CF)',
    # Placebo
    'PlaceboND',
]


# ─────────────────────────────────────────────────────────────────────────────
#  run_once
# ─────────────────────────────────────────────────────────────────────────────

def run_once(seed: int, splits: dict, cfg: dict) -> dict:
    """Re-estimate all stochastic models with *seed* and return results dict.

    Parameters
    ----------
    seed   : RNG seed for PyTorch and NumPy.
    splits : dict returned by ``experiments.dominicks.data.load()``.
             Must contain all train/test arrays, grids, and store encodings.
    cfg    : configuration dict (same as ``CFG`` in the original script).

    Returns
    -------
    dict with keys: perf, elast, welf, cross_elast, mdp_structural,
        welf_by_pct, r_*, kl_*, curves, curves_by_shock, delta_*,
        hist_*, KW, SPECS, …
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Unpack splits ──────────────────────────────────────────────────────
    p_tr  = splits['p_tr'];  p_te  = splits['p_te']
    w_tr  = splits['w_tr'];  w_te  = splits['w_te']
    mw_tr = splits['mw_tr']; mw_te = splits['mw_te']
    y_tr  = splits['y_tr'];  y_te  = splits['y_te']
    xb_tr = splits['xb_tr']; xb_te = splits['xb_te']
    qp_tr = splits['qp_tr']; qp_te = splits['qp_te']
    ls_tr = splits['ls_tr']; ls_te = splits['ls_te']
    s_tr  = splits['s_tr'];  s_te  = splits['s_te']
    Z_tr  = splits['Z_tr']

    N_STORES      = splits['N_STORES']
    STORE_EMB_DIM = splits['STORE_EMB_DIM']
    s_tr_idx      = splits['s_tr_idx']
    s_te_idx      = splits['s_te_idx']
    s_te_mode_idx = splits['s_te_mode_idx']

    tpx_all = splits['tpx_all']
    pgr_all = splits['pgr_all']
    tpx     = splits['tpx']
    pgr     = splits['pgr']
    fy      = splits['fy']
    p_mn    = splits['p_mn']
    y_mn    = splits['y_mn']
    xb_mn   = splits['xb_mn']
    qp_mn   = splits['qp_mn']
    p0w     = splits['p0w']
    p1w     = splits['p1w']

    sg   = cfg['shock_good']
    ss   = cfg['shock_pct']
    N_GR = len(pgr)
    dev  = cfg['device']

    # ── Train models ──────────────────────────────────────────────────────
    aids_m   = LAAIDS().fit(p_tr, w_tr, y_tr)
    blp_m    = BLPLogitIV().fit(p_tr, mw_tr, Z_tr, y_tr)
    quaids_m = QUAIDS().fit(p_tr, w_tr, y_tr)
    series_m = SeriesDemand().fit(p_tr, w_tr, y_tr)

    th_sh = run_lirl(feat_shared,        p_tr, y_tr, w_tr, cfg)
    th_gs = run_lirl(feat_good_specific, p_tr, y_tr, w_tr, cfg)
    th_or = run_lirl(feat_orth,          p_tr, y_tr, w_tr, cfg)

    nirl_m, hist_n = _train(StaticND(cfg['nirl_hidden']),
                             p_tr, y_tr, w_tr, 'nirl', cfg,
                            tag=f'StaticND s={seed}')

    sw_m = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=False,
        store_ids_tr=s_tr, store_ids_val=s_te,
        hidden=cfg.get('mdp_hidden', cfg.get('mdp_e2e_hidden', cfg['nirl_hidden'])),
        tag=f'ND-Habit s={seed}')
    mdp_m = sw_m['best_model']
    hist_m = sw_m['best_hist']
    delta_mdp = float(sw_m['delta_hat'])

    # ── Window IRL ────────────────────────────────────────────────────────
    _WIRL_W   = 4
    _lp_tr    = np.log(np.maximum(p_tr, 1e-8))
    _ly_tr    = np.log(np.maximum(y_tr, 1e-8))
    _q_tr     = w_tr * y_tr[:, None] / np.maximum(p_tr, 1e-8)
    _lq_tr    = np.log(np.maximum(_q_tr, 1e-6))
    _wf_tr    = build_window_features(_lp_tr, _ly_tr, _lq_tr,
                                      window=_WIRL_W, store_ids=s_tr)
    
    # Build test features for evaluation
    _lp_te    = np.log(np.maximum(p_te, 1e-8))
    _ly_te    = np.log(np.maximum(y_te, 1e-8))
    _q_te     = w_te * y_te[:, None] / np.maximum(p_te, 1e-8)
    _lq_te    = np.log(np.maximum(_q_te, 1e-6))
    _wf_te    = build_window_features(_lp_te, _ly_te, _lq_te,
                                      window=_WIRL_W, store_ids=s_te)

    wirl_m, hist_wirl = train_window_irl(
        WindowND(n_goods=G, hidden_dim=cfg['nirl_hidden'], window=_WIRL_W),
        _wf_tr, w_tr,
        epochs=cfg['nirl_epochs'], lr=cfg['nirl_lr'],
        batch_size=cfg['nirl_batch'],
        lam_mono=cfg['nirl_lam_mono'], lam_slut=cfg['nirl_lam_slut'],
        slut_start_frac=cfg['nirl_slut_start'],
        device=dev, verbose=bool(cfg.get("verbose", False)), tag=f'WindowND s={seed}',
        cache_dir=cfg.get('model_cache_dir'))
    _wirl_lp_mean = _lp_tr.mean(0)
    _wirl_lq_mean = _lq_tr.mean(0)

    # ── Store-FE model variants ────────────────────────────────────────────
    nirl_fe_m, hist_nf = _train(
        StaticND_FE(cfg['nirl_hidden'], n_stores=N_STORES, emb_dim=STORE_EMB_DIM),
        p_tr, y_tr, w_tr, 'nirl', cfg,
        store_idx_tr=s_tr_idx,
        tag=f'StaticND-FE s={seed}')

    sw_mf = fit_nd_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg, with_fe=True,
        n_stores=N_STORES, store_emb_dim=STORE_EMB_DIM,
        store_ids_tr=s_tr, store_ids_val=s_te,
        store_idx_tr=s_tr_idx, store_idx_val=s_te_idx,
        hidden=cfg.get('mdp_hidden', cfg.get('mdp_e2e_hidden', cfg['nirl_hidden'])),
        tag=f'ND-Habit-FE s={seed}')
    mdp_fe_m = sw_mf['best_model']
    hist_mf = sw_mf['best_hist']
    delta_mdp_fe = float(sw_mf['delta_hat'])

    # ── Control-Function (CF) endogeneity correction ──────────────────────
    _log_p_tr = np.log(np.maximum(p_tr, 1e-8))
    v_hat_tr, _cf_rsq = cf_first_stage(_log_p_tr, Z_tr)
    print(f'   CF first-stage R²: {_cf_rsq.round(3)}')

    nirl_cf_m, hist_ncf = _train(
        StaticND(cfg['nirl_hidden'], n_cf=G),
        p_tr, y_tr, w_tr, 'nirl', cfg,
        v_hat_tr=v_hat_tr,
        tag=f'StaticND-CF s={seed}')

    sw_mcf = fit_mdp_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg,
        v_hat_tr=v_hat_tr,
        store_ids_tr=s_tr, store_ids_val=s_te,
        hidden=cfg.get('mdp_hidden', cfg.get('mdp_e2e_hidden', cfg['nirl_hidden'])),
        model_class=HabitND,
        pred_spec='mdp-cf',
        pred_model_key='mdp_cf',
        extra_model_kw={'n_cf': G},
        tag=f'ND-Habit-CF s={seed}')
    mdp_cf_m = sw_mcf['best_model']
    hist_mcf = sw_mcf['best_hist']
    delta_mdp_cf = float(sw_mcf['delta_hat'])

    sw_mfcf = fit_mdp_delta_grid_dom(
        p_tr, y_tr, w_tr, ls_tr,
        p_te, y_te, w_te, ls_te,
        cfg,
        v_hat_tr=v_hat_tr,
        store_ids_tr=s_tr, store_ids_val=s_te,
        store_idx_tr=s_tr_idx, store_idx_val=s_te_idx,
        hidden=cfg.get('mdp_hidden', cfg.get('mdp_e2e_hidden', cfg['nirl_hidden'])),
        model_class=HabitND_FE,
        pred_spec='mdp-fe-cf',
        pred_model_key='mdp_fe_cf',
        extra_model_kw={'n_stores': N_STORES, 'emb_dim': STORE_EMB_DIM, 'n_cf': G},
        tag=f'ND-Habit-FE-CF s={seed}')
    mdp_fe_cf_m = sw_mfcf['best_model']
    hist_mfcf = sw_mfcf['best_hist']
    delta_mdp_fe_cf = float(sw_mfcf['delta_hat'])

    # ── Placebo Habit ─────────────────────────────────────────────────────
    perm_idx = np.random.permutation(len(xb_tr))
    xb_placebo_tr = xb_tr[perm_idx]
    qp_placebo_tr = qp_tr[perm_idx]
    placebo_nd_m, hist_placebo_nd = _train(HabitND(cfg['mdp_hidden']),
                              p_tr, y_tr, w_tr, 'mdp', cfg,
                              xb_prev_tr=xb_placebo_tr,
                              q_prev_tr=qp_placebo_tr,
                              tag=f'PlaceboND s={seed}')

    _wirl_kw = dict(
        wirl=wirl_m,
        wirl_log_p_hist=_wirl_lp_mean,
        wirl_log_q_hist=_wirl_lq_mean,
        wirl_window=_WIRL_W,
        wirl_features=_wf_te,
    )
    KW = dict(
        aids=aids_m, blp=blp_m, quaids=quaids_m, series=series_m,
        nirl=nirl_m, mdp=mdp_m, placebo_nd=placebo_nd_m,
        nirl_fe=nirl_fe_m, mdp_fe=mdp_fe_m,
        nirl_cf=nirl_cf_m, mdp_cf=mdp_cf_m, mdp_fe_cf=mdp_fe_cf_m,
        ff=feat_shared, theta=th_sh,
        **_wirl_kw,
    )

    ls_te_t = torch.tensor(ls_te, dtype=torch.float32, device=dev)
    with torch.no_grad():
        xb_mdp_te = compute_xbar_e2e(torch.tensor(delta_mdp, dtype=torch.float32, device=dev),
                                     ls_te_t, store_ids=s_te).cpu().numpy()
        xb_mdp_fe_te = compute_xbar_e2e(torch.tensor(delta_mdp_fe, dtype=torch.float32, device=dev),
                                        ls_te_t, store_ids=s_te).cpu().numpy()
        xb_mdp_cf_te = compute_xbar_e2e(torch.tensor(delta_mdp_cf, dtype=torch.float32, device=dev),
                                        ls_te_t, store_ids=s_te).cpu().numpy()
        xb_mdp_fe_cf_te = compute_xbar_e2e(torch.tensor(delta_mdp_fe_cf, dtype=torch.float32, device=dev),
                                           ls_te_t, store_ids=s_te).cpu().numpy()

    _mdp_te = (xb_mdp_te, qp_te)
    # Placebo test set: shuffle test habit stock too?
    # Or use the shuffled train habit stock if we want to test "random habit"?
    # The hypothesis is that the *structure* of habit matters.
    # If we feed random noise as habit, it shouldn't help.
    # We can just permute xb_te.
    perm_idx_te = np.random.permutation(len(xb_te))
    _placebo_nd_te = (xb_te[perm_idx_te], qp_te[perm_idx_te])

    SPECS = [
        ('LA-AIDS',          'aids',       {},                                            None),
        ('Logit-IV',         'blp',        {},                                            None),
        ('QUAIDS',           'quaids',     {},                                            None),
        ('Series Est.',      'series',     {},                                            None),
        ('Neural Demand (window)',  'window-irl', {},                                            None),
        ('Linear Demand (Shared)',           'lirl',       {'ff': feat_shared,        'theta': th_sh},   None),
        ('Linear Demand (GoodSpec)',         'lirl',       {'ff': feat_good_specific, 'theta': th_gs},   None),
        ('Linear Demand (Orth)',             'lirl',       {'ff': feat_orth,          'theta': th_or},   None),
        ('Neural Demand (static)', 'nirl',       {},                                            None),
        ('Neural Demand (habit)',  'mdp',        {},                                            _mdp_te),
        ('PlaceboND', 'mdp',       {'mdp': placebo_nd_m},                        _placebo_nd_te),
    ]

    # ── Table 1: accuracy ─────────────────────────────────────────────────
    def _xbt_kw(xbt):
        if xbt is None:
            return {}
        xbp, qp = xbt
        return {'xb_prev': xbp, 'q_prev': qp}

    perf = {}
    for nm, sp, ek, xbt in SPECS:
        perf[nm] = get_metrics(sp, p_te, y_te, w_te, cfg,
                               **_xbt_kw(xbt), **{**KW, **ek})
    for _fe_nm, _fe_sp, _fe_xkw in [
        ('Neural Demand (FE)',          'nirl-fe',    {}),
        ('Neural Demand (habit, FE)',   'mdp-fe',     {'xb_prev': xb_mdp_fe_te, 'q_prev': qp_te}),
    ]:
        try:
            perf[_fe_nm] = get_metrics(_fe_sp, p_te, y_te, w_te, cfg,
                                       store_idx=s_te_idx, **{**KW, **_fe_xkw},
                                       s_te_mode_idx=s_te_mode_idx)
        except Exception:
            perf[_fe_nm] = {'RMSE': np.nan, 'MAE': np.nan}
    for _cf_nm, _cf_sp, _cf_xkw in [
        ('Neural Demand (CF)',          'nirl-cf',   {}),
        ('Neural Demand (habit, CF)',   'mdp-cf',    {'xb_prev': xb_mdp_cf_te, 'q_prev': qp_te}),
        ('Neural Demand (habit, FE, CF)', 'mdp-fe-cf', {'xb_prev': xb_mdp_fe_cf_te, 'q_prev': qp_te}),
    ]:
        try:
            _si_kw = ({'store_idx': s_te_idx, 's_te_mode_idx': s_te_mode_idx}
                      if _cf_sp == 'mdp-fe-cf' else {})
            perf[_cf_nm] = get_metrics(_cf_sp, p_te, y_te, w_te, cfg,
                                       **{**KW, **_cf_xkw, **_si_kw})
        except Exception:
            perf[_cf_nm] = {'RMSE': np.nan, 'MAE': np.nan}

    # ── Table 2: elasticities ─────────────────────────────────────────────
    elast = {}
    for nm, sp, ek, xbt in SPECS:
        try:
            mdp_kw = ({'xb_prev0': xb_mn, 'q_prev0': qp_mn}
                      if xbt is not None else {})
            elast[nm] = own_elasticity(sp, p_mn, y_mn, cfg,
                                        **mdp_kw, **{**KW, **ek},
                                        s_te_mode_idx=s_te_mode_idx)
        except Exception:
            elast[nm] = np.full(G, np.nan)
    for _fe_nm, _fe_sp, _fe_xkw in [
        ('Neural Demand (FE)',          'nirl-fe',    {}),
        ('Neural Demand (habit, FE)',   'mdp-fe',     {'xb_prev0': xb_mdp_fe_te.mean(0), 'q_prev0': qp_mn}),
    ]:
        try:
            elast[_fe_nm] = own_elasticity(_fe_sp, p_mn, y_mn, cfg,
                                           **{**KW, **_fe_xkw},
                                           s_te_mode_idx=s_te_mode_idx)
        except Exception:
            elast[_fe_nm] = np.full(G, np.nan)
    for _cf_nm, _cf_sp, _cf_xkw in [
        ('Neural Demand (CF)',            'nirl-cf',   {}),
        ('Neural Demand (habit, CF)',     'mdp-cf',    {'xb_prev0': xb_mdp_cf_te.mean(0), 'q_prev0': qp_mn}),
        ('Neural Demand (habit, FE, CF)', 'mdp-fe-cf', {'xb_prev0': xb_mdp_fe_cf_te.mean(0), 'q_prev0': qp_mn}),
    ]:
        try:
            elast[_cf_nm] = own_elasticity(_cf_sp, p_mn, y_mn, cfg,
                                           **{**KW, **_cf_xkw},
                                           s_te_mode_idx=s_te_mode_idx)
        except Exception:
            elast[_cf_nm] = np.full(G, np.nan)

    # ── Cross-price elasticity matrices ───────────────────────────────────
    _cp_specs = [
        ('LA-AIDS',       'aids',   {},   None),
        ('Logit-IV',      'blp',    {},   None),
        ('QUAIDS',        'quaids', {},   None),
        ('Neural Demand (static)', 'nirl',   {},   None),
        ('Neural Demand (habit)',  'mdp',    {},   (xb_mn, qp_mn)),
    ]
    cross_elast = {}
    for nm, sp, ek, xbt in _cp_specs:
        try:
            _mdp_kw = ({'xb_prev0': xbt[0], 'q_prev0': xbt[1]}
                       if xbt is not None else {})
            cross_elast[nm] = full_elasticity_matrix(
                sp, p_mn, y_mn, cfg, **_mdp_kw, **{**KW, **ek},
                s_te_mode_idx=s_te_mode_idx)
        except Exception:
            cross_elast[nm] = np.full((G, G), np.nan)
    for _fe_nm, _fe_sp, _fe_xkw in [
        ('Neural Demand (FE)',          'nirl-fe',    {}),
        ('Neural Demand (habit, FE)',   'mdp-fe',     {'xb_prev0': xb_mdp_fe_te.mean(0), 'q_prev0': qp_mn}),
    ]:
        try:
            cross_elast[_fe_nm] = full_elasticity_matrix(
                _fe_sp, p_mn, y_mn, cfg, **{**KW, **_fe_xkw},
                s_te_mode_idx=s_te_mode_idx)
        except Exception:
            cross_elast[_fe_nm] = np.full((G, G), np.nan)

    for _cf_nm, _cf_sp, _cf_xkw in [
        ('Neural Demand (CF)',            'nirl-cf',   {}),
        ('Neural Demand (habit, CF)',     'mdp-cf',    {'xb_prev0': xb_mdp_cf_te.mean(0), 'q_prev0': qp_mn}),
        ('Neural Demand (habit, FE, CF)', 'mdp-fe-cf', {'xb_prev0': xb_mdp_fe_cf_te.mean(0), 'q_prev0': qp_mn}),
    ]:
        try:
            cross_elast[_cf_nm] = full_elasticity_matrix(
                _cf_sp, p_mn, y_mn, cfg, **{**KW, **_cf_xkw},
                s_te_mode_idx=s_te_mode_idx)
        except Exception:
            cross_elast[_cf_nm] = np.full((G, G), np.nan)

    # ── MDP structural demand curve (fixed mean xbar) ─────────────────────
    _xb_fixed = np.tile(xb_mdp_te.mean(0), (N_GR, 1))
    _qp_fixed = np.tile(qp_mn, (N_GR, 1))
    try:
        _mdp_structural = pred('mdp', tpx, fy, cfg,
                               xb_prev=_xb_fixed, q_prev=_qp_fixed, **KW)
    except Exception:
        _mdp_structural = np.full((N_GR, G), np.nan)

    # ── Table 3: welfare ──────────────────────────────────────────────────
    welf = {}     # Default (cfg shock_good)
    welf_all = {} # All goods
    
    for g_idx in range(G):
        p1w_g = p_mn.copy()
        p1w_g[g_idx] *= 1 + ss
        welf_all[g_idx] = {}
        
        # Helper to run welfare for a specific shock
        def _run_w(nm, sp, kw_args):
            try:
                return comp_var(sp, p0w, p1w_g, y_mn, cfg, **kw_args)
            except Exception:
                return np.nan

        for nm, sp, ek, xbt in SPECS:
            mdp_kw = ({'xb_prev0': xb_mn, 'q_prev0': qp_mn} if xbt is not None else {})
            welf_all[g_idx][nm] = _run_w(nm, sp, {**mdp_kw, **KW, **ek})
            
        for _fe_nm, _fe_sp, _fe_xkw in [
            ('Neural Demand (FE)',          'nirl-fe',    {}),
            ('Neural Demand (habit, FE)',   'mdp-fe',     {'xb_prev0': xb_mdp_fe_te.mean(0), 'q_prev0': qp_mn}),
        ]:
            welf_all[g_idx][_fe_nm] = _run_w(_fe_nm, _fe_sp, {**KW, **_fe_xkw})
            
        for _cf_nm, _cf_sp, _cf_xkw in [
            ('Neural Demand (CF)',            'nirl-cf',   {}),
            ('Neural Demand (habit, CF)',     'mdp-cf',    {'xb_prev0': xb_mdp_cf_te.mean(0), 'q_prev0': qp_mn}),
            ('Neural Demand (habit, FE, CF)', 'mdp-fe-cf', {'xb_prev0': xb_mdp_fe_cf_te.mean(0), 'q_prev0': qp_mn}),
        ]:
            welf_all[g_idx][_cf_nm] = _run_w(_cf_nm, _cf_sp, {**KW, **_cf_xkw})

    welf = welf_all[sg] # Default for backward compatibility

    # ── Welfare across xbar percentiles ───────────────────────────────────
    _D_PCTS  = [10, 25, 50, 75, 90]
    _xb_pcts = np.percentile(xb_tr, _D_PCTS, axis=0)
    _qp_pcts = np.percentile(qp_tr, _D_PCTS, axis=0)
    welf_by_pct = {}
    for _pi, _pct in enumerate(_D_PCTS):
        _xb_pt = _xb_pcts[_pi]
        _qp_pt = _qp_pcts[_pi]
        welf_by_pct[_pct] = {}
        for _cf_nm, _cf_sp, _cf_xkw in [
            ('Neural Demand (static)',        'nirl',     {}),
            ('Neural Demand (habit)',         'mdp',      {'xb_prev0': _xb_pt, 'q_prev0': _qp_pt}),
            ('Neural Demand (CF)',            'nirl-cf',  {}),
            ('Neural Demand (habit, CF)',     'mdp-cf',   {'xb_prev0': _xb_pt, 'q_prev0': _qp_pt}),
            ('Neural Demand (habit, FE, CF)', 'mdp-fe-cf',
             {'xb_prev0': _xb_pt, 'q_prev0': _qp_pt}),
        ]:
            try:
                welf_by_pct[_pct][_cf_nm] = comp_var(
                    _cf_sp, p0w, p1w, y_mn, cfg, **{**KW, **_cf_xkw})
            except Exception:
                welf_by_pct[_pct][_cf_nm] = np.nan

    # ── Table 4: MDP advantage ────────────────────────────────────────────
    r_a    = perf['LA-AIDS']['RMSE']
    r_blp  = perf['Logit-IV']['RMSE']
    r_q    = perf['QUAIDS']['RMSE']
    r_s    = perf['Series Est.']['RMSE']
    r_wirl = perf['Neural Demand (window)']['RMSE']
    r_n    = perf['Neural Demand (static)']['RMSE']
    r_m    = perf['Neural Demand (habit)']['RMSE']
    r_nf   = perf['Neural Demand (FE)']['RMSE']
    r_mf   = perf['Neural Demand (habit, FE)']['RMSE']

    # Diebold-Mariano Test (Static vs Habit)
    # Re-predict to get residuals (perf only has RMSE)
    try:
        _wp_n = pred('nirl', p_te, y_te, cfg, **KW)
        _wp_m = pred('mdp', p_te, y_te, cfg, xb_prev=xb_te, q_prev=qp_te, **KW)
        _res_n = w_te - _wp_n
        _res_m = w_te - _wp_m
        dm_stat, dm_p, dm_diff = dm_test_by_store(_res_n, _res_m, s_te)
    except Exception:
        dm_stat, dm_p, dm_diff = np.nan, np.nan, np.nan

    kl_a    = kl_div('aids',       p_te, y_te, w_te, cfg, **KW)
    kl_blp  = kl_div('blp',        p_te, y_te, w_te, cfg, **KW)
    kl_q    = kl_div('quaids',     p_te, y_te, w_te, cfg, **KW)
    kl_s    = kl_div('series',     p_te, y_te, w_te, cfg, **KW)
    kl_wirl = kl_div('window-irl', p_te, y_te, w_te, cfg, **KW)
    kl_n    = kl_div('nirl',       p_te, y_te, w_te, cfg, **KW)
    kl_m    = kl_div('mdp',        p_te, y_te, w_te, cfg,
                     xb_prev=xb_te, q_prev=qp_te, **KW)
    try:
        kl_nf = kl_div('nirl-fe', p_te, y_te, w_te, cfg,
                        store_idx=s_te_idx, s_te_mode_idx=s_te_mode_idx, **KW)
    except Exception:
        kl_nf = np.nan
    try:
        kl_mf = kl_div('mdp-fe', p_te, y_te, w_te, cfg,
                        xb_prev=xb_te, q_prev=qp_te, store_idx=s_te_idx,
                        s_te_mode_idx=s_te_mode_idx, **KW)
    except Exception:
        kl_mf = np.nan
    try:
        kl_ncf = kl_div('nirl-cf', p_te, y_te, w_te, cfg, **KW)
    except Exception:
        kl_ncf = np.nan
    try:
        kl_mcf = kl_div('mdp-cf', p_te, y_te, w_te, cfg,
                         xb_prev=xb_te, q_prev=qp_te, **KW)
    except Exception:
        kl_mcf = np.nan
    try:
        kl_mp = kl_div('mdp', p_te, y_te, w_te, cfg,
                       xb_prev=_placebo_nd_te[0], q_prev=_placebo_nd_te[1], **{**KW, 'mdp': placebo_nd_m})
    except Exception:
        kl_mp = np.nan

    # ── Demand curves ─────────────────────────────────────────────────────
    _xbr_sg, _qpr_sg = mdp_price_cond_habit(pgr, sg, p_te, xb_te, qp_te)
    _xb_struct_sg = np.tile(xb_mn, (N_GR, 1))
    _qp_struct_sg = np.tile(qp_mn, (N_GR, 1))

    curves = {}
    _curve_specs_full = [
        ('aids',       {},                                    None,                   'LA-AIDS'),
        ('blp',        {},                                    None,                   'Logit-IV'),
        ('quaids',     {},                                    None,                   'QUAIDS'),
        ('series',     {},                                    None,                   'Series Est.'),
        ('window-irl', {},                                    None,                   'Neural Demand (window)'),
        ('lirl',       {'ff': feat_shared, 'theta': th_sh},  None,                   'Linear Demand (Shared)'),
        ('lirl',       {'ff': feat_orth,   'theta': th_or},  None,                   'Linear Demand (Orth)'),
        ('nirl',       {},                                    None,                   'Neural Demand (static)'),
        ('mdp',        {},                                    (_xbr_sg, _qpr_sg),     'Neural Demand (habit)'),
        ('mdp',        {},  (_xb_struct_sg, _qp_struct_sg),                          'Neural Demand (habit, struct)'),
        ('mdp',        {'mdp': placebo_nd_m},  (_placebo_nd_te[0], _placebo_nd_te[1]), 'PlaceboND'),
    ]
    for sp, ek, xbt, lbl in _curve_specs_full:
        try:
            curves[lbl] = pred(sp, tpx, fy, cfg,
                               **_xbt_kw(xbt), **{**KW, **ek},
                               s_te_mode_idx=s_te_mode_idx)
        except Exception:
            curves[lbl] = np.full((len(pgr), G), np.nan)
    try:
        curves['Neural Demand (FE)'] = pred(
            'nirl-fe', tpx, fy, cfg, **KW, s_te_mode_idx=s_te_mode_idx)
    except Exception:
        curves['Neural Demand (FE)'] = np.full((len(pgr), G), np.nan)
    try:
        curves['Neural Demand (habit, FE)'] = pred(
            'mdp-fe', tpx, fy, cfg,
            xb_prev=_xb_struct_sg, q_prev=_qp_struct_sg, **KW,
            s_te_mode_idx=s_te_mode_idx)
    except Exception:
        curves['Neural Demand (habit, FE)'] = np.full((len(pgr), G), np.nan)
    try:
        curves['Neural Demand (CF)'] = pred(
            'nirl-cf', tpx, fy, cfg, **KW, s_te_mode_idx=s_te_mode_idx)
    except Exception:
        curves['Neural Demand (CF)'] = np.full((len(pgr), G), np.nan)
    try:
        curves['Neural Demand (habit, CF)'] = pred(
            'mdp-cf', tpx, fy, cfg,
            xb_prev=_xb_struct_sg, q_prev=_qp_struct_sg, **KW,
            s_te_mode_idx=s_te_mode_idx)
    except Exception:
        curves['Neural Demand (habit, CF)'] = np.full((len(pgr), G), np.nan)

    curves_by_shock = {}
    _fy_g = np.full(N_GR, float(y_te.mean()))
    _xb_struct_g = np.tile(xb_mn, (N_GR, 1))
    _qp_struct_g = np.tile(qp_mn, (N_GR, 1))
    for _g in range(G):
        _pgr_g = pgr_all[_g]
        _tpx_g = tpx_all[_g]
        _xbr_g, _qpr_g = mdp_price_cond_habit(_pgr_g, _g, p_te, xb_te, qp_te)
        _cs = [
            ('aids',       {},                                  None,                    'LA-AIDS'),
            ('blp',        {},                                  None,                    'Logit-IV'),
            ('quaids',     {},                                  None,                    'QUAIDS'),
            ('series',     {},                                  None,                    'Series Est.'),
            ('window-irl', {},                                  None,                    'Neural Demand (window)'),
            ('lirl',       {'ff': feat_orth, 'theta': th_or},  None,                    'Linear Demand (Orth)'),
            ('nirl',       {},                                  None,                    'Neural Demand (static)'),
            ('mdp',        {},  (_xb_struct_g, _qp_struct_g),                           'Neural Demand (habit, struct)'),
            ('mdp',        {},  (_xbr_g, _qpr_g),                                       'Neural Demand (habit)'),
            ('mdp',        {'mdp': placebo_nd_m},  (_placebo_nd_te[0], _placebo_nd_te[1]), 'PlaceboND'),
        ]
        _cv = {}
        for sp, ek, xbt, lbl in _cs:
            try:
                _cv[lbl] = pred(sp, _tpx_g, _fy_g, cfg,
                                **_xbt_kw(xbt), **{**KW, **ek},
                                s_te_mode_idx=s_te_mode_idx)
            except Exception:
                _cv[lbl] = np.full((N_GR, G), np.nan)
        curves_by_shock[_g] = _cv

    # ── δ values ──────────────────────────────────────────────────────────
    delta_mdp    = float(delta_mdp)
    delta_mdp_fe = float(delta_mdp_fe)

    # ── CF scalar shortcuts for Table 4 ───────────────────────────────────
    r_ncf  = perf['Neural Demand (CF)']['RMSE']
    r_mcf  = perf['Neural Demand (habit, CF)']['RMSE']

    return dict(
        perf=perf, elast=elast, welf=welf, welf_all=welf_all,
        cross_elast=cross_elast,
        mdp_structural=_mdp_structural,
        welf_by_pct=welf_by_pct,
        r_a=r_a, r_blp=r_blp, r_q=r_q, r_s=r_s, r_wirl=r_wirl,
        r_n=r_n, r_m=r_m,
        r_ncf=r_ncf, r_mcf=r_mcf,
        r_nf=r_nf, r_mf=r_mf,
        kl_a=kl_a, kl_blp=kl_blp, kl_q=kl_q, kl_s=kl_s, kl_wirl=kl_wirl,
        kl_n=kl_n, kl_m=kl_m,
        kl_ncf=kl_ncf, kl_mcf=kl_mcf,
        kl_nf=kl_nf, kl_mf=kl_mf,
        kl_mp=kl_mp,
        dm_stat=dm_stat, dm_p=dm_p, dm_diff=dm_diff,
        curves=curves,
        curves_by_shock=curves_by_shock,
        delta_mdp=delta_mdp,
        delta_mdp_fe=delta_mdp_fe,
        cf_rsq=_cf_rsq,
        hist_n=hist_n, hist_m=hist_m,
        hist_nf=hist_nf, hist_mf=hist_mf,
        hist_ncf=hist_ncf, hist_mcf=hist_mcf,
        hist_placebo_nd=hist_placebo_nd,
        hist_wirl=hist_wirl,
        nirl_m=nirl_m, mdp_m=mdp_m,
        nirl_fe_m=nirl_fe_m, mdp_fe_m=mdp_fe_m,
        aids_m=aids_m, blp_m=blp_m, quaids_m=quaids_m, series_m=series_m,
        wirl_m=wirl_m,
        th_sh=th_sh, th_gs=th_gs, th_or=th_or,
        KW=KW, SPECS=SPECS,
        placebo_nd_te=_placebo_nd_te,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(all_runs: list) -> dict:
    """Aggregate a list of ``run_once`` result dicts across seeds.

    Returns a dict with mean/std for all scalar and array quantities,
    matching the original aggregation logic in Section 10.
    """
    n = len(all_runs)
    ddof = min(1, n - 1)
    last = all_runs[-1]

    def _agg_metric(metric):
        raw = {nm: [r['perf'][nm][metric] for r in all_runs] for nm in MODEL_NAMES}
        return (
            {nm: np.nanmean(v) for nm, v in raw.items()},
            {nm: np.nanstd(v, ddof=ddof) for nm, v in raw.items()},
        )

    def _agg_welf():
        raw = {nm: [r['welf'][nm] for r in all_runs] for nm in MODEL_NAMES}
        return (
            {nm: np.nanmean(v) for nm, v in raw.items()},
            {nm: np.nanstd(v, ddof=ddof) for nm, v in raw.items()},
        )

    def _agg_welf_all():
        if 'welf_all' not in all_runs[0]: return {}, {}
        G_loc = len(all_runs[0]['welf_all'])
        means = {}
        stds  = {}
        for g in range(G_loc):
            means[g] = {}
            stds[g]  = {}
            for nm in MODEL_NAMES:
                vals = [r['welf_all'][g].get(nm, np.nan) for r in all_runs]
                means[g][nm] = np.nanmean(vals)
                stds[g][nm]  = np.nanstd(vals, ddof=ddof)
        return means, stds

    def _agg_elast():
        raw = {nm: [r['elast'][nm] for r in all_runs] for nm in MODEL_NAMES}
        return (
            {nm: np.nanmean(v, axis=0) for nm, v in raw.items()},
            {nm: np.nanstd(v, axis=0, ddof=ddof) for nm, v in raw.items()},
        )

    def _agg_curves():
        labels = list(all_runs[0]['curves'].keys())
        means  = {lbl: np.nanmean(np.stack([r['curves'][lbl] for r in all_runs]), 0)
                  for lbl in labels}
        stds   = {lbl: np.nanstd(np.stack([r['curves'][lbl] for r in all_runs]), 0,
                                  ddof=ddof)
                  for lbl in labels}
        return means, stds

    def _agg_curves_by_shock():
        labels = list(all_runs[0]['curves_by_shock'][0].keys())
        G_loc  = len(all_runs[0]['curves_by_shock'])
        means  = {g: {lbl: np.nanmean(
                      np.stack([r['curves_by_shock'][g][lbl] for r in all_runs]), 0)
                      for lbl in labels}
                  for g in range(G_loc)}
        stds   = {g: {lbl: np.nanstd(
                      np.stack([r['curves_by_shock'][g][lbl] for r in all_runs]), 0,
                      ddof=ddof)
                      for lbl in labels}
                  for g in range(G_loc)}
        return means, stds

    rmse_mean, rmse_std = _agg_metric('RMSE')
    mae_mean,  mae_std  = _agg_metric('MAE')
    elast_mean, elast_std = _agg_elast()
    welf_mean, welf_std   = _agg_welf()
    welf_all_mean, welf_all_std = _agg_welf_all()
    curve_mean, curve_std = _agg_curves()
    cbs_mean,   cbs_std   = _agg_curves_by_shock()

    # Welfare by xbar percentile
    _D_PCTS_AGG  = [10, 25, 50, 75, 90]
    _wbp_models  = list(all_runs[0]['welf_by_pct'][10].keys())
    welf_pct_mean = {}
    welf_pct_std  = {}
    for _pct in _D_PCTS_AGG:
        welf_pct_mean[_pct] = {}
        welf_pct_std[_pct]  = {}
        for _nm in _wbp_models:
            vals = [r['welf_by_pct'][_pct][_nm] for r in all_runs]
            welf_pct_mean[_pct][_nm] = np.nanmean(vals)
            welf_pct_std[_pct][_nm]  = np.nanstd(vals, ddof=ddof)

    # CF R²
    cf_rsq_mean = np.stack([r['cf_rsq'] for r in all_runs], 0).mean(0)

    # Cross-price elasticity
    _cp_names = ['LA-AIDS', 'Logit-IV', 'QUAIDS',
                 'Neural Demand (static)', 'Neural Demand (habit)',
                 'Neural Demand (FE)', 'Neural Demand (habit, FE)',
                 'Neural Demand (CF)', 'Neural Demand (habit, CF)']
    cross_elast_mean = {
        nm: np.nanmean(np.stack([r['cross_elast'][nm] for r in all_runs], 0), 0)
        for nm in _cp_names if nm in all_runs[0]['cross_elast']
    }
    cross_elast_std = {
        nm: np.nanstd(np.stack([r['cross_elast'][nm] for r in all_runs], 0), 0, ddof=ddof)
        for nm in _cp_names if nm in all_runs[0]['cross_elast']
    }

    mdp_structural_mean = np.nanmean(
        np.stack([r['mdp_structural'] for r in all_runs], 0), 0)

    def _arr(key): return np.array([r[key] for r in all_runs])

    return dict(
        # ── per-model metrics ───────────────────────────────────────────
        rmse_mean=rmse_mean,  rmse_std=rmse_std,
        mae_mean=mae_mean,    mae_std=mae_std,
        elast_mean=elast_mean, elast_std=elast_std,
        welf_mean=welf_mean,  welf_std=welf_std,
        welf_all_mean=welf_all_mean, welf_all_std=welf_all_std,
        curve_mean=curve_mean, curve_std=curve_std,
        cbs_mean=cbs_mean,    cbs_std=cbs_std,
        cross_elast_mean=cross_elast_mean,
        cross_elast_std=cross_elast_std,
        mdp_structural_mean=mdp_structural_mean,
        # ── DM Test ─────────────────────────────────────────────────────
        dm_stat_mu=_arr('dm_stat').mean(), dm_stat_se=_arr('dm_stat').std(ddof=ddof),
        dm_p_mu=_arr('dm_p').mean(),       dm_p_se=_arr('dm_p').std(ddof=ddof),
        dm_diff_mu=_arr('dm_diff').mean(), dm_diff_se=_arr('dm_diff').std(ddof=ddof),
        # ── welfare by pct ───────────────────────────────────────────────
        welf_pct_mean=welf_pct_mean,
        welf_pct_std=welf_pct_std,
        # ── CF R² ───────────────────────────────────────────────────────
        cf_rsq_mean=cf_rsq_mean,
        # ── scalar means / SEs ──────────────────────────────────────────
        r_a_mu=_arr('r_a').mean(),    r_a_se=_arr('r_a').std(ddof=ddof),
        r_blp_mu=_arr('r_blp').mean(), r_blp_se=_arr('r_blp').std(ddof=ddof),
        r_q_mu=_arr('r_q').mean(),    r_q_se=_arr('r_q').std(ddof=ddof),
        r_s_mu=_arr('r_s').mean(),    r_s_se=_arr('r_s').std(ddof=ddof),
        r_wirl_mu=_arr('r_wirl').mean(), r_wirl_se=_arr('r_wirl').std(ddof=ddof),
        r_n_mu=_arr('r_n').mean(),    r_n_se=_arr('r_n').std(ddof=ddof),
        r_m_mu=_arr('r_m').mean(),    r_m_se=_arr('r_m').std(ddof=ddof),
        r_ncf_mu=np.nanmean(_arr('r_ncf')), r_ncf_se=np.nanstd(_arr('r_ncf'), ddof=ddof),
        r_mcf_mu=np.nanmean(_arr('r_mcf')), r_mcf_se=np.nanstd(_arr('r_mcf'), ddof=ddof),
        r_nf_mu=np.nanmean(_arr('r_nf')),   r_nf_se=np.nanstd(_arr('r_nf'),   ddof=ddof),
        r_mf_mu=np.nanmean(_arr('r_mf')),   r_mf_se=np.nanstd(_arr('r_mf'),   ddof=ddof),
        kl_a_mu=_arr('kl_a').mean(),    kl_a_se=_arr('kl_a').std(ddof=ddof),
        kl_blp_mu=_arr('kl_blp').mean(), kl_blp_se=_arr('kl_blp').std(ddof=ddof),
        kl_q_mu=_arr('kl_q').mean(),    kl_q_se=_arr('kl_q').std(ddof=ddof),
        kl_s_mu=_arr('kl_s').mean(),    kl_s_se=_arr('kl_s').std(ddof=ddof),
        kl_wirl_mu=_arr('kl_wirl').mean(), kl_wirl_se=_arr('kl_wirl').std(ddof=ddof),
        kl_n_mu=_arr('kl_n').mean(),    kl_n_se=_arr('kl_n').std(ddof=ddof),
        kl_m_mu=_arr('kl_m').mean(),    kl_m_se=_arr('kl_m').std(ddof=ddof),
        kl_ncf_mu=np.nanmean(_arr('kl_ncf')), kl_ncf_se=np.nanstd(_arr('kl_ncf'), ddof=ddof),
        kl_mcf_mu=np.nanmean(_arr('kl_mcf')), kl_mcf_se=np.nanstd(_arr('kl_mcf'), ddof=ddof),
        kl_nf_mu=np.nanmean(_arr('kl_nf')),   kl_nf_se=np.nanstd(_arr('kl_nf'),   ddof=ddof),
        kl_mf_mu=np.nanmean(_arr('kl_mf')),   kl_mf_se=np.nanstd(_arr('kl_mf'),   ddof=ddof),
        kl_mp_mu=np.nanmean(_arr('kl_mp')),   kl_mp_se=np.nanstd(_arr('kl_mp'),   ddof=ddof),
        # ── delta ────────────────────────────────────────────────────────
        delta_m_mu=np.nanmean(_arr('delta_mdp')), delta_m_se=np.nanstd(_arr('delta_mdp'), ddof=ddof),
        delta_mf_mu=np.nanmean(_arr('delta_mdp_fe')), delta_mf_se=np.nanstd(_arr('delta_mdp_fe'), ddof=ddof),
        # ── last run convenience refs ─────────────────────────────────────
        last=last,
        all_runs=all_runs,
        n_runs=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURES  (Sections 12)
# ─────────────────────────────────────────────────────────────────────────────

def _make_figures(agg: dict, splits: dict, cfg: dict) -> None:
    """Generate and save all Dominick's experiment figures (Fig 1–9).

    Parameters
    ----------
    agg    : dict returned by ``aggregate()``
    splits : dict returned by ``data.load()``
    cfg    : experiment config dict
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = cfg["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    N_RUNS   = agg["n_runs"]
    TEAL     = "#009688"
    last     = agg["last"]
    all_runs = agg["all_runs"]

    pgr     = splits["pgr"]
    pgr_all = splits["pgr_all"]
    p_mn    = splits["p_mn"]
    p_te    = splits["p_te"]
    w_te    = splits["w_te"]
    xb_te   = splits["xb_te"]
    qp_te   = splits["qp_te"]
    s_te    = splits["s_te"]
    s_te_idx = splits["s_te_idx"]
    sg      = cfg["shock_good"]

    curve_mean       = agg["curve_mean"]
    curve_std        = agg["curve_std"]
    cbs_mean         = agg["cbs_mean"]
    cbs_std          = agg["cbs_std"]
    rmse_mean        = agg["rmse_mean"]
    rmse_std         = agg["rmse_std"]
    cross_elast_mean = agg["cross_elast_mean"]
    mdp_structural_mean = agg["mdp_structural_mean"]

    # ── Fig 1: Demand curves — all models, with ±1-std shaded bands ─────────
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    curve_defs = [
        ("r--",  2.0, None,       "LA-AIDS"),
        ("--",   2.0, "#9C27B0",  "Logit-IV"),
        ("g-.",  2.0, None,       "QUAIDS"),
        (":",    2.0, "#FB8C00",  "Series Est."),
        ("--",   2.0, "#6D4C41",  "Neural Demand (window)"),
        ("c:",   1.8, None,       "Linear Demand (Orth)"),
        ("b-",   2.5, None,       "Neural Demand (static)"),
        ("-",    2.0, TEAL,       "Neural Demand (habit, struct)"),
        ("--",   1.8, "#E91E63",  "Neural Demand (CF)"),
        ("-.",   1.8, "#FF5722",  "Neural Demand (habit, CF)"),
        (":",    1.8, "#795548",  "PlaceboND"),
    ]
    for sty, lw, col, lbl in curve_defs:
        mu  = curve_mean.get(lbl)
        std = curve_std.get(lbl)
        if mu is None:
            continue
        kw_plot = dict(lw=lw, label=lbl, alpha=0.9)
        if col:
            kw_plot["color"] = col
        line, = ax1.plot(pgr, mu[:, 0], sty, **kw_plot)
        if N_RUNS > 1 and std is not None:
            ax1.fill_between(pgr,
                             (mu[:, 0] - std[:, 0]).clip(0),
                              mu[:, 0] + std[:, 0],
                             color=line.get_color(), alpha=0.12)
    se_note = f"  (shaded bands = ±1 SD, n={N_RUNS})" if N_RUNS > 1 else ""
    ax1.set_xlabel("Ibuprofen Unit Price ($/100 tablets)", fontsize=14)
    ax1.set_ylabel("Aspirin Budget Share $w_0$", fontsize=14)
    ax1.legend(fontsize=9, ncol=2, framealpha=0.93)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    for ext in ("pdf", "png"):
        fig1.savefig(f"{fig_dir}/fig_demand_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved: fig_demand_curves")

    # ── Fig 2: Cross-price demand matrix — 3×3 grid ──────────────────────────
    fig2, axes2 = plt.subplots(3, 3, figsize=(16, 14), sharey="row")
    _cpm_defs = [
        ("r--", 1.8, None,      "LA-AIDS",                          "LA-AIDS"),
        ("--",  1.8, "#9C27B0", "Logit-IV",                         "Logit-IV"),
        ("g-.", 1.8, None,      "QUAIDS",                           "QUAIDS"),
        (":",   1.8, "#FB8C00", "Series Est.",                      "Series Est."),
        ("--",  1.8, "#6D4C41", "Neural Demand (window)",           "Neural Demand (window)"),
        ("b-.", 2.0, None,      "Neural Demand (static)",           "Neural Demand (static)"),
        ("-",   2.5, TEAL,      r"Neural Demand (habit, fixed $\bar{x}$)", "Neural Demand (habit, struct)"),
        (":",   1.8, "#795548", "PlaceboND",          "PlaceboND"),
    ]
    _price_labels = [f"{g} Price ($/100 tab)" for g in GOODS]
    for shock_g in range(G):
        _pgr_g = pgr_all[shock_g]
        for resp_g in range(G):
            ax = axes2[shock_g, resp_g]
            for sty, lw, col, lbl_disp, lbl_key in _cpm_defs:
                mu  = cbs_mean[shock_g].get(lbl_key)
                std = cbs_std[shock_g].get(lbl_key)
                if mu is None:
                    continue
                kw_p = dict(lw=lw, label=lbl_disp, alpha=0.9)
                if col:
                    kw_p["color"] = col
                line, = ax.plot(_pgr_g, mu[:, resp_g], sty, **kw_p)
                if N_RUNS > 1 and std is not None:
                    ax.fill_between(_pgr_g,
                                    (mu[:, resp_g] - std[:, resp_g]).clip(0),
                                     mu[:, resp_g] + std[:, resp_g],
                                    color=line.get_color(), alpha=0.10)
            ax.set_xlabel(_price_labels[shock_g], fontsize=9)
            ax.set_ylabel(f"$w_{resp_g}$  ({GOODS[resp_g]})", fontsize=9)
            ax.grid(True, alpha=0.3)
    axes2[0, 0].legend(fontsize=7, loc="best", framealpha=0.85)
    for _g, ax in enumerate(axes2[:, 0]):
        ax.annotate(f"↕  {GOODS[_g]} price",
                    xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-42, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, rotation=90, color="#555")
    for _g, ax in enumerate(axes2[0, :]):
        ax.set_title(f"{GOODS[_g]} share  $w_{_g}$", fontsize=10, fontweight="bold")
    se_note2 = f"  (bands = ±1 SD, n={N_RUNS})" if N_RUNS > 1 else ""
    fig2.suptitle(
        f"Cross-Price Demand Matrix — Dominick's Analgesics{se_note2}\n"
        "Row = price that varies  ·  Column = budget share response",
        fontsize=12, fontweight="bold")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(f"{fig_dir}/fig_mdp_advantage.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved: fig_mdp_advantage  (3×3 cross-price demand matrix)")

    # ── EMA helper ────────────────────────────────────────────────────────────
    def _ema(vals, alpha=0.3):
        out = []; s = vals[0]
        for v in vals:
            s = alpha * v + (1 - alpha) * s
            out.append(s)
        return out

    # ── Fig 3: Training convergence (last run) ────────────────────────────────
    _hist_keys = [
        ("hist_n",    "b",       "Neural Demand (static)"),
        ("hist_m",    TEAL,      "Neural Demand (habit)"),
        ("hist_nf",   "#0288D1", "Neural Demand (FE)"),
        ("hist_mf",   "#00897B", "Neural Demand (habit, FE)"),
        ("hist_wirl", "#6D4C41", "Neural Demand (window)"),
        ("hist_ncf",  "#E91E63", "Neural Demand (CF)"),
        ("hist_mcf",  "#FF5722", "Neural Demand (habit, CF)"),
        ("hist_placebo_nd", "#795548", "PlaceboND"),
    ]
    fig3, axes3 = plt.subplots(4, 2, figsize=(14, 18))
    axes3_flat = axes3.ravel()
    for _idx, (hkey, ck, title) in enumerate(_hist_keys):
        ax  = axes3_flat[_idx]
        hist = last.get(hkey, [])
        if hist:
            ex     = [h["epoch"] for h in hist]
            ky     = [h["kl"]    for h in hist]
            ky_ema = _ema(ky)
            ax.plot(ex, ky, "-", lw=0.8, color=ck, alpha=0.30)
            ax.plot(ex, ky_ema, "-", lw=2.2, color=ck, label="KL Loss (EMA α=0.3)")
            ax.legend(fontsize=8)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("KL Divergence", color=ck, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{title}\n(no history available)", fontsize=11)
            ax.axis("off")
    # Turn off the unused last panel if present
    if len(_hist_keys) < len(axes3_flat):
        for _ax in axes3_flat[len(_hist_keys):]:
            _ax.axis("off")
    seed_last = 42 + (N_RUNS - 1) * 15
    fig3.suptitle(
        f"Training Convergence — Dominick's Analgesics  (last run, seed={seed_last})\n"
        "KL divergence: raw (faint) and EMA-smoothed (bold)",
        fontsize=11, fontweight="bold")
    fig3.tight_layout()
    for ext in ("pdf", "png"):
        fig3.savefig(f"{fig_dir}/fig_convergence.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved: fig_convergence")

    # ── Fig 4: Observed vs predicted scatter (last run) ──────────────────────
    try:
        from sklearn.metrics import mean_squared_error as _mse_fn
        KW_last  = last["KW"]
        th_or    = last["th_or"]
        placebo_nd_te = last["placebo_nd_te"]

        scat_defs = [
            ("LA-AIDS",              "aids",      {},                            {},                                        "#E53935"),
            ("Logit-IV",             "blp",       {},                            {},                                        "#9C27B0"),
            ("QUAIDS",               "quaids",    {},                            {},                                        "#43A047"),
            ("Series Est.",          "series",    {},                            {},                                        "#FB8C00"),
            ("N. Demand\n(window)",  "window-irl",{},                            {},                                        "#6D4C41"),
            ("LDS\n(Orth)",          "lirl",      {"ff": feat_orth, "theta": th_or}, {},                                    "#00ACC1"),
            ("N. Demand\n(static)",  "nirl",      {},                            {},                                        "#1E88E5"),
            ("N. Demand\n(habit)",   "mdp",       {},                            {"xb_prev": xb_te, "q_prev": qp_te},       TEAL),
            ("N. Demand\n(CF)",      "nirl-cf",   {},                            {},                                        "#E91E63"),
            ("N. Demand\n(habit, CF)","mdp-cf",   {},                            {"xb_prev": xb_te, "q_prev": qp_te},       "#FF5722"),
            ("N. Demand\n(FE)",      "nirl-fe",   {},                            {"store_idx": s_te_idx},                   "#0288D1"),
            ("N. Demand\n(habit, FE)","mdp-fe",   {},                            {"xb_prev": xb_te, "q_prev": qp_te,
                                                                                  "store_idx": s_te_idx},                   "#00897B"),
            ("N. Demand\n(placebo)", "mdp",       {},                            {"xb_prev": placebo_nd_te[0], "q_prev": placebo_nd_te[1]}, "#795548"),
        ]
        n_scat = len(scat_defs)
        y_te   = splits["y_te"]
        
        for mn, sp, ek, pred_kw, col in scat_defs:
            # Create a 1x3 figure for this model
            fig_m, axes_m = plt.subplots(1, G, figsize=(15, 5))
            
            try:
                wp = pred(sp, p_te, y_te, cfg, **pred_kw, **{**KW_last, **ek})
            except Exception:
                wp = np.full((len(p_te), G), np.nan)
                
            for gi, gn in enumerate(GOODS):
                ax = axes_m[gi]
                valid = ~np.isnan(wp[:, gi])
                if valid.any():
                    ax.scatter(w_te[valid, gi], wp[valid, gi],
                               alpha=0.30, s=6, color=col, rasterized=True)
                lo = 0.0
                hi = max(float(w_te[:, gi].max()),
                         float(np.nanmax(wp[:, gi])) if valid.any() else 0.0) * 1.05
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                ri = (float(np.sqrt(_mse_fn(w_te[valid, gi], wp[valid, gi])))
                      if valid.any() else float("nan"))
                ax.set_title(f"{gn}\nRMSE={ri:.4f}", fontsize=10, fontweight="bold")
                ax.set_xlabel("Observed", fontsize=9)
                ax.set_ylabel("Predicted", fontsize=9)
                ax.grid(True, alpha=0.3)
            
            clean_name = mn.replace("\n", " ").replace("(", "").replace(")", "").replace(" ", "_").replace(".", "").lower()
            # Handle "N. Demand" -> "n_demand" or similar
            # "N. Demand\n(habit)" -> "n_demand_habit"
            
            fig_m.suptitle(f"{mn.replace(chr(10), ' ')} — Observed vs Predicted", fontsize=12, fontweight="bold")
            fig_m.tight_layout()
            
            fname = f"fig_scatter_{clean_name}"
            for ext in ("pdf", "png"):
                fig_m.savefig(f"{fig_dir}/{fname}.{ext}", dpi=150, bbox_inches="tight")
            plt.close(fig_m)
            print(f"  Saved: {fname}")

    except Exception as _e4:
        print(f"  [fig_scatter split skipped: {_e4}]")

    # ── Fig 6: RMSE bar chart with error bars ─────────────────────────────────
    if N_RUNS > 1:
        fig6, ax6 = plt.subplots(figsize=(12, 5))
        xp   = np.arange(len(MODEL_NAMES))
        rmu  = np.array([rmse_mean.get(nm, float("nan")) for nm in MODEL_NAMES])
        rse  = np.array([rmse_std.get(nm,  float("nan")) for nm in MODEL_NAMES])
        clrs = plt.cm.tab10(np.linspace(0, 0.8, len(MODEL_NAMES)))
        ax6.bar(xp, rmu, yerr=rse, capsize=5, color=clrs,
                edgecolor="k", alpha=0.85, error_kw=dict(lw=1.5, ecolor="#333"))
        ax6.set_xticks(xp)
        ax6.set_xticklabels(MODEL_NAMES, rotation=25, ha="right", fontsize=9)
        ax6.set_ylabel("Out-of-Sample RMSE", fontsize=11)
        ax6.set_title(f"Out-of-Sample RMSE — Mean ± 1 SD  ({N_RUNS} runs)\n"
                      "Dominick's Analgesics", fontsize=12, fontweight="bold")
        ax6.grid(True, axis="y", alpha=0.35)
        fig6.tight_layout()
        for ext in ("pdf", "png"):
            fig6.savefig(f"{fig_dir}/fig_rmse_bars.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig6)
        print("  Saved: fig_rmse_bars")

    # ── Fig 7: Cross-price elasticity heatmaps ────────────────────────────────
    _hm_models = ["LA-AIDS", "Logit-IV", "QUAIDS", "Neural Demand (static)", "Neural Demand (habit)",
                  "Neural Demand (FE)", "Neural Demand (habit, FE)",
                  "Neural Demand (CF)", "Neural Demand (habit, CF)"]
    _hm_avail  = [nm for nm in _hm_models if nm in cross_elast_mean]
    if _hm_avail:
        n_cols = 3
        n_rows = (len(_hm_avail) + n_cols - 1) // n_cols
        fig7, axes7 = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
        axes7_flat = axes7.flatten()
        
        _vabs = max(max(np.nanmax(np.abs(cross_elast_mean[nm])) for nm in _hm_avail), 0.1)
        
        for idx, ax7 in enumerate(axes7_flat):
            if idx < len(_hm_avail):
                nm = _hm_avail[idx]
                mat = cross_elast_mean[nm]
                im  = ax7.imshow(mat, cmap="RdBu_r", vmin=-_vabs, vmax=_vabs, aspect="auto")
                for i in range(G):
                    for j in range(G):
                        v = mat[i, j]
                        ax7.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                                 color="white" if abs(v) > 0.4 * _vabs else "black",
                                 fontweight="bold")
                ax7.set_xticks(range(G)); ax7.set_yticks(range(G))
                ax7.set_xticklabels([f"$w_{{{g}}}$\n({GOODS[g][:4]}.)" for g in range(G)], fontsize=9)
                ax7.set_yticklabels([f"$p_{{{g}}}$\n({GOODS[g][:4]}.)" for g in range(G)], fontsize=9)
                ax7.set_xlabel("Response share  $w_j$", fontsize=9)
                ax7.set_ylabel("Shock price  $p_i$", fontsize=9)
                ax7.set_title(nm, fontsize=11, fontweight="bold")
                plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04,
                             label=r"$\varepsilon_{ij}$ = $\partial\log w_j/\partial\log p_i$")
            else:
                ax7.axis('off')
        
        fig7.suptitle(
            "Cross-Price Elasticity Heatmaps — Dominick's Analgesics\n"
            r"Evaluated at mean test prices  ·  MDP: fixed mean $\bar{x}$ (sorting removed)"
            "\nDiagonal = own-price  ·  Off-diagonal = cross-price",
            fontsize=10, fontweight="bold")
        fig7.tight_layout()
        for ext in ("pdf", "png"):
            fig7.savefig(f"{fig_dir}/fig_cross_elast_heatmap.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig7)
        print("  Saved: fig_cross_elast_heatmap")

    # ── Fig 8: Market segmentation + habit-sorting diagnostics ───────────────
    try:
        fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(12, 5))
        ax8a.scatter(w_te[:, 2], w_te[:, 0], alpha=0.25, s=6,
                     color="#1565C0", rasterized=True)
        _rho8 = np.corrcoef(w_te[:, 2], w_te[:, 0])[0, 1]
        ax8a.set_xlabel(f"{GOODS[2]} budget share $w_2$", fontsize=11)
        ax8a.set_ylabel(f"{GOODS[0]} budget share $w_0$", fontsize=11)
        ax8a.set_title(f"Market segmentation: {GOODS[0]} vs {GOODS[2]}\n"
                       f"Test-set store×weeks  |  ρ = {_rho8:+.3f}",
                       fontsize=11, fontweight="bold")
        ax8a.grid(True, alpha=0.3)
        _is_first_te = np.concatenate([[True], s_te[1:] != s_te[:-1]])
        _valid8 = ~_is_first_te
        _n_bins = 20
        _ibu_p  = p_te[_valid8, 2]
        _asp_xb = xb_te[_valid8, 0]
        _bins8  = np.percentile(_ibu_p, np.linspace(0, 100, _n_bins + 1))
        _bin_idx8 = np.digitize(_ibu_p, _bins8[1:-1])
        _bin_mid = np.array([_ibu_p[_bin_idx8 == b].mean()
                             for b in range(_n_bins) if (_bin_idx8 == b).sum() > 0])
        _xb_mu  = np.array([_asp_xb[_bin_idx8 == b].mean()
                            for b in range(_n_bins) if (_bin_idx8 == b).sum() > 0])
        _xb_se8 = np.array([_asp_xb[_bin_idx8 == b].std() /
                            np.sqrt((_bin_idx8 == b).sum())
                            for b in range(_n_bins) if (_bin_idx8 == b).sum() > 0])
        ax8b.scatter(_ibu_p, _asp_xb, alpha=0.12, s=5, color="#555",
                     rasterized=True, label="Individual obs.")
        ax8b.errorbar(_bin_mid, _xb_mu, yerr=_xb_se8, fmt="o-", color=TEAL,
                      ms=6, lw=2, capsize=4, label="Bin mean ± 1 SE")
        _rho_xb8 = np.corrcoef(_ibu_p, _asp_xb)[0, 1]
        ax8b.set_xlabel(f"{GOODS[2]} price $p_2$ ($/100 tab)", fontsize=11)
        ax8b.set_ylabel(r"Aspirin habit stock $\bar{x}_0$ (log-norm.)", fontsize=11)
        ax8b.set_title(f"Habit sorting: high-{GOODS[2]}-price stores have lower aspirin habit\n"
                       f"Test set (first-in-store obs. excluded)  |  ρ = {_rho_xb8:+.3f}",
                       fontsize=11, fontweight="bold")
        ax8b.legend(fontsize=9); ax8b.grid(True, alpha=0.3)
        fig8.suptitle("Why aspirin demand falls with ibuprofen price:\n"
                      "market segmentation (A) and store-level habit sorting (B)",
                      fontsize=12, fontweight="bold")
        fig8.tight_layout()
        for ext in ("pdf", "png"):
            fig8.savefig(f"{fig_dir}/fig_segmentation_sorting.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig8)
        print("  Saved: fig_segmentation_sorting")
    except Exception as _e8:
        print(f"  [fig8 skipped: {_e8}]")

    # ── Fig 9: Neural Demand decomposition — structural vs sorting ───────────
    try:
        fig9, ax9 = plt.subplots(figsize=(9, 5.5))
        _nirl_mu = curve_mean.get("Neural Demand (static)")
        _mdp_tot = curve_mean.get("Neural Demand (habit)")
        _mdp_str = mdp_structural_mean
        if _nirl_mu is not None:
            ax9.plot(pgr, _nirl_mu[:, 0], "b-", lw=2.2,
                     label="Neural Demand (static, no habit)")
        if _mdp_str is not None and not np.all(np.isnan(_mdp_str)):
            ax9.plot(pgr, _mdp_str[:, 0], "--", lw=2.2, color="#43A047",
                     label=r"Neural Demand (habit) — structural only (fixed mean $\bar{x}$, no sorting)")
        if _mdp_tot is not None:
            ax9.plot(pgr, _mdp_tot[:, 0], "-", lw=2.5, color=TEAL,
                     label=r"Neural Demand (habit) — total (price-conditional $\bar{x}$, incl. sorting)")
            if _mdp_str is not None and not np.all(np.isnan(_mdp_str)):
                ax9.fill_between(pgr, _mdp_str[:, 0], _mdp_tot[:, 0],
                                 alpha=0.22, color=TEAL, label="← Sorting contribution")
        ax9.set_xlabel(f"{GOODS[sg]} price ($/100 tab)", fontsize=12)
        ax9.set_ylabel(f"{GOODS[0]} budget share $w_{{asp}}$", fontsize=12)
        ax9.set_title(
            "Neural Demand decomposition: structural price effect vs. habit-stock sorting",
            fontsize=10, fontweight="bold")
        ax9.legend(fontsize=9, loc="best", framealpha=0.92)
        ax9.grid(True, alpha=0.3)
        fig9.tight_layout()
        for ext in ("pdf", "png"):
            fig9.savefig(f"{fig_dir}/fig_mdp_decomposition.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig9)
        print("  Saved: fig_mdp_decomposition")
    except Exception as _e9:
        print(f"  [fig9 skipped: {_e9}]")

    print(f"\n  All figures saved to: {fig_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
#  TABLES  (Sections 13–14: CSV + LaTeX)
# ─────────────────────────────────────────────────────────────────────────────

def _make_tables(agg: dict, splits: dict, cfg: dict) -> None:
    """Write CSV tables and a LaTeX file for all Dominick's results.

    Parameters
    ----------
    agg    : dict returned by ``aggregate()``
    splits : dict returned by ``data.load()``
    cfg    : experiment config dict
    """
    import os
    import pandas as pd

    out_dir = cfg["out_dir"]
    fig_dir = cfg["fig_dir"]
    os.makedirs(out_dir, exist_ok=True)

    N_RUNS     = agg["n_runs"]
    ddof       = min(1, N_RUNS - 1)
    last       = agg["last"]
    all_runs   = agg["all_runs"]
    rmse_mean  = agg["rmse_mean"]
    rmse_std   = agg["rmse_std"]
    mae_mean   = agg["mae_mean"]
    mae_std    = agg["mae_std"]
    elast_mean = agg["elast_mean"]
    elast_std  = agg["elast_std"]
    welf_mean  = agg["welf_mean"]
    welf_std   = agg["welf_std"]

    ss  = cfg["shock_pct"]
    sg  = cfg["shock_good"]

    # Scalars for Table 4
    r_a_mu    = agg["r_a_mu"];    r_a_se    = agg["r_a_se"]
    r_n_mu    = agg["r_n_mu"];    r_n_se    = agg["r_n_se"]
    r_m_mu    = agg["r_m_mu"];    r_m_se    = agg["r_m_se"]
    r_ncf_mu  = agg["r_ncf_mu"];  r_ncf_se  = agg["r_ncf_se"]
    r_mcf_mu  = agg["r_mcf_mu"];  r_mcf_se  = agg["r_mcf_se"]
    kl_a_mu   = agg["kl_a_mu"];   kl_a_se   = agg["kl_a_se"]
    kl_n_mu   = agg["kl_n_mu"];   kl_n_se   = agg["kl_n_se"]
    kl_m_mu   = agg["kl_m_mu"];   kl_m_se   = agg["kl_m_se"]
    kl_ncf_mu = agg["kl_ncf_mu"]; kl_ncf_se = agg["kl_ncf_se"]
    kl_mcf_mu = agg["kl_mcf_mu"]; kl_mcf_se = agg["kl_mcf_se"]
    kl_mp_mu  = agg["kl_mp_mu"];  kl_mp_se  = agg["kl_mp_se"]

    delta_m_mu  = agg["delta_m_mu"]
    delta_m_se  = agg["delta_m_se"]
    delta_mf_mu = agg["delta_mf_mu"]
    delta_mf_se = agg["delta_mf_se"]

    nw_mu = welf_mean.get("Neural Demand (static)", float("nan"))

    desc     = splits["desc"]
    tr       = splits["tr"]
    te       = splits["te"]
    N_STORES = splits["N_STORES"]
    weeks    = splits["weeks"]

    # ── Table 0: Descriptive stats ────────────────────────────────────────────
    desc.to_csv(f"{out_dir}/table0_desc.csv", index=False)
    print(f"  Saved: {out_dir}/table0_desc.csv")

    # ── Table 1: Accuracy ─────────────────────────────────────────────────────
    t1 = pd.DataFrame({
        "Model":     MODEL_NAMES,
        "RMSE_mean": [rmse_mean.get(nm, float("nan")) for nm in MODEL_NAMES],
        "RMSE_std":  [rmse_std.get(nm,  float("nan")) for nm in MODEL_NAMES],
        "MAE_mean":  [mae_mean.get(nm,  float("nan")) for nm in MODEL_NAMES],
        "MAE_std":   [mae_std.get(nm,   float("nan")) for nm in MODEL_NAMES],
        "n_runs":    N_RUNS,
    }).round(6)
    t1.to_csv(f"{out_dir}/table1_accuracy.csv", index=False)
    print(f"  Saved: {out_dir}/table1_accuracy.csv")

    # ── Table 2: Elasticities ─────────────────────────────────────────────────
    t2_rows = []
    for nm in MODEL_NAMES:
        row = {"Model": nm}
        for gi, gn in enumerate(GOODS):
            row[f"{gn}_mean"] = round(float(elast_mean.get(nm, np.full(G, float("nan")))[gi]), 4)
            row[f"{gn}_std"]  = round(float(elast_std.get(nm,  np.full(G, float("nan")))[gi]), 4)
        row["n_runs"] = N_RUNS
        t2_rows.append(row)
    pd.DataFrame(t2_rows).to_csv(f"{out_dir}/table2_elasticities.csv", index=False)
    print(f"  Saved: {out_dir}/table2_elasticities.csv")

    # ── Table 3: Welfare ──────────────────────────────────────────────────────
    welf_all_mean = agg.get("welf_all_mean", {})
    welf_all_std  = agg.get("welf_all_std", {})
    
    # Default (shock_good)
    t3 = pd.DataFrame({
        "Model":        MODEL_NAMES,
        "CV_Loss_mean": [welf_mean.get(nm, float("nan")) * 100.0 for nm in MODEL_NAMES],
        "CV_Loss_std":  [welf_std.get(nm,  float("nan")) * 100.0 for nm in MODEL_NAMES],
        "n_runs":       N_RUNS,
    }).round(6)
    t3.to_csv(f"{out_dir}/table3_welfare.csv", index=False)
    print(f"  Saved: {out_dir}/table3_welfare.csv")

    for g_idx in welf_all_mean:
        t3g = pd.DataFrame({
            "Model":        MODEL_NAMES,
            "CV_Loss_mean": [welf_all_mean[g_idx].get(nm, float("nan")) * 100.0 for nm in MODEL_NAMES],
            "CV_Loss_std":  [welf_all_std[g_idx].get(nm,  float("nan")) * 100.0 for nm in MODEL_NAMES],
            "n_runs":       N_RUNS,
        }).round(6)
        gn = GOODS[g_idx]
        t3g.to_csv(f"{out_dir}/table3_welfare_{gn}.csv", index=False)
        print(f"  Saved: {out_dir}/table3_welfare_{gn}.csv")

    # ── Table 4: Neural Demand habit advantage ────────────────────────────────
    def _pct(base, v): return f"{100*(base-v)/base:.1f}%" if not np.isnan(v) else "n/a"
    t4 = pd.DataFrame([
        {"Model": "LA-AIDS",                    "RMSE_mean": r_a_mu,   "RMSE_std": r_a_se,
         "KL_mean": kl_a_mu,   "KL_std": kl_a_se,   "Reduction": "baseline", "n_runs": N_RUNS},
        {"Model": "Neural Demand (static)",     "RMSE_mean": r_n_mu,   "RMSE_std": r_n_se,
         "KL_mean": kl_n_mu,   "KL_std": kl_n_se,   "Reduction": _pct(r_a_mu, r_n_mu), "n_runs": N_RUNS},
        {"Model": "Neural Demand (habit)",      "RMSE_mean": r_m_mu,   "RMSE_std": r_m_se,
         "KL_mean": kl_m_mu,   "KL_std": kl_m_se,   "Reduction": _pct(r_a_mu, r_m_mu), "n_runs": N_RUNS},
        {"Model": "Neural Demand (CF)",         "RMSE_mean": r_ncf_mu, "RMSE_std": r_ncf_se,
         "KL_mean": kl_ncf_mu, "KL_std": kl_ncf_se, "Reduction": _pct(r_a_mu, r_ncf_mu), "n_runs": N_RUNS},
        {"Model": "Neural Demand (habit, CF)",  "RMSE_mean": r_mcf_mu, "RMSE_std": r_mcf_se,
         "KL_mean": kl_mcf_mu, "KL_std": kl_mcf_se, "Reduction": _pct(r_a_mu, r_mcf_mu), "n_runs": N_RUNS},
    ]).round(6)
    t4.to_csv(f"{out_dir}/table4_mdp.csv", index=False)
    print(f"  Saved: {out_dir}/table4_mdp.csv")

    # ── Table 8: Cross-Price Elasticities ─────────────────────────────────────
    # We report the full matrix for Neural Demand (habit) and maybe others
    # User requested SEs.
    cross_elast_mean = agg["cross_elast_mean"]
    cross_elast_std  = agg["cross_elast_std"]
    
    # We'll make a table for Neural Demand (habit) specifically, or comparison?
    # "Add cross-price elasticity standard errors... The key structural claim is that the aspirin-ibuprofen cross-price effect is near zero"
    # So we definitely need Neural Demand (habit).
    
    t8_rows = []
    target_model = "Neural Demand (habit)"
    if target_model in cross_elast_mean:
        mat_mu = cross_elast_mean[target_model]
        mat_se = cross_elast_std[target_model]
        for i in range(G):
            for j in range(G):
                t8_rows.append({
                    "Model": target_model,
                    "Shock_Good": GOODS[i],
                    "Response_Good": GOODS[j],
                    "Elasticity_mean": mat_mu[i, j],
                    "Elasticity_std": mat_se[i, j],
                })
    pd.DataFrame(t8_rows).round(4).to_csv(f"{out_dir}/table8_cross_elast.csv", index=False)
    print(f"  Saved: {out_dir}/table8_cross_elast.csv")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    def L(*lines): return list(lines)

    tex = []
    tex += L(
        r"% ================================================================",
        r"% EMPIRICAL APPLICATION — Dominick's Analgesics",
        r"% Auto-generated by run_dominicks_experiments.py",
        rf"% n\_runs = {N_RUNS}  (mean $\pm$ std across independent re-estimations)",
        r"% Packages: booktabs, threeparttable, graphicx, siunitx",
        r"% ================================================================", "")

    # Table D0: Descriptive stats
    tex += L(
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Descriptive Statistics: Dominick's Analgesics Scanner Panel}",
        r"  \label{tab:dom_desc}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{lS[table-format=2.3]S[table-format=1.3]S[table-format=1.4]S[table-format=1.4]}",
        r"      \toprule",
        r"      \textbf{Good} & {\textbf{Mean Price}} & {\textbf{Std Price}} & {\textbf{Mean Share}} & {\textbf{Std Share}} \\",
        r"      & {(\$/100 tablets)} & & & \\",
        r"      \midrule",
    )
    for _, row in desc.iterrows():
        tex.append(f"      {row['Good']} & {row['Mean price']:.3f} & "
                   f"{row['Std price']:.3f} & {row['Mean share']:.4f} & "
                   f"{row['Std share']:.4f} \\\\")
    tex += L(
        r"      \midrule",
        f"      \\multicolumn{{5}}{{l}}{{\\textit{{Train: {len(tr):,} obs\\quad"
        f" Test: {len(te):,} obs\\quad Stores: {N_STORES}\\quad"
        f" Weeks: {int(weeks.min())}--{int(weeks.max())}}}}} \\\\",
        r"      \bottomrule",
        r"    \end{tabular}",
        r"    \begin{tablenotes}\small",
        r"      \item Unit prices standardised to per-100-tablet equivalent.",
        r"    \end{tablenotes}",
        r"  \end{threeparttable}",
        r"\end{table}", "")

    # Table D1: Accuracy
    best_rmse_nm = min(MODEL_NAMES, key=lambda nm: rmse_mean.get(nm, float("inf")))
    tex += L(
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Out-of-Sample Predictive Accuracy --- Dominick's Analgesics "
        rf"({N_RUNS} independent re-estimations; mean $\pm$ std)}}",
        r"  \label{tab:dom_acc}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{lcc}",
        r"      \toprule",
        r"      \textbf{Model} & \textbf{RMSE} & \textbf{MAE} \\",
        r"      \midrule",
    )
    for nm in MODEL_NAMES:
        b  = r"\textbf{" if nm == best_rmse_nm else ""
        bc = "}" if b else ""
        rm = rmse_mean.get(nm, float("nan")); rs = rmse_std.get(nm, float("nan"))
        mm = mae_mean.get(nm,  float("nan")); ms = mae_std.get(nm,  float("nan"))
        r_str = (f"${rm:.5f} \\pm {rs:.5f}$" if N_RUNS > 1 else f"${rm:.5f}$")
        m_str = (f"${mm:.5f} \\pm {ms:.5f}$" if N_RUNS > 1 else f"${mm:.5f}$")
        tex.append(f"      {b}{nm}{bc} & {b}{r_str}{bc} & {m_str} \\\\")
    tex += L(r"      \bottomrule",
             r"    \end{tabular}",
             r"    \begin{tablenotes}\small",
             rf"      \item RMSE and MAE on held-out test observations, mean $\pm$ std over {N_RUNS} run(s).",
             r"    \end{tablenotes}",
             r"  \end{threeparttable}",
             r"\end{table}", "")

    # Table D2: Elasticities
    tex += L(
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Own-Price Quantity Elasticities --- Dominick's Analgesics "
        rf"({N_RUNS} run(s); mean $\pm$ std)}}",
        r"  \label{tab:dom_elast}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{lccc}",
        r"      \toprule",
        r"      \textbf{Model} & {$\hat{\varepsilon}_{00}$ (Aspirin)} "
        r"& {$\hat{\varepsilon}_{11}$ (Acetaminophen)} "
        r"& {$\hat{\varepsilon}_{22}$ (Ibuprofen)} \\",
        r"      \midrule",
    )
    for nm in MODEL_NAMES:
        e_mu = elast_mean.get(nm, np.full(G, float("nan")))
        e_se = elast_std.get(nm,  np.full(G, float("nan")))
        def _fe(mu, se):
            if np.isnan(mu): return "{---}"
            return (f"${mu:.3f} \\pm {se:.3f}$" if N_RUNS > 1 else f"${mu:.3f}$")
        row = " & ".join(_fe(e_mu[j], e_se[j]) for j in range(G))
        tex.append(f"      {nm} & {row} \\\\")
    tex += L(r"      \bottomrule",
             r"    \end{tabular}",
             r"    \begin{tablenotes}\small",
             r"      \item Numerical own-price quantity elasticities at mean test prices.",
             r"    \end{tablenotes}",
             r"  \end{threeparttable}",
             r"\end{table}", "")

    # Table D3: Welfare
    tex += L(
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{Consumer Surplus Loss from {int(ss*100)}\% Ibuprofen Price Increase --- "
        rf"Dominick\'s Analgesics ({N_RUNS} run(s); mean $\pm$ std)}}",
        r"  \label{tab:dom_welfare}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{lcr}",
        r"      \toprule",
        r"      \textbf{Model} & \textbf{CV Loss (\$)} & \textbf{vs Neural Demand (static)} \\",
        r"      \midrule",
    )
    for nm in MODEL_NAMES:
        v  = welf_mean.get(nm,  float("nan"))
        se = welf_std.get(nm,   float("nan"))
        diff = ("" if nm == "Neural Demand (static)" or np.isnan(nw_mu)
                else f"{100*(v-nw_mu)/abs(nw_mu):+.1f}\\%")
        cv_str = (f"${v:+.4f} \\pm {se:.4f}$" if N_RUNS > 1 else f"${v:+.4f}$")
        tex.append(f"      {nm} & {cv_str} & {diff} \\\\")
    tex += L(r"      \bottomrule",
             r"    \end{tabular}",
             r"    \begin{tablenotes}\small",
             rf"      \item Compensating variation via 100-step Riemann sum, "
             rf"$p_{{\mathrm{{Ibu}}}}\to(1+{ss})\,p_{{\mathrm{{Ibu}}}}$.",
             r"    \end{tablenotes}",
             r"  \end{threeparttable}",
             r"\end{table}", "")

    # Extra Welfare Tables
    for g_idx in welf_all_mean:
        gn = GOODS[g_idx]
        tex += L(
            r"\begin{table}[htbp]",
            r"  \centering",
            rf"  \caption{{Consumer Surplus Loss from {int(ss*100)}\% {gn} Price Increase --- "
            rf"Dominick\'s Analgesics ({N_RUNS} run(s); mean $\pm$ std)}}",
            rf"  \label{{tab:dom_welfare_{gn.lower()}}}",
            r"  \begin{threeparttable}",
            r"    \begin{tabular}{lcr}",
            r"      \toprule",
            r"      \textbf{Model} & \textbf{CV Loss (\$)} & \textbf{vs Neural Demand (static)} \\",
            r"      \midrule",
        )
        for nm in MODEL_NAMES:
            v  = welf_all_mean[g_idx].get(nm, float("nan")) * 100.0
            se = welf_all_std[g_idx].get(nm,  float("nan")) * 100.0
            nw_mu_g = welf_all_mean[g_idx].get("Neural Demand (static)", float("nan")) * 100.0
            diff = ("" if nm == "Neural Demand (static)" or np.isnan(nw_mu_g)
                    else f"{100*(v-nw_mu_g)/abs(nw_mu_g):+.1f}\\%")
            cv_str = (f"${v:+.4f} \\pm {se:.4f}$" if N_RUNS > 1 else f"${v:+.4f}$")
            tex.append(f"      {nm} & {cv_str} & {diff} \\\\")
        tex += L(r"      \bottomrule",
                 r"    \end{tabular}",
                 r"    \begin{tablenotes}\small",
                 rf"      \item Compensating variation via 100-step Riemann sum, "
                 rf"$p_{{{gn}}}\to(1+{ss})\,p_{{{gn}}}$.",
                 r"    \end{tablenotes}",
                 r"  \end{threeparttable}",
                 r"\end{table}", "")

    # Table D4: MDP advantage
    # Need to extract Placebo RMSE from perf dict
    r_mp_mu = np.nanmean([r['perf']['PlaceboND']['RMSE'] for r in all_runs])
    r_mp_se = np.nanstd([r['perf']['PlaceboND']['RMSE'] for r in all_runs], ddof=ddof)

    mdp_rows = [
        ("LA-AIDS",                   r_a_mu,   r_a_se,   kl_a_mu,   kl_a_se,   "baseline"),
        ("Neural Demand (static)",    r_n_mu,   r_n_se,   kl_n_mu,   kl_n_se,   _pct(r_a_mu, r_n_mu)),
        ("Neural Demand (habit)",     r_m_mu,   r_m_se,   kl_m_mu,   kl_m_se,   _pct(r_a_mu, r_m_mu)),
        ("PlaceboND",   r_mp_mu,  r_mp_se,  kl_mp_mu,  kl_mp_se,  _pct(r_a_mu, r_mp_mu)),
        ("Neural Demand (CF)",        r_ncf_mu, r_ncf_se, kl_ncf_mu, kl_ncf_se, _pct(r_a_mu, r_ncf_mu)),
        ("Neural Demand (habit, CF)", r_mcf_mu, r_mcf_se, kl_mcf_mu, kl_mcf_se, _pct(r_a_mu, r_mcf_mu)),
    ]
    # Placebo row
    # r_mp_mu = np.nanmean(_arr('perf')['PlaceboND']['RMSE']) if 'PlaceboND' in _arr('perf')[0] else np.nan
    # r_mp_se = np.nanstd(_arr('perf')['PlaceboND']['RMSE'], ddof=ddof) if 'PlaceboND' in _arr('perf')[0] else np.nan
    # kl_mp_mu = np.nanmean(_arr('kl_m_placebo')) if 'kl_m_placebo' in all_runs[0] else np.nan # Need to extract this
    # kl_mp_se = np.nanstd(_arr('kl_m_placebo'), ddof=ddof) if 'kl_m_placebo' in all_runs[0] else np.nan

    # Extract KL for placebo from all_runs manually since it wasn't in _arr keys
    # kl_mp_vals = [kl_div('mdp', p_te, y_te, w_te, cfg, xb_prev=r['KW']['placebo_nd'].xb_prev_data, q_prev=r['KW']['placebo_nd'].q_prev_data) for r in all_runs] # Wait, this is tricky. run_once returns specific keys.
    
    # Better: Add kl_mp to run_once return dict
    
    dm_stat = agg["dm_stat_mu"]
    dm_p    = agg["dm_p_mu"]
    dm_diff = agg["dm_diff_mu"]
    dm_sig  = "^{***}" if dm_p < 0.001 else "^{**}" if dm_p < 0.01 else "^{*}" if dm_p < 0.05 else ""
    
    tex += L(
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{MDP State Augmentation: Brand Loyalty in Analgesic Demand "
        rf"({N_RUNS} run(s); mean $\pm$ std)}}",
        r"  \label{tab:dom_mdp}",
        r"  \begin{threeparttable}",
        r"    \begin{tabular}{lcccc}",
        r"      \toprule",
        r"      \textbf{Model} & \textbf{RMSE} & \textbf{KL Div.} & \textbf{Reduction} \\",
        r"      \midrule",
    )
    for (mn, rm, rs, km, ks, rd) in mdp_rows:
        b  = r"\textbf{" if ("MDP" in mn or "Window" in mn) else ""
        bc = "}" if b else ""
        r_str = (f"${rm:.5f} \\pm {rs:.5f}$" if N_RUNS > 1 else f"${rm:.5f}$")
        k_str = (f"${km:.5f} \\pm {ks:.5f}$" if N_RUNS > 1 else f"${km:.5f}$")
        tex.append(f"      {b}{mn}{bc} & {b}{r_str}{bc} & {k_str} & {rd} \\\\")
    tex += L(r"      \bottomrule",
             r"    \end{tabular}",
             r"    \begin{tablenotes}\small",
             rf"      \item Habit-decay $\hat{{\delta}}$ (habit model) = {delta_m_mu:.3f}$\pm${delta_m_se:.3f}; "
             rf"FE variant: {delta_mf_mu:.3f}$\pm${delta_mf_se:.3f}.",
             rf"      RMSE and KL: mean $\pm$ std over {N_RUNS} independent re-estimation(s).",
             rf"      \item \textbf{{Diebold-Mariano Test}}: Static vs Habit (blocked by store). "
             rf"DM stat = {dm_stat:.2f}{dm_sig} ($p={dm_p:.3e}$). "
             rf"Positive stat favors Habit model.",
             r"    \end{tablenotes}",
             r"  \end{threeparttable}",
             r"\end{table}", "")

    # Table D8: Cross-Price Elasticities (Neural Demand habit)
    if "Neural Demand (habit)" in cross_elast_mean:
        mat_mu = cross_elast_mean["Neural Demand (habit)"]
        mat_se = cross_elast_std["Neural Demand (habit)"]
        tex += L(
            r"\begin{table}[htbp]",
            r"  \centering",
            rf"  \caption{{Cross-Price Elasticity Matrix (Neural Demand (habit)) "
            rf"({N_RUNS} run(s); mean $\pm$ std)}}",
            r"  \label{tab:dom_cross_elast_table}",
            r"  \begin{threeparttable}",
            r"    \begin{tabular}{lccc}",
            r"      \toprule",
            r"      & \multicolumn{3}{c}{\textbf{Price of}} \\",
            r"      \cmidrule(lr){2-4}",
            r"      \textbf{Quantity of} & \textbf{Aspirin} & \textbf{Acetaminophen} & \textbf{Ibuprofen} \\",
            r"      \midrule",
        )
        for i in range(G):
            row_cells = []
            for j in range(G):
                val = mat_mu[i, j]
                se  = mat_se[i, j]
                cell = (f"${val:.3f} \\pm {se:.3f}$" if N_RUNS > 1 else f"${val:.3f}$")
                if i == j: cell = r"\textbf{" + cell + "}"
                row_cells.append(cell)
            tex.append(f"      {GOODS[i]} & " + " & ".join(row_cells) + r" \\")
        tex += L(r"      \bottomrule",
                 r"    \end{tabular}",
                 r"    \begin{tablenotes}\small",
                 r"      \item Row $i$, Column $j$: $\epsilon_{ij} = \partial \log q_i / \partial \log p_j$.",
                 r"      \item Evaluated at mean test prices. Mean $\pm$ std over runs.",
                 r"    \end{tablenotes}",
                 r"  \end{threeparttable}",
                 r"\end{table}", "")

    # Figure environments
    _se_band_note = (rf" Shaded bands indicate $\pm 1$ standard deviation across "
                     rf"{N_RUNS} independent re-estimations." if N_RUNS > 1 else "")
    FDEFS = [
        ("fig_demand_curves",
         f"Aspirin Budget Share as a Function of Ibuprofen Unit Price --- "
         f"Dominick's Analgesics.{_se_band_note}", "fig:dom_demand"),
        ("fig_mdp_advantage",
         f"Cross-Price Demand Matrix — Dominick's Analgesics.{_se_band_note}", "fig:dom_mdp"),
        ("fig_convergence",
         f"Training Convergence — Dominick's Analgesics (last run).", "fig:dom_conv"),
    ]
    if N_RUNS > 1:
        FDEFS += [
            ("fig_rmse_bars",
             f"Out-of-Sample RMSE across all models — mean $\\pm 1$ SD over {N_RUNS} runs.",
             "fig:dom_rmse_bars"),
        ]
    FDEFS += [
        ("fig_cross_elast_heatmap", "Cross-Price Elasticity Heatmaps.", "fig:dom_cross_elast"),
        ("fig_segmentation_sorting", "Market segmentation and habit-sorting diagnostics.",
         "fig:dom_sorting"),
        ("fig_mdp_decomposition",
         "MDP demand decomposition: structural price effect vs. habit-stock sorting.",
         "fig:dom_decomp"),
    ]
    for fn, cap, lbl in FDEFS:
        tex += L(r"\begin{figure}[htbp]",
                 r"  \centering",
                 f"  \\includegraphics[width=\\textwidth]{{{fig_dir}/{fn}.pdf}}",
                 f"  \\caption{{{cap}}}",
                 f"  \\label{{{lbl}}}",
                 r"\end{figure}", "")

    tp = f"{out_dir}/dominicks_latex.tex"
    with open(tp, "w") as f:
        f.write("\n".join(tex))
    print(f"  Saved: {tp}")
