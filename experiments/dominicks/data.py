"""
experiments/dominicks/data.py
==============================
Section 1 and Section 8 of dominicks_multiple_runs.py.

Data loading, UPC classification, panel construction, and train/test split.
All functions are pure (no global side effects) and accept a config dict.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  PRODUCT CLASSIFICATION KEYS
# ─────────────────────────────────────────────────────────────────────────────

_IBU_KEYS = [
    'ADVIL', '#ADVIL', 'MOTRIN', '#MOTRIN', '~MOTRIN',
    'IBUPROFEN', 'NUPRIN', '~NUPRIN', 'ALEVE', 'IBUPRIN',
    'ACTRON', 'ORUDIS', 'HALTRAN', 'MEDIPREN',
    'MIDOL 200', 'MIDOL IB', '~MIDOL IB', 'MYDOL 200', 'MENADOL',
    'TRENDAR', 'VITALAN', 'DOM IBUPROFEN', 'DOMINICKS TAB-PROFEN',
    'TC IBUROFEN', 'TOPCARE IBUPROFEN',
    'EXCEDRIN IB', '~EXCEDRIN IB', '~ARTH FNDT IB',
    'F/S ADVIL', 'TRL-SZ ADVIL', 'TRL-SZ MOTRIN',
]
_ACET_KEYS = [
    'TYLENOL', 'TYLNL', 'ACETAMINOPHEN',
    'EXCEDRIN ASPIRIN-FREE', 'EXCEDRIN A/F', 'A/F EXCEDRIN',
    'A/F 20 CT EXCEDRIN', 'A/F 80 CT EXCEDRIN',
    '~A/F 40 CT EXCEDRIN', '*EXCEDRIN',
    'EXCEDRIN 175', 'EXCEDRIN CAPS', 'EXCEDRIN CAP ',
    'EXCEDRIN DUAL', 'EXCEDRIN GELTAB', 'EXCEDRIN TAB',
    'EXCEDRIN TABS', 'EXCEDRIN X', 'EXCEDRIN PM', 'EXCEDRIN P M',
    'EXCEDRIN P.M', 'EXCERIN PM', 'TRL-SZ EXCEDRIN', '~EXCEDRIN P.M',
    'PANADOL', 'DATRIL',
    'PAMPRIN', '~PAMPRIN',
    'ANACIN-3', 'ANACIN 3 ', 'M/S ANACIN 3', 'JHO-ANACIN 3',
    'ANACIN A/F', '~ANACIN A/F', '~ASP FREE ANANCIN', '~ASPIRIN FREE ANACIN',
    'DOM NON-ASP', 'DOM NON ASP', 'DOM X/S NON', 'DOM X-STR NON',
    'DOM REG STR NON', 'DOM CHILD CHEW', 'DOM CHILDS', 'DOM E/S NON',
    'DOM ADDED STRENGTH A', 'DOM X/S PAIN RELIEVE',
    'ST JOSEPH A/F', 'TC NON-ASP', 'TC X/STR N/A',
    'TEMPRA', 'FEVERALL', 'DORCOL', 'FEVERNOL', 'LIQUIPRIN',
    'PEDIA CARE INFANT', '$TC PAIN REL INFANT',
    'CHILD CHEW FRT TYLE', 'CHILD CHEW GRAP TYLE',
    'TS TYLENOL CHEW', 'CHLDRN PANADOL',
    'VALUE TIME ACETA',
    'MIDOL CAPLET', 'MIDOL CAPLETS MAX', 'MIDOL M/S', 'MIDOL MAXIMUM',
    'MIDOL PM', 'MIDOL PMS', 'MIDOLPMS', 'T/S MIDOL PM', '~MIDOL PM',
    'TEEN MIDOL', '~TEEN MIDOL',
    'PREMSYN PMS', 'PMS BALANCE', '~TC PMS',
    'X/S CAPLETS', 'XS LIQUID', 'X/S TYLENOL',
    'YL PHARMACIST', 'E.S TYLENOL', 'ES TYLENOL', 'EX STR TYLENOL',
    'EX TYLENOL', 'TRL-SZ EXCEDRIN',
    '~TYLENOL', '~ANACIN A/F',
]
_ASP_KEYS = [
    'ASPIRIN', 'BAYER', '~BAYER', 'BUFFERIN', '~BUFFERIN',
    'ASCRIPTIN', '~ACSRIPTIN', 'ECOTRIN', 'EMPIRIN', 'HALFRIN',
    'NORWICH', 'VALUE TIME ASP', 'DOM COATED ASP', 'DOM ENTERIC COATED',
    'DOM TRI0BUFFERED', 'RUGBY COATED ASPIRIN', 'GENUINE BAYER',
    'EXTRA STRENGTH BAYER', 'CAMA ARTHRITIS', 'VANQUISH',
    'COPE TABS', 'MOMENTUM CAP', 'MOMENTUN CAP', '~MOMENTUM',
    'BACKAID', 'B.C ', 'BC HEADACHE', 'CONGESPIRIN',
    'DOMINICKS CHILD ASPI',
    'ANACIN TABS', 'ANACIN CAP', 'ANACIN CPLT', 'ANACIN MAX',
    'ANACIN ARTH', 'ANACIN TABLETS',
    'JHO-ANACIN PEGABLE', 'T/S F/S ANACIN PM',
    'ARTH FNDTN ASP',
    '~ANACIN 125', '~ANACIN EASY', '~ANACIN MAX TAB', '~ANANCIN PAIN FORM',
    'MAX BAYER', 'EX BAYER', 'F/S BAYER', 'T/S BAYER', 'F/S NORWICH',
]

GOODS = ['Aspirin', 'Acetaminophen', 'Ibuprofen']
G = 3


# ─────────────────────────────────────────────────────────────────────────────
#  UPC CLASSIFICATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _classify(descrip: str) -> str:
    """Classify one DESCRIP string. IBU checked first to handle Excedrin IB."""
    d = descrip.upper()
    for k in _IBU_KEYS:
        if k.upper() in d:
            return 'IBU'
    for k in _ACET_KEYS:
        if k.upper() in d:
            return 'ACET'
    for k in _ASP_KEYS:
        if k.upper() in d:
            return 'ASP'
    return 'OTHER'


def _parse_tablets(size: str) -> float:
    """Extract tablet count from SIZE string.

    '100 CT' → 100  |  '2/50 C' → 100  |  '2-100' → 200  |  '4 OZ' → NaN
    """
    if not isinstance(size, str):
        return np.nan
    s = size.strip().upper()
    m = re.match(r'(\d+)[/\-](\d+)', s)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.match(r'(\d+\.?\d*)\s*CT', s)
    if m:
        return float(m.group(1))
    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  PANEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_panel(cfg: dict) -> pd.DataFrame:
    """Load raw files → balanced (store, week, good) panel.

    Steps
    -----
    1. Classify each UPC into ASP / ACET / IBU / OTHER.
    2. unit_price = PRICE × STD_TABLETS / tablets  ($/100-tablet basis).
    3. Aggregate per (store, week, category):
         price   = revenue-weighted mean unit price
         units   = Σ MOVE × QTY
         revenue = Σ MOVE × QTY × PRICE
    4. Forward-/backward-fill prices within store×good for zero-sales weeks.
    5. Pivot to wide format; drop rows with any missing price.
    6. Drop stores below min_store_wks.
    """
    print('\n[1/7] Loading data...')
    wdf = pd.read_csv(cfg['weekly_path'])
    udf = pd.read_csv(cfg['upc_path'])
    print(f'  Weekly panel : {len(wdf):,} rows | '
          f'{wdf.STORE.nunique()} stores | {wdf.UPC.nunique()} UPCs | '
          f'weeks {wdf.WEEK.min()}–{wdf.WEEK.max()}')
    print(f'  UPC catalogue: {len(udf):,} products')

    udf = udf.copy()
    udf['CAT']     = udf['DESCRIP'].apply(_classify)
    udf['TABLETS'] = udf['SIZE'].apply(_parse_tablets)
    cc = udf.CAT.value_counts()
    print(f'  UPC assignment — ASP: {cc.get("ASP",0)}, '
          f'ACET: {cc.get("ACET",0)}, IBU: {cc.get("IBU",0)}, '
          f'OTHER (excluded): {cc.get("OTHER",0)}')

    merged = wdf.merge(udf[['UPC', 'CAT', 'TABLETS']], on='UPC', how='left')
    merged['UNITS']   = merged['MOVE'] * merged['QTY']
    merged['REVENUE'] = merged['UNITS'] * merged['PRICE']

    other_rev = (merged[merged['CAT'] == 'OTHER']
                 .groupby(['STORE', 'WEEK'])['REVENUE'].sum()
                 .reset_index().rename(columns={'REVENUE': 'R_OTHER'}))
    print(f'  OTHER revenue: {len(other_rev):,} store×week obs with other-analgesic sales')

    merged = merged[merged['CAT'].isin(['ASP', 'ACET', 'IBU'])].copy()
    std = cfg['std_tablets']
    merged['UNIT_PX'] = np.where(
        (merged['TABLETS'] > 0) & (merged['PRICE'] > 0),
        merged['PRICE'] * std / merged['TABLETS'], np.nan)

    def _rev_wtd(grp):
        pos = grp[(grp['UNITS'] > 0)].dropna(subset=['UNIT_PX'])
        val = grp.dropna(subset=['UNIT_PX'])
        if len(pos):
            return np.average(pos['UNIT_PX'], weights=pos['UNITS'])
        return val['UNIT_PX'].mean() if len(val) else np.nan

    agg = (merged.groupby(['STORE', 'WEEK', 'CAT'])
                 .apply(lambda g: pd.Series({
                     'PX':  _rev_wtd(g),
                     'QQ':  g['UNITS'].sum(),
                     'RR':  g['REVENUE'].sum()}))
                 .reset_index())

    agg = agg.sort_values(['STORE', 'CAT', 'WEEK'])
    agg['PX'] = agg.groupby(['STORE', 'CAT'])['PX'].transform(
        lambda s: s.ffill().bfill())

    cats = ['ASP', 'ACET', 'IBU']

    def _piv(col):
        p = agg.pivot_table(index=['STORE', 'WEEK'], columns='CAT',
                            values=col, aggfunc='first')
        p.columns.name = None
        return p

    px = _piv('PX'); qq = _piv('QQ'); rr = _piv('RR')
    panel = px.copy()
    for c in cats:
        panel[f'U_{c}'] = qq[c] if c in qq.columns else 0.0
        panel[f'R_{c}'] = rr[c] if c in rr.columns else 0.0
    panel = panel.dropna(subset=cats).reset_index()

    panel = panel.merge(other_rev, on=['STORE', 'WEEK'], how='left')
    panel['R_OTHER'] = panel['R_OTHER'].fillna(0.0)

    sw = panel.groupby('STORE')['WEEK'].count()
    panel = panel[panel['STORE'].isin(
        sw[sw >= cfg['min_store_wks']].index)].reset_index(drop=True)

    n_pos = (panel[[f'U_{c}' for c in cats]].sum(axis=1) > 0).sum()
    other_frac = panel['R_OTHER'].sum() / (
        panel[['R_ASP', 'R_ACET', 'R_IBU', 'R_OTHER']].sum().sum() + 1e-9)
    print(f'  Panel: {len(panel):,} store×week obs | '
          f'{panel.STORE.nunique()} stores | {n_pos:,} with positive sales')
    print(f'  OTHER outside share: {other_frac:.1%} of total analgesic revenue')
    return panel


# ─────────────────────────────────────────────────────────────────────────────
#  ARRAY CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_arrays(panel: pd.DataFrame, cfg: dict) -> dict:
    """Convert wide panel to numpy arrays.

    Habit stock is tracked in **log-share space** rather than quantity space.
    This avoids unit/price-dependency and the log-of-negative issue that arises
    when demeaning raw quantities (which can be negative after subtracting the
    store mean).  Log-shares are always well-defined (shares are strictly positive)
    and naturally aligned with the model's output space.

    The habit update rule in log-share space is:
        log_xb_t = delta * log_xb_{t-1} + (1-delta) * log(w_{t-1})

    The model learns its own delta; the delta used here only affects the warmup
    initialisation and has negligible impact after the first few observations.
    """
    cats  = ['ASP', 'ACET', 'IBU']
    panel = panel.sort_values(['STORE', 'WEEK']).reset_index(drop=True)

    prices  = panel[cats].values.astype(float)
    rev     = np.stack([panel[f'R_{c}'].fillna(0).values for c in cats], 1).astype(float)
    r_other = panel['R_OTHER'].fillna(0).values.astype(float)   # outside-option revenue

    # 1. Total Revenue (Income Proxy) — inside goods only, as before
    tot = rev.sum(1, keepdims=True)

    # 2a. Conditional shares (sum to 1 over inside goods) — used by all models.
    #     Use safe division to avoid RuntimeWarning for zero-revenue weeks.
    tot_safe = np.where(tot > 0, tot, 1.0)
    shares = np.where(tot > 0, rev / tot_safe, 1.0 / G)
    shares = np.clip(shares, 1e-6, 1.0)
    shares /= shares.sum(1, keepdims=True)

    # 2b. Market shares (sum to 1 over inside + outside) — kept for diagnostics.
    #     Outside option = OTHER analgesics not classified as ASP/ACET/IBU.
    tot_mkt = rev.sum(1) + r_other                                  # (N,)
    mkt_rev = np.column_stack([rev, r_other])                       # (N, G+1)
    
    tot_mkt_safe = np.where(tot_mkt[:, None] > 0, tot_mkt[:, None], 1.0)
    mkt_shares = np.where(
        tot_mkt[:, None] > 0,
        mkt_rev / tot_mkt_safe,
        np.array([1.0/G]*G + [0.0]))                                # (N, G+1)
    mkt_shares = np.clip(mkt_shares, 1e-8, 1.0)
    mkt_shares /= mkt_shares.sum(1, keepdims=True)                  # renormalise

    # 3. Income Scaling: Convert to "Hundreds of Dollars"
    #    Real revenue ~ $1000 -> 10.0. Log(10) ~ 2.3.
    #    This aligns log_y with log_p (which is ~1.5)
    income = np.maximum(tot.squeeze(), 1.0) / 100.0

    # 4. Habit Stock: Track LOG-SHARES (not quantities)
    #    log-shares are in (-inf, 0), always finite, and already in the same
    #    natural space as the model outputs.  The model's forward pass computes
    #    xb_input = delta * log_xb_prev + (1-delta) * log_q_prev, which is a
    #    convex combination of two log-share vectors -- well-defined and stable.
    log_w  = np.log(np.maximum(shares, 1e-6))   # (N, G) log-shares
    delta  = cfg['habit_decay']                  # warmup only; model learns its own delta

    xb     = np.zeros_like(log_w)   # log-habit stock at time i  (= log_xb_prev)
    q_prev = np.zeros_like(log_w)   # log-share of previous period (= log_q_prev)
    stv    = panel['STORE'].values
    gm     = log_w.mean(0)
    prev   = gm.copy()
    prev_q = gm.copy()

    for i in range(len(shares)):
        if i > 0 and stv[i] != stv[i - 1]:
            prev   = gm.copy()
            prev_q = gm.copy()
        xb[i]     = prev
        q_prev[i] = prev_q
        prev_q    = log_w[i]
        prev      = delta * prev + (1.0 - delta) * log_w[i]

    return dict(prices=prices, shares=shares, mkt_shares=mkt_shares,
                income=income, xbar=xb, q_prev=q_prev,
                log_shares=log_w,
                week=panel['WEEK'].values, store=stv)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN/TEST SPLIT AND STORE ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def prepare_splits(data: dict, cfg: dict) -> dict:
    """Build train/test index split and store-FE encoding.

    Returns a dict with all tr/te arrays, store maps, grid arrays, etc.
    """
    import torch
    from src.models.dominicks import hausman_iv

    prices     = data['prices']
    shares     = data['shares']
    mkt_shares = data['mkt_shares']
    income     = data['income']
    xbar       = data['xbar']
    q_prev_raw = data['q_prev']
    log_shares = data['log_shares']
    weeks      = data['week']
    stores     = data['store']

    print('\n[2/7] Descriptive statistics:')
    desc = pd.DataFrame({
        'Good':       GOODS,
        'Mean price': prices.mean(0).round(3),
        'Std price':  prices.std(0).round(3),
        'Mean share': shares.mean(0).round(4),
        'Std share':  shares.std(0).round(4),
    })
    print(desc.to_string(index=False))

    print('\n[3/7] Train / test split...')
    tr = np.where(weeks < cfg['test_cutoff'])[0]
    te = np.where(weeks >= cfg['test_cutoff'])[0]
    if len(te) < 30 or len(tr) < 50:
        print(f'  Week split insufficient (tr={len(tr)}, te={len(te)}). '
              'Using random 75/25.')
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(prices))
        tr  = idx[:int(0.75 * len(prices))]
        te  = idx[int(0.75 * len(prices)):]

    p_tr, p_te     = prices[tr],      prices[te]
    w_tr, w_te     = shares[tr],      shares[te]
    mw_tr, mw_te   = mkt_shares[tr],  mkt_shares[te]
    y_tr, y_te     = income[tr],      income[te]
    xb_tr, xb_te   = xbar[tr],        xbar[te]
    qp_tr, qp_te   = q_prev_raw[tr],  q_prev_raw[te]
    ls_tr, ls_te   = log_shares[tr],  log_shares[te]
    s_tr, s_te     = stores[tr],      stores[te]
    wk_tr, wk_te   = weeks[tr],       weeks[te]
    print(f'  Train: {len(tr):,}  |  Test: {len(te):,}')

    # Store-index encoding
    _store_uniq = np.sort(np.unique(stores))
    _store_map  = {int(s): i for i, s in enumerate(_store_uniq)}
    N_STORES     = int(len(_store_uniq))
    STORE_EMB_DIM = 8
    s_tr_idx = np.array([_store_map[int(s)] for s in s_tr], dtype=np.int64)
    s_te_idx = np.array([_store_map[int(s)] for s in s_te], dtype=np.int64)
    s_te_mode_idx = int(np.bincount(s_te_idx).argmax())
    print(f'  Store FE: {N_STORES} unique stores → emb_dim={STORE_EMB_DIM} '
          f'| modal test store idx={s_te_mode_idx}')

    # Instruments  (same computation as dominicks_multiple_runs.py line 899)
    print('\n[4/7] Building Hausman instruments...')
    Z_tr = hausman_iv(p_tr, s_tr, wk_tr)
    # Diagnostic: correlation between own price and its Hausman IV
    for _g in range(G):
        _r = float(np.corrcoef(p_tr[:, _g], Z_tr[:, _g])[0, 1])
        print(f'  IV corr good {_g}: r={_r:.3f}  '
              f'(low → weak instrument → BLP alpha → 0)')
    print(f'  N_train={len(p_tr):,}  unique_stores={np.unique(s_tr).size}  '
          f'unique_weeks={np.unique(wk_tr).size}')

    # Price grids for demand curves
    sg   = cfg['shock_good']
    ss   = cfg['shock_pct']
    N_GR = 80

    pgr_all = []
    tpx_all = []
    for _g in range(G):
        _plo = float(np.percentile(p_te[:, _g], 5))
        _phi = float(np.percentile(p_te[:, _g], 95))
        _pgr = np.linspace(_plo, _phi, N_GR)
        _tpx = np.tile(p_te.mean(0), (N_GR, 1))
        _tpx[:, _g] = _pgr
        pgr_all.append(_pgr)
        tpx_all.append(_tpx)

    pgr = pgr_all[sg]
    tpx = tpx_all[sg]
    fy  = np.full(N_GR, float(y_te.mean()))

    p_mn  = p_te.mean(0)
    y_mn  = float(y_te.mean())
    xb_mn = xb_te.mean(0)
    qp_mn = qp_te.mean(0)
    p0w   = p_mn.copy()
    p1w   = p_mn.copy()
    p1w[sg] *= 1 + ss

    return dict(
        # indices
        tr=tr, te=te,
        # descriptive stats
        desc=desc,
        # arrays
        p_tr=p_tr, p_te=p_te,
        w_tr=w_tr, w_te=w_te,
        mw_tr=mw_tr, mw_te=mw_te,
        y_tr=y_tr, y_te=y_te,
        xb_tr=xb_tr, xb_te=xb_te,
        qp_tr=qp_tr, qp_te=qp_te,
        ls_tr=ls_tr, ls_te=ls_te,
        s_tr=s_tr, s_te=s_te,
        wk_tr=wk_tr, wk_te=wk_te,
        Z_tr=Z_tr,
        # store encoding
        _store_map=_store_map,
        N_STORES=N_STORES,
        STORE_EMB_DIM=STORE_EMB_DIM,
        s_tr_idx=s_tr_idx,
        s_te_idx=s_te_idx,
        s_te_mode_idx=s_te_mode_idx,
        # demand curve grids
        pgr_all=pgr_all, tpx_all=tpx_all,
        pgr=pgr, tpx=tpx, fy=fy,
        p_mn=p_mn, y_mn=y_mn,
        xb_mn=xb_mn, qp_mn=qp_mn,
        p0w=p0w, p1w=p1w,
        # raw
        shares=shares, stores=stores, weeks=weeks,
        log_shares=log_shares,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TOP-LEVEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load(cfg: dict) -> tuple[dict, dict]:
    """Load panel, build arrays, and prepare splits.

    Returns
    -------
    (data, splits) : raw arrays dict and split/grid dict.
    """
    import os
    os.makedirs(cfg['fig_dir'],  exist_ok=True)
    os.makedirs(cfg['out_dir'],  exist_ok=True)

    panel  = load_panel(cfg)
    data   = build_arrays(panel, cfg)
    splits = prepare_splits(data, cfg)
    return data, splits
