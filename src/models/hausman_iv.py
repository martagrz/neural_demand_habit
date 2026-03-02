"""Hausman IV construction and control-function first stage."""

import numpy as np


def hausman_iv(prices, stores, weeks):
    """Mean price of same good across other stores in same week."""

    G = prices.shape[1]
    Z = np.zeros_like(prices)
    for j in range(G):
        for i, (s, wk) in enumerate(zip(stores, weeks)):
            mask = (stores != s) & (weeks == wk)
            Z[i, j] = prices[mask, j].mean() if mask.sum() else prices[:, j].mean()
    return Z


def cf_first_stage(log_p, Z):
    """OLS first stage for the control-function (CF) endogeneity correction.

    For each good j, regresses log(p_j) on [1, Z_j] to obtain fitted values.
    Returns the OLS residuals  v̂_j = log(p_j) − ŷ_j  as additional inputs
    to the neural demand network so that price endogeneity is absorbed.

    At *evaluation* time (counterfactual welfare / elasticities) pass
    v_hat = zeros so the network recovers the structural demand without
    the endogenous component.

    Parameters
    ----------
    log_p : ndarray (N, G)
        Log prices.
    Z     : ndarray (N, G)
        Instruments.  For the Dominick's pipeline these are Hausman mean
        prices across stores; for the simulation pipeline these are the
        cost-shifter vectors Z used to generate prices.

    Returns
    -------
    v_hat : ndarray (N, G)
        First-stage OLS residuals.
    r_sq  : ndarray (G,)
        First-stage R² for each good.
    """
    N, G = log_p.shape
    v_hat = np.zeros((N, G), dtype=np.float64)
    r_sq  = np.zeros(G, dtype=np.float64)
    for j in range(G):
        X_j     = np.column_stack([np.ones(N), Z[:, j]])
        gamma_j = np.linalg.lstsq(X_j, log_p[:, j], rcond=None)[0]
        fitted  = X_j @ gamma_j
        resid   = log_p[:, j] - fitted
        v_hat[:, j] = resid
        ss_tot  = np.sum((log_p[:, j] - log_p[:, j].mean()) ** 2)
        r_sq[j] = 1.0 - np.sum(resid ** 2) / max(ss_tot, 1e-12)
    return v_hat.astype(np.float32), r_sq
