"""LA-AIDS, QUAIDS, and nonparametric Series demand model implementations."""

import numpy as np


class LAAIDS:
    """Linear Approximate AIDS with Stone price index, estimated by OLS."""

    name = "LA-AIDS"

    def fit(self, p, w, y):
        lp = np.log(np.maximum(p, 1e-8))
        lPs = (w * lp).sum(1, keepdims=True)
        ly = np.log(np.maximum(y, 1e-8)).reshape(-1, 1) - lPs
        X = np.c_[np.ones(len(y)), lp, ly.squeeze()]
        self.coef_ = np.linalg.lstsq(X, w, rcond=None)[0]
        return self

    def predict(self, p, y):
        g = p.shape[1]
        lp = np.log(np.maximum(p, 1e-8))
        lP = (np.full((len(p), g), 1.0 / g) * lp).sum(1, keepdims=True)
        ly = np.log(np.maximum(y, 1e-8)).reshape(-1, 1) - lP
        w = np.clip(np.c_[np.ones(len(p)), lp, ly.squeeze()] @ self.coef_, 1e-6, 1.0)
        return w / w.sum(1, keepdims=True)


class AIDSBench(LAAIDS):
    """Simulation benchmark alias keeping original non-chaining fit API."""

    name = "LA-AIDS"

    def fit(self, p, w, y):
        super().fit(p, w, y)


class QUAIDS(LAAIDS):
    """Quadratic AIDS (Banks, Blundell & Lewbel 1997).

    Extends LA-AIDS by adding a (ln(y/P))^2 term to each share equation,
    allowing the Engel curves to be quadratic in log income.  Estimated
    equation-by-equation by OLS (equivalent to SUR under homoskedasticity).

    w_i = α_i + Σ_j γ_ij ln(p_j) + β_i (ln y − ln P) + δ_i (ln y − ln P)^2

    where ln P ≈ Σ_j w̄_j ln(p_j)  (Stone price index at training means).
    """

    name = "QUAIDS"

    def fit(self, p, w, y):
        lp  = np.log(np.maximum(p, 1e-8))
        lPs = (w * lp).sum(1, keepdims=True)          # Stone price index (train)
        ly  = np.log(np.maximum(y, 1e-8)).reshape(-1, 1) - lPs
        ly2 = ly ** 2
        X = np.c_[np.ones(len(y)), lp, ly.squeeze(), ly2.squeeze()]
        self.coef_ = np.linalg.lstsq(X, w, rcond=None)[0]
        # Save training-mean share weights for Stone index at prediction time
        self._w_mean = w.mean(0)
        return self

    def predict(self, p, y):
        g   = p.shape[1]
        lp  = np.log(np.maximum(p, 1e-8))
        # Stone price index using training-mean share weights (avoids endogeneity)
        lP  = (lp * self._w_mean).sum(1, keepdims=True)
        ly  = np.log(np.maximum(y, 1e-8)).reshape(-1, 1) - lP
        ly2 = ly ** 2
        w = np.clip(
            np.c_[np.ones(len(p)), lp, ly.squeeze(), ly2.squeeze()] @ self.coef_,
            1e-6, 1.0)
        return w / w.sum(1, keepdims=True)


class SeriesDemand:
    """Nonparametric polynomial series (sieve) demand estimator.

    Approximates each budget share as a degree-D polynomial in log-prices
    and log-income, estimated equation-by-equation by OLS (equivalent to
    multivariate OLS here).  With degree=2 the model includes all linear
    terms, all squared terms, and all pairwise cross-products — giving a
    flexible quadratic sieve in the log-price/log-income space without
    imposing any parametric demand structure.

    Parameters
    ----------
    degree : int
        Polynomial degree.  2 (default) gives 15 features for a 3-good
        system (3 prices + 1 income = 4 covariates → C(4+2,2)=15 terms).
    """

    name = "Series Estm."

    def __init__(self, degree: int = 2):
        self.degree = degree

    def _features(self, p, y):
        """Build (N, K) polynomial feature matrix."""
        lp = np.log(np.maximum(p, 1e-8))          # (N, G)
        ly = np.log(np.maximum(y, 1e-8)).reshape(-1, 1)  # (N, 1)
        z  = np.hstack([lp, ly])                   # (N, G+1)
        n_z = z.shape[1]

        parts = [np.ones((len(p), 1)), z]          # degree-0 and degree-1
        if self.degree >= 2:
            for i in range(n_z):
                for j in range(i, n_z):
                    parts.append((z[:, i] * z[:, j]).reshape(-1, 1))
        if self.degree >= 3:
            for i in range(n_z):
                parts.append((z[:, i] ** 3).reshape(-1, 1))
        return np.hstack(parts).astype(np.float64)

    def fit(self, p, w, y):
        X = self._features(p, y)
        self.coef_ = np.linalg.lstsq(X, w, rcond=None)[0]
        return self

    def predict(self, p, y):
        X = self._features(p, y)
        w = np.clip(X @ self.coef_, 1e-6, 1.0)
        return w / w.sum(1, keepdims=True)

