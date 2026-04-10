"""Logit-IV demand model (aggregate logit with Hausman instruments).

The model treats the last column of w as the **outside option** share.
With the Dominick's data, this is the "OTHER" category (all analgesics not
classified as ASP, ACET, or IBU), giving a proper 3-inside-good / 1-outside
specification.  For simulations there is no genuine outside good; the caller
should add a small constant outside share (e.g. 1 %) before calling fit().

This is an aggregate logit with full-price 2SLS identification (a simplified
version of Berry 1994 / BLP 1995 without random coefficients):
  - First stage  : for each good g, regress p_g on [1, z_0, …, z_{G-1}]
                   (all G Hausman IVs) to obtain fitted prices p_hat_g.
  - Second stage : for each good g, regress log(s_g / s_0) on
                   [1, p_hat_0, …, p_hat_{G-1}, y] (fitted prices + income),
                   yielding own- and cross-price utility coefficients and an
                   income coefficient per good.

Both stages use ridge regression (penalty `ridge_lambda`) so that the
estimator remains numerically stable when K (number of goods) is large and
instrument columns are correlated.  The intercept is never penalised.

The resulting price_coefs_ matrix is (G, G): entry [g, j] is the effect
of good j's price on good g's mean utility (own-price on diagonal,
cross-price off-diagonal).  income_coefs_ is (G,): one income slope per good.
"""

import numpy as np

# Threshold above which weak-instrument / conditioning diagnostics are always
# printed regardless of the `verbose` flag.
_HIGH_K_WARN = 10


def _ridge_solve(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression: β = (XᵀX + λ·D)⁻¹ Xᵀy, intercept not penalised.

    D is a diagonal penalty matrix that is 0 for the first (intercept) column
    and 1 for all remaining columns.
    """
    K = X.shape[1]
    pen = np.eye(K)
    pen[0, 0] = 0.0          # do not penalise the intercept
    A = X.T @ X + lam * pen
    b = X.T @ y
    return np.linalg.solve(A, b)


class BLPLogitIV:
    """Logit-IV demand model — aggregate logit with Hausman instrument 2SLS.

    First stage:  for each good g, p_hat_g = ridge([1, z_0,…,z_{G-1}], p_g)
    Second stage: for each good g,
                  log(s_g / s_0) = delta_g + Σ_j alpha_{g,j} * p_hat_j
                                           + gamma_g * y   (if y supplied)

    Parameters
    ----------
    p            : (N, G)   prices of the G inside goods
    w            : (N, G+1) market shares; last column = outside option
    z            : (N, G)   Hausman IVs; z[:,g] = mean price of good g in other stores
    y            : (N,)     income / budget (optional; improves income-effect recovery)
    ridge_lambda : float    L2 penalty applied to both stages (intercept excluded)
    """

    name = "Logit-IV"

    def fit(self, p, w, z, y=None, verbose: bool = True,
            ridge_lambda: float = 1e-4):
        """Full-price 2SLS logit-IV with ridge regularisation.

        Parameters
        ----------
        p            : (N, G)   raw prices of inside goods
        w            : (N, G+1) market shares; last col = outside-option share
        z            : (N, G)   Hausman IVs (mean price across other stores)
        y            : (N,)     income / budget; if provided, added to second stage
        verbose      : bool     if True, always print diagnostics
        ridge_lambda : float    ridge penalty (both stages); intercept not penalised
        """
        N, G = p.shape
        n_inside = w.shape[1] - 1
        assert n_inside == G, (
            f"p has {G} goods but w has {w.shape[1]-1} inside goods")

        s0  = np.maximum(w[:, G:G+1], 1e-8)             # (N, 1) outside-option share
        lhs = np.log(np.maximum(w[:, :G], 1e-8) / s0)   # (N, G) log(s_g / s_0)

        # ── First stage: ridge-fit each p_g on [1, z_0, …, z_{G-1}] ─────────
        Z_aug  = np.c_[np.ones(N), z]      # (N, G+1)
        P_hat  = np.zeros((N, G))
        fs_rsq = np.zeros(G)

        for g in range(G):
            beta_g      = _ridge_solve(Z_aug, p[:, g], ridge_lambda)
            p_hat_g     = Z_aug @ beta_g
            P_hat[:, g] = p_hat_g

            ss_tot = np.sum((p[:, g] - p[:, g].mean()) ** 2)
            ss_res = np.sum((p[:, g] - p_hat_g) ** 2)
            fs_rsq[g] = 1.0 - ss_res / max(ss_tot, 1e-12)

        # ── Second stage: ridge-fit log(s_g/s_0) on P_hat [and income] ───────
        has_income = y is not None
        if has_income:
            X_aug = np.c_[np.ones(N), P_hat, np.asarray(y)]   # (N, G+2)
        else:
            X_aug = np.c_[np.ones(N), P_hat]                   # (N, G+1)

        delta        = np.zeros(G)
        price_coefs  = np.zeros((G, G))   # [g, j] = effect of p_j on good g
        income_coefs = np.zeros(G)        # [g]    = effect of income on good g
        ss_rsq       = np.zeros(G)

        for g in range(G):
            coefs_g        = _ridge_solve(X_aug, lhs[:, g], ridge_lambda)
            delta[g]       = coefs_g[0]
            price_coefs[g] = coefs_g[1:G+1]
            if has_income:
                income_coefs[g] = coefs_g[G+1]
            fitted = X_aug @ coefs_g
            ss_tot = np.sum((lhs[:, g] - lhs[:, g].mean()) ** 2)
            ss_res = np.sum((lhs[:, g] - fitted) ** 2)
            ss_rsq[g] = 1.0 - ss_res / max(ss_tot, 1e-12)

        # ── Diagnostics ───────────────────────────────────────────────────────
        # Always warn when K is large or instruments are weak; verbose prints
        # the full coefficient summary in addition.
        _warn = G > _HIGH_K_WARN or float(fs_rsq.min()) < 0.1
        if _warn or verbose:
            print(f"  [Logit-IV] K={G} goods, N={N} obs, λ={ridge_lambda}")
            print(f"  [Logit-IV] first-stage  R²: {np.round(fs_rsq, 3)}")
            print(f"  [Logit-IV] second-stage R²: {np.round(ss_rsq, 3)}")
        if _warn and float(fs_rsq.min()) < 0.1:
            print(f"  [Logit-IV] WARNING: min first-stage R²={fs_rsq.min():.3f} "
                  f"— weak instruments, price coefs may be unreliable.")
        if verbose:
            print(f"  [Logit-IV] own-price alpha (diagonal): "
                  f"{np.round(np.diag(price_coefs), 4)}")
            if has_income:
                print(f"  [Logit-IV] income coefs: {np.round(income_coefs, 4)}")
        if np.all(np.abs(np.diag(price_coefs)) < 1e-4):
            print("  [Logit-IV] WARNING: all own-price coefs ≈ 0.")

        self.intercept_        = delta
        self.alpha_            = np.diag(price_coefs)  # (G,) own-price coefs
        self.price_coefs_      = price_coefs            # (G, G) full matrix
        self.income_coefs_     = income_coefs           # (G,) income coefs
        self.has_income_       = has_income
        self.first_stage_rsq_  = fs_rsq                 # (G,)
        self.second_stage_rsq_ = ss_rsq                 # (G,)
        self.n_inside_         = G
        return self

    def predict(self, p, y=None):
        """Return (N, G+1) market shares [s_1, …, s_G, s_0].

        Mean utility:
            lgt_g = delta_g + Σ_j price_coefs_[g, j] * p_j
                            + income_coefs_[g] * y   (if fitted with income)
        Logit shares:
            s_g = exp(lgt_g) / (1 + Σ_j exp(lgt_j))
            s_0 = 1          / (1 + Σ_j exp(lgt_j))
        """
        G   = self.n_inside_
        lgt = self.intercept_[None, :] + p[:, :G] @ self.price_coefs_.T
        if self.has_income_ and y is not None:
            y_arr = np.asarray(y).reshape(-1)
            lgt  += y_arr[:, None] * self.income_coefs_[None, :]
        lgt = np.clip(lgt, -30, 30)
        eu    = np.exp(lgt)
        denom = 1.0 + eu.sum(1, keepdims=True)
        return np.c_[eu / denom, 1.0 / denom]          # (N, G+1)


class NestedLogitIV:
    """Berry (1994) Nested Logit demand model estimated via 2SLS.

    Model (one sigma per good, but shared within nest in practice):
        log(s_g) - log(s_0) = delta_g
                            + Σ_j alpha_{g,j} * p_hat_j
                            + sigma_g * log(s_{g|h(g)}_hat)
                            + gamma_g * y

    where s_{g|h} = s_g / Σ_{j ∈ h} s_j  is the within-nest conditional share.

    Endogenous regressors
    ----------------------
    - prices p_g   → instrumented by Hausman IVs z_g
    - log within-nest share log(s_{g|h(g)})
      → instrumented by sum of Hausman IVs of other goods in the same nest;
        this is the standard Berry (1994) "aggregate IV" construction

    Parameters
    ----------
    ridge_lambda : float   L2 penalty for both stages (intercept excluded)
    """

    name = "Nested Logit (2SLS)"

    def fit(self, p, w, z, nest_ids, y=None, verbose=False, ridge_lambda=1e-4):
        """Fit nested logit via 2SLS.

        Parameters
        ----------
        p        : (N, G)   prices of inside goods
        w        : (N, G+1) market shares; last col = outside option
        z        : (N, G)   Hausman IVs
        nest_ids : (G,)     integer nest assignments for each inside good
        y        : (N,)     income (optional)
        """
        N, G = p.shape
        nest_ids = np.asarray(nest_ids, dtype=int)
        nests = np.unique(nest_ids)

        s0 = np.maximum(w[:, G:G+1], 1e-8)
        s_in = np.maximum(w[:, :G], 1e-8)
        lhs = np.log(s_in / s0)   # (N, G)  log(s_g / s_0)

        # ── Within-nest shares ────────────────────────────────────────────────
        within = np.zeros((N, G))
        for h in nests:
            mask = nest_ids == h
            nest_tot = s_in[:, mask].sum(axis=1, keepdims=True)
            within[:, mask] = s_in[:, mask] / np.maximum(nest_tot, 1e-8)
        log_within = np.log(np.maximum(within, 1e-8))  # (N, G)

        # ── Instruments for within-nest shares ───────────────────────────────
        # For good g in nest h: IV = sum of Hausman IVs of other goods in h.
        z_nest = np.zeros((N, G))
        for h in nests:
            in_h = np.where(nest_ids == h)[0]
            z_nest_total = z[:, in_h].sum(axis=1)  # (N,)
            for g in in_h:
                z_nest[:, g] = z_nest_total - z[:, g]

        # ── First stage ───────────────────────────────────────────────────────
        # Instruments: [1, z (Hausman for prices), z_nest (for within-share), y]
        has_income = y is not None
        if has_income:
            Z_aug = np.c_[np.ones(N), z, z_nest, np.asarray(y)]
        else:
            Z_aug = np.c_[np.ones(N), z, z_nest]

        P_hat = np.zeros((N, G))
        W_hat = np.zeros((N, G))
        fs_rsq_p = np.zeros(G)
        fs_rsq_w = np.zeros(G)

        for g in range(G):
            beta_p = _ridge_solve(Z_aug, p[:, g], ridge_lambda)
            P_hat[:, g] = Z_aug @ beta_p
            ss_tot = np.sum((p[:, g] - p[:, g].mean()) ** 2)
            ss_res = np.sum((p[:, g] - P_hat[:, g]) ** 2)
            fs_rsq_p[g] = 1.0 - ss_res / max(ss_tot, 1e-12)

            beta_w = _ridge_solve(Z_aug, log_within[:, g], ridge_lambda)
            W_hat[:, g] = Z_aug @ beta_w
            ss_tot = np.sum((log_within[:, g] - log_within[:, g].mean()) ** 2)
            ss_res = np.sum((log_within[:, g] - W_hat[:, g]) ** 2)
            fs_rsq_w[g] = 1.0 - ss_res / max(ss_tot, 1e-12)

        if verbose or G <= _HIGH_K_WARN:
            print(f"  [NestedLogitIV] K={G}, {len(nests)} nests, N={N}")
            print(f"  [NestedLogitIV] price 1st-stage R²: mean={fs_rsq_p.mean():.3f} "
                  f"min={fs_rsq_p.min():.3f}")
            print(f"  [NestedLogitIV] within-nest 1st-stage R²: mean={fs_rsq_w.mean():.3f} "
                  f"min={fs_rsq_w.min():.3f}")

        # ── Second stage ──────────────────────────────────────────────────────
        # For good g: regress lhs_g on [1, P_hat, W_hat_g, y]
        delta_coefs = np.zeros(G)
        price_coefs = np.zeros((G, G))
        sigma_coefs = np.zeros(G)
        income_coefs = np.zeros(G)

        for g in range(G):
            if has_income:
                X_g = np.c_[np.ones(N), P_hat, W_hat[:, g:g+1], np.asarray(y)]
            else:
                X_g = np.c_[np.ones(N), P_hat, W_hat[:, g:g+1]]
            coefs = _ridge_solve(X_g, lhs[:, g], ridge_lambda)
            delta_coefs[g] = coefs[0]
            price_coefs[g] = coefs[1:G+1]
            sigma_coefs[g] = coefs[G+1]
            if has_income:
                income_coefs[g] = coefs[G+2]

        self.delta_        = delta_coefs
        self.price_coefs_  = price_coefs
        self.sigma_        = np.clip(sigma_coefs, 0.0, 0.999)  # keep in (0, 1)
        self.income_coefs_ = income_coefs
        self.has_income_   = has_income
        self.n_inside_     = G
        self.nest_ids_     = nest_ids
        self.first_stage_rsq_price_  = fs_rsq_p
        self.first_stage_rsq_within_ = fs_rsq_w
        return self

    def predict(self, p, y=None, max_iter=20, tol=1e-6):
        """Return (N, G+1) market shares via fixed-point iteration on nest shares.

        Starting from logit predictions (sigma=0), iteratively refine by adding
        the sigma * log(s_{g|h}) term until convergence.
        """
        G = self.n_inside_
        nests = np.unique(self.nest_ids_)
        nest_ids = self.nest_ids_

        # Mean-utility (price + income, no sigma term yet)
        lgt_base = self.delta_[None, :] + p[:, :G] @ self.price_coefs_.T
        if self.has_income_ and y is not None:
            lgt_base += np.asarray(y).reshape(-1, 1) * self.income_coefs_[None, :]

        # Initialise with simple logit shares
        lgt = np.clip(lgt_base, -30, 30)
        eu = np.exp(lgt)
        s = eu / (1.0 + eu.sum(axis=1, keepdims=True))  # (N, G)

        for _ in range(max_iter):
            # Recompute within-nest shares
            within = np.zeros_like(s)
            for h in nests:
                mask = nest_ids == h
                nest_tot = s[:, mask].sum(axis=1, keepdims=True)
                within[:, mask] = s[:, mask] / np.maximum(nest_tot, 1e-8)
            log_w = np.log(np.maximum(within, 1e-8))

            lgt_new = np.clip(lgt_base + self.sigma_[None, :] * log_w, -30, 30)
            eu_new = np.exp(lgt_new)
            s_new = eu_new / (1.0 + eu_new.sum(axis=1, keepdims=True))

            if np.max(np.abs(s_new - s)) < tol:
                s = s_new
                break
            s = s_new

        s0 = 1.0 / (1.0 + np.exp(lgt_new).sum(axis=1, keepdims=True))
        return np.c_[s, s0]   # (N, G+1)


class BLPBench(BLPLogitIV):
    """Simulation benchmark alias keeping original non-chaining fit API."""

    name = "Logit-IV"

    def fit(self, p, w, z, y=None, verbose: bool = False,
            ridge_lambda: float = 1e-4):
        super().fit(p, w, z, y=y, verbose=verbose, ridge_lambda=ridge_lambda)
        return self
