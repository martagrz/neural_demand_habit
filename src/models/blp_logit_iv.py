"""BLP logit-IV model implementation.

The model treats the last column of w as the **outside option** share.
With the Dominick's data, this is the "OTHER" category (all analgesics not
classified as ASP, ACET, or IBU), giving a proper 3-inside-good / 1-outside
specification.  For simulations there is no genuine outside good; the caller
should add a small constant outside share (e.g. 1 %) before calling fit().

Identification follows BLP (1995) with full-price 2SLS:
  - First stage  : for each good g, regress p_g on [1, z_0, …, z_{G-1}]
                   (all G Hausman IVs) to obtain fitted prices p_hat_g.
  - Second stage : for each good g, regress log(s_g / s_0) on
                   [1, p_hat_0, …, p_hat_{G-1}] (all G fitted prices),
                   yielding own- and cross-price utility coefficients.

The resulting price_coefs_ matrix is (G, G): entry [g, j] is the effect
of good j's price on good g's mean utility (own-price on diagonal,
cross-price off-diagonal).
"""

import numpy as np


class BLPLogitIV:
    """BLP logit with Hausman IV — full-price 2SLS.

    First stage:  for each good g, p_hat_g = OLS([1, z_0,…,z_{G-1}], p_g)
    Second stage: for each good g,
                  log(s_g / s_0) = delta_g + Σ_j alpha_{g,j} * p_hat_j

    Parameters
    ----------
    p : (N, G)   prices of the G inside goods
    w : (N, G+1) market shares; last column = outside option
    z : (N, G)   Hausman IVs; z[:,g] = mean price of good g in other stores
    """

    name = "BLP (IV)"

    def fit(self, p, w, z, verbose: bool = True):
        """Full-price 2SLS BLP.

        Parameters
        ----------
        p       : (N, G)   raw prices of inside goods
        w       : (N, G+1) market shares; last col = outside-option share
        z       : (N, G)   Hausman IVs (mean price across other stores)
        verbose : bool     if True, print diagnostics
        """
        N, G = p.shape
        n_inside = w.shape[1] - 1
        assert n_inside == G, (
            f"p has {G} goods but w has {w.shape[1]-1} inside goods")

        s0 = np.maximum(w[:, G:G+1], 1e-8)            # (N, 1) outside-option share
        y  = np.log(np.maximum(w[:, :G], 1e-8) / s0)  # (N, G) log(s_g / s_0)

        # ── First stage: fit each p_g on [1, z_0, …, z_{G-1}] ───────────────
        Z_aug  = np.c_[np.ones(N), z]      # (N, G+1)
        P_hat  = np.zeros((N, G))
        fs_rsq = np.zeros(G)

        for g in range(G):
            beta_g      = np.linalg.lstsq(Z_aug, p[:, g], rcond=None)[0]
            p_hat_g     = Z_aug @ beta_g
            P_hat[:, g] = p_hat_g

            ss_tot = np.sum((p[:, g] - p[:, g].mean()) ** 2)
            ss_res = np.sum((p[:, g] - p_hat_g) ** 2)
            fs_rsq[g] = 1.0 - ss_res / max(ss_tot, 1e-12)

        # ── Second stage: for each good g regress log(s_g/s_0) on P_hat ─────
        X_aug = np.c_[np.ones(N), P_hat]               # (N, G+1)

        delta       = np.zeros(G)
        price_coefs = np.zeros((G, G))  # [g, j] = effect of p_j on good g

        for g in range(G):
            coefs_g        = np.linalg.lstsq(X_aug, y[:, g], rcond=None)[0]
            delta[g]       = coefs_g[0]
            price_coefs[g] = coefs_g[1:]

        if verbose:
            print(f"  [BLP] first-stage R²: {np.round(fs_rsq, 3)}")
            print(f"  [BLP] own-price alpha (diagonal): "
                  f"{np.round(np.diag(price_coefs), 4)}")
            if np.all(np.abs(np.diag(price_coefs)) < 1e-4):
                print("  [BLP] WARNING: all own-price coefs ≈ 0.")

        self.intercept_       = delta
        self.alpha_           = np.diag(price_coefs)  # (G,) own-price coefs
        self.price_coefs_     = price_coefs            # (G, G) full matrix
        self.first_stage_rsq_ = fs_rsq
        self.n_inside_        = G
        return self

    def predict(self, p):
        """Return (N, G+1) market shares [s_1, …, s_G, s_0].

        Mean utility:
            lgt_g = delta_g + Σ_j price_coefs_[g, j] * p_j
        Logit shares:
            s_g = exp(lgt_g) / (1 + Σ_j exp(lgt_j))
            s_0 = 1          / (1 + Σ_j exp(lgt_j))
        """
        G   = self.n_inside_
        lgt = self.intercept_[None, :] + p[:, :G] @ self.price_coefs_.T
        lgt = np.clip(lgt, -30, 30)
        eu    = np.exp(lgt)
        denom = 1.0 + eu.sum(1, keepdims=True)
        return np.c_[eu / denom, 1.0 / denom]          # (N, G+1)


class BLPBench(BLPLogitIV):
    """Simulation benchmark alias keeping original non-chaining fit API."""

    name = "BLP (IV)"

    def fit(self, p, w, z, verbose: bool = False):
        super().fit(p, w, z, verbose=verbose)
        return self
