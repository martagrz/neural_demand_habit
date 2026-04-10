"""Shared Neural IRL model used by both pipelines."""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  Base model (no store effects)
# ─────────────────────────────────────────────────────────────────────────────

class StaticND(nn.Module):
    name = "Neural Demand (static)"

    def __init__(self, h=256, n_goods=3, hidden_dim=None, n_cf=0):
        """
        Parameters
        ----------
        n_cf : int
            Number of control-function residuals to append to the state.
            Set to n_goods when using the CF endogeneity correction;
            0 (default) preserves the original behaviour.
        """
        super().__init__()
        self.n_goods = n_goods
        self.n_cf    = n_cf
        hdim = hidden_dim if hidden_dim is not None else h
        self.net = nn.Sequential(
            nn.Linear(n_goods + 1 + n_cf, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim // 2),
            nn.SiLU(),
            nn.Linear(hdim // 2, n_goods),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    @property
    def beta(self):
        return torch.tensor(1.0)

    def forward(self, log_p, log_y, v_hat=None):
        """
        log_p : (B, G)   – log prices
        log_y : (B, 1)   – log income
        v_hat : (B, n_cf) or None – first-stage CF residuals.
                Pass zeros (or None) for structural / counterfactual evaluation.
        """
        inp = [log_p, log_y]
        if v_hat is not None:
            inp.append(v_hat)
        return torch.softmax(self.net(torch.cat(inp, dim=1)), dim=1)

    def _slutsky_symmetry_penalty(self, log_p, log_y, v_hat=None, max_goods=None):
        """True Slutsky symmetry penalty with income-effect correction.

        Penalizes asymmetry of the corrected matrix
            M_ij = ∂w_i/∂log p_j + w_j · ∂w_i/∂log y,
        which is symmetric iff the Slutsky substitution matrix is symmetric
        (i.e. iff the demand system is consistent with utility maximisation).
        The plain Jacobian-symmetry condition ∂w_i/∂log p_j = ∂w_j/∂log p_i
        holds only under homotheticity (zero income effects); this formulation
        is correct for general non-homothetic preferences.

        When max_goods < G, a random square submatrix of M is used, giving an
        unbiased stochastic estimate of the full symmetry penalty.
        """
        lp_d = log_p.detach().requires_grad_(True)
        ly_d = log_y.detach().requires_grad_(True)
        w = self.forward(lp_d, ly_d, v_hat)
        G = self.n_goods
        n_s = min(G, int(max_goods)) if max_goods is not None else G
        idx = (
            torch.randperm(G, device=lp_d.device)[:n_s]
            if n_s < G else torch.arange(G, device=lp_d.device)
        )
        # Rows of price Jacobian for sampled goods: J_rows[b, k, j] = ∂w_{idx[k]}/∂log p_j
        J_rows = torch.stack([
            torch.autograd.grad(
                w[:, idx[k]].sum(), lp_d, create_graph=True, retain_graph=True
            )[0]
            for k in range(n_s)
        ], dim=1)  # (B, n_s, G)
        J_sub = J_rows[:, :, idx]  # (B, n_s, n_s) — square submatrix
        # Income derivatives for sampled goods: d_sub[b, k] = ∂w_{idx[k]}/∂log y
        d_sub = torch.stack([
            torch.autograd.grad(
                w[:, idx[k]].sum(), ly_d, create_graph=True, retain_graph=True
            )[0].squeeze(1)
            for k in range(n_s)
        ], dim=1)  # (B, n_s)
        w_sub = w[:, idx]   # (B, n_s)
        M_sub = J_sub + d_sub.unsqueeze(2) * w_sub.unsqueeze(1)  # (B, n_s, n_s)
        return ((M_sub - M_sub.transpose(1, 2)) ** 2).mean()

    # Dominicks API
    def slutsky(self, lp, ly, v_hat=None, max_goods=None):
        return self._slutsky_symmetry_penalty(lp, ly, v_hat, max_goods=max_goods)

    # Simulation API
    def slutsky_penalty(self, log_p, log_y, v_hat=None, max_goods=None):
        return self._slutsky_symmetry_penalty(log_p, log_y, v_hat, max_goods=max_goods)


# ─────────────────────────────────────────────────────────────────────────────
#  Store-fixed-effects variant (Dominick's pipeline only)
# ─────────────────────────────────────────────────────────────────────────────

class StaticND_FE(nn.Module):
    """Neural IRL with store fixed effects via learned dense embeddings.

    Each store gets a learnable embedding vector of dimension *emb_dim*.
    The embedding is concatenated with the standard (log_p, log_y) input
    before the hidden layers.  This allows the demand network to absorb
    time-invariant store heterogeneity (demographics, competition, shelf
    layout) that is not captured by the price/income features alone.

    Parameters
    ----------
    n_stores  : int — number of unique stores in the dataset.
    emb_dim   : int — embedding dimension (default 8; suitable for ~100 stores).
    n_cf      : int — number of CF residuals appended to state (0 = disabled).
    """
    name = "Neural Demand (static, FE)"

    def __init__(self, h: int = 256, n_goods: int = 3, hidden_dim: int = None,
                 n_stores: int = 100, emb_dim: int = 8, n_cf: int = 0):
        super().__init__()
        self.n_goods = n_goods
        self.n_cf    = n_cf
        hdim = hidden_dim if hidden_dim is not None else h
        self.store_emb = nn.Embedding(n_stores, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(n_goods + 1 + emb_dim + n_cf, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim // 2),
            nn.SiLU(),
            nn.Linear(hdim // 2, n_goods),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.normal_(self.store_emb.weight, std=0.01)   # small init → near-zero start

    @property
    def beta(self):
        return torch.tensor(1.0)

    def forward(self, log_p, log_y, store_idx, v_hat=None):
        """
        log_p     : (B, G)    – log prices
        log_y     : (B, 1)    – log income
        store_idx : (B,)      – integer tensor of store indices ∈ {0, …, n_stores-1}
        v_hat     : (B, n_cf) or None – CF residuals; None → no endogeneity correction
        """
        emb = self.store_emb(store_idx)   # (B, emb_dim)
        inp = [log_p, log_y, emb]
        if v_hat is not None:
            inp.append(v_hat)
        return torch.softmax(self.net(torch.cat(inp, dim=1)), dim=1)

    def _slutsky_symmetry_penalty(self, log_p, log_y, store_idx, v_hat=None, max_goods=None):
        """True Slutsky symmetry penalty with income-effect correction.

        Penalizes asymmetry of M_ij = ∂w_i/∂log p_j + w_j · ∂w_i/∂log y.
        Store embedding is treated as fixed state (not differentiated).
        When max_goods < G, uses a random square submatrix (same as StaticND).
        """
        lp_d = log_p.detach().requires_grad_(True)
        ly_d = log_y.detach().requires_grad_(True)
        w = self.forward(lp_d, ly_d, store_idx, v_hat)
        G = self.n_goods
        n_s = min(G, int(max_goods)) if max_goods is not None else G
        idx = (
            torch.randperm(G, device=lp_d.device)[:n_s]
            if n_s < G else torch.arange(G, device=lp_d.device)
        )
        J_rows = torch.stack([
            torch.autograd.grad(
                w[:, idx[k]].sum(), lp_d, create_graph=True, retain_graph=True
            )[0]
            for k in range(n_s)
        ], dim=1)  # (B, n_s, G)
        J_sub = J_rows[:, :, idx]
        d_sub = torch.stack([
            torch.autograd.grad(
                w[:, idx[k]].sum(), ly_d, create_graph=True, retain_graph=True
            )[0].squeeze(1)
            for k in range(n_s)
        ], dim=1)  # (B, n_s)
        w_sub = w[:, idx]
        M_sub = J_sub + d_sub.unsqueeze(2) * w_sub.unsqueeze(1)
        return ((M_sub - M_sub.transpose(1, 2)) ** 2).mean()

    def slutsky(self, lp, ly, store_idx, v_hat=None, max_goods=None):
        return self._slutsky_symmetry_penalty(lp, ly, store_idx, v_hat, max_goods=max_goods)

    def slutsky_penalty(self, log_p, log_y, store_idx, v_hat=None, max_goods=None):
        return self._slutsky_symmetry_penalty(log_p, log_y, store_idx, v_hat, max_goods=max_goods)
