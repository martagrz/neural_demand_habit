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

    def _jacobian_symmetry_penalty(self, log_p, log_y, v_hat=None):
        lp_d = log_p.detach().requires_grad_(True)
        w = self.forward(lp_d, log_y, v_hat)
        rows = [
            torch.autograd.grad(w[:, i].sum(), lp_d, create_graph=True, retain_graph=True)[0].unsqueeze(2)
            for i in range(self.n_goods)
        ]
        J = torch.cat(rows, 2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    # Dominicks API
    def slutsky(self, lp, ly, v_hat=None):
        return self._jacobian_symmetry_penalty(lp, ly, v_hat)

    # Simulation API
    def slutsky_penalty(self, log_p, log_y, v_hat=None):
        return self._jacobian_symmetry_penalty(log_p, log_y, v_hat)


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

    def _jacobian_symmetry_penalty(self, log_p, log_y, store_idx, v_hat=None):
        """Slutsky symmetry w.r.t. log_p only; store embedding is treated as fixed."""
        lp_d = log_p.detach().requires_grad_(True)
        w = self.forward(lp_d, log_y, store_idx, v_hat)
        rows = [
            torch.autograd.grad(
                w[:, i].sum(), lp_d, create_graph=True, retain_graph=True
            )[0].unsqueeze(2)
            for i in range(self.n_goods)
        ]
        J = torch.cat(rows, dim=2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    def slutsky(self, lp, ly, store_idx, v_hat=None):
        return self._jacobian_symmetry_penalty(lp, ly, store_idx, v_hat)

    def slutsky_penalty(self, log_p, log_y, store_idx, v_hat=None):
        return self._jacobian_symmetry_penalty(log_p, log_y, store_idx, v_hat)
