"""Shared MDP-aware Neural IRL model used by both pipelines."""

import numpy as np
import torch
import torch.nn as nn


class HabitND(nn.Module):
    name = "Neural Demand (habit)"

    def __init__(self, h=256, n_goods=3, hidden_dim=None, delta_init=0.5, n_cf=0):
        """
        Parameters
        ----------
        h / hidden_dim : int
            Width of the hidden layers.
        n_goods : int
            Number of goods in the demand system.
        delta_init : float in (0, 1)
            Fixed habit-decay parameter δ used throughout training/evaluation.
        n_cf : int
            Number of control-function residuals appended to the state.
            Set to n_goods when using the CF endogeneity correction;
            0 (default) preserves the original behaviour.
        """
        super().__init__()
        self.n_goods = n_goods
        self.n_cf    = n_cf
        hdim = hidden_dim if hidden_dim is not None else h
        self.net = nn.Sequential(
            nn.Linear(n_goods * 2 + 1 + n_cf, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim // 2),
            nn.SiLU(),
            nn.Linear(hdim // 2, n_goods),
        )
        # Fixed habit-decay parameter (no learning).
        delta_init = float(np.clip(delta_init, 1e-3, 1.0 - 1e-3))
        self.register_buffer("_delta", torch.tensor(delta_init, dtype=torch.float32))

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    # ------------------------------------------------------------------ #
    #  Learned parameters                                                  #
    # ------------------------------------------------------------------ #

    @property
    def beta(self):
        return torch.tensor(1.0)

    @property
    def delta(self):
        """Fixed habit-decay rate δ ∈ (0, 1)."""
        return self._delta

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, log_p, log_y, log_xb_prev, log_q_prev, v_hat=None):
        """
        Compute predicted budget shares.

        Parameters
        ----------
        log_p      : Tensor (B, G)  – log prices
        log_y      : Tensor (B, 1)  – log income
        log_xb_prev : Tensor (B, G) – pre-processed log of the previous-period
                      habit stock  x̄_{t-1}.
        log_q_prev  : Tensor (B, G) – pre-processed log of the previous-period
                      quantities q_{t-1}.
        v_hat       : Tensor (B, n_cf) or None – first-stage CF residuals.
                      Pass zeros (or None) for structural evaluation.

        The model forms its working habit measure as

            log_x̄_t^eff = δ · log_xb_prev + (1 − δ) · log_q_prev

        i.e. a convex combination in log-space with fixed δ.
        """
        delta = self.delta
        xb_input = delta * log_xb_prev + (1.0 - delta) * log_q_prev
        inp = torch.cat([log_p, log_y, xb_input], dim=1)
        if v_hat is not None:
            inp = torch.cat([inp, v_hat], dim=1)
        return torch.softmax(self.net(inp), dim=1)

    # ------------------------------------------------------------------ #
    #  Slutsky symmetry penalty                                            #
    # ------------------------------------------------------------------ #

    def _jacobian_symmetry_penalty(self, log_p, log_y, log_xb_prev, log_q_prev,
                                   v_hat=None):
        lp_d = log_p.detach().requires_grad_(True)
        w = self.forward(lp_d, log_y, log_xb_prev, log_q_prev, v_hat)
        rows = []
        for i in range(self.n_goods):
            grad = torch.autograd.grad(
                w[:, i].sum(), lp_d, create_graph=True, retain_graph=True
            )[0]
            if grad is None:
                 grad = torch.zeros_like(lp_d)
            rows.append(grad.unsqueeze(2))
        
        J = torch.cat(rows, dim=2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    # Dominicks API
    def slutsky(self, lp, ly, log_xb_prev, log_q_prev, v_hat=None):
        return self._jacobian_symmetry_penalty(lp, ly, log_xb_prev, log_q_prev, v_hat)

    # Simulation API
    def slutsky_penalty(self, log_p, log_y, log_xb_prev, log_q_prev, v_hat=None):
        return self._jacobian_symmetry_penalty(log_p, log_y, log_xb_prev, log_q_prev, v_hat)


# ─────────────────────────────────────────────────────────────────────────────
#  Store-fixed-effects variant (Dominick's pipeline only)
# ─────────────────────────────────────────────────────────────────────────────

class HabitND_FE(nn.Module):
    """MDP Neural IRL with store fixed effects via learned dense embeddings.

    Same habit-formation model as HabitND but with an additional
    store embedding concatenated to the (log_p, log_y, habit_input) state.

    Parameters
    ----------
    n_stores  : int — number of unique stores.
    emb_dim   : int — embedding dimension (default 8).
    delta_init : float — initial habit-decay δ ∈ (0, 1).
    n_cf      : int — number of CF residuals appended to state (0 = disabled).
    """
    name = "Neural Demand (habit, FE)"

    def __init__(self, h: int = 256, n_goods: int = 3, hidden_dim: int = None,
                 delta_init: float = 0.5, n_stores: int = 100, emb_dim: int = 8,
                 n_cf: int = 0):
        super().__init__()
        self.n_goods = n_goods
        self.n_cf    = n_cf
        hdim = hidden_dim if hidden_dim is not None else h
        self.store_emb = nn.Embedding(n_stores, emb_dim)
        # input: log_p (G) + log_y (1) + habit_input (G) + store_emb (emb_dim) + v_hat (n_cf)
        self.net = nn.Sequential(
            nn.Linear(n_goods * 2 + 1 + emb_dim + n_cf, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim // 2),
            nn.SiLU(),
            nn.Linear(hdim // 2, n_goods),
        )
        delta_init = float(np.clip(delta_init, 1e-3, 1.0 - 1e-3))
        self.register_buffer("_delta", torch.tensor(delta_init, dtype=torch.float32))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.normal_(self.store_emb.weight, std=0.01)

    @property
    def beta(self):
        return torch.tensor(1.0)

    @property
    def delta(self):
        return self._delta

    def forward(self, log_p, log_y, log_xb_prev, log_q_prev, store_idx, v_hat=None):
        """
        log_p       : (B, G)    – log prices
        log_y       : (B, 1)    – log income
        log_xb_prev : (B, G)    – previous habit stock
        log_q_prev  : (B, G)    – previous log-share
        store_idx   : (B,)      – integer store indices
        v_hat       : (B, n_cf) or None – CF residuals
        """
        delta = self.delta
        xb_input = delta * log_xb_prev + (1.0 - delta) * log_q_prev
        emb = self.store_emb(store_idx)   # (B, emb_dim)
        inp = [log_p, log_y, xb_input, emb]
        if v_hat is not None:
            inp.append(v_hat)
        return torch.softmax(self.net(torch.cat(inp, dim=1)), dim=1)

    def _jacobian_symmetry_penalty(self, log_p, log_y, log_xb_prev, log_q_prev,
                                   store_idx, v_hat=None):
        lp_d = log_p.detach().requires_grad_(True)
        w = self.forward(lp_d, log_y, log_xb_prev, log_q_prev, store_idx, v_hat)
        rows = [
            torch.autograd.grad(
                w[:, i].sum(), lp_d, create_graph=True, retain_graph=True
            )[0].unsqueeze(2)
            for i in range(self.n_goods)
        ]
        J = torch.cat(rows, dim=2)
        return ((J - J.transpose(1, 2)) ** 2).mean()

    def slutsky(self, lp, ly, log_xb_prev, log_q_prev, store_idx, v_hat=None):
        return self._jacobian_symmetry_penalty(lp, ly, log_xb_prev, log_q_prev,
                                               store_idx, v_hat)

    def slutsky_penalty(self, log_p, log_y, log_xb_prev, log_q_prev, store_idx,
                        v_hat=None):
        return self._jacobian_symmetry_penalty(log_p, log_y, log_xb_prev, log_q_prev,
                                               store_idx, v_hat)
