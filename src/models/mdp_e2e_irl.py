"""
Sequential habit/state utilities and Window demand model helpers.

This module provides:
  - `compute_xbar_e2e`: sequential EWMA habit-stock construction for fixed δ
  - `build_window_features`: lag-window feature matrix construction
  - `train_mdp_e2e`: fixed-δ habit-model trainer helper
  - `WindowND` + `train_window_irl`: window-based neural demand model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: sequential x̄ computation with differentiable δ
# ─────────────────────────────────────────────────────────────────────────────

def compute_xbar_e2e(delta, log_q_tensor, store_ids=None, init_val=None):
    """Compute the habit stock x̄ for all N observations, keeping the
    computation graph live so that gradients flow back to *delta*.

    The update rule is
        x̄_0  = init_val
        x̄_t  = δ · x̄_{t-1} + (1−δ) · q_{t-1}
    which matches the HabitFormationConsumer DGP (decay applied to the
    running average, not to individual observations).

    Parameters
    ----------
    delta        : scalar Tensor — learned decay ∈ (0, 1), from sigmoid.
    log_q_tensor : Tensor (N, G) — log-quantities (or log-shares) in
                   *sequential / temporal order*.
    store_ids    : array-like (N,) of int — resets x̄ at store boundaries.
                   Pass None for a single continuous sequence (simulation).
    init_val     : Tensor (G,) — initial habit; defaults to mean of log_q.

    Returns
    -------
    xbar : Tensor (N, G) with computation graph through delta.
    """
    N, G = log_q_tensor.shape
    if init_val is None:
        init_val = log_q_tensor.detach().mean(0)   # (G,) — no grad needed for init

    xb = init_val.clone()
    xbar_list: list = []

    for i in range(N):
        # ── Reset at store boundary ──────────────────────────────────────────
        if store_ids is not None and i > 0 and store_ids[i] != store_ids[i - 1]:
            xb = init_val.clone()

        xbar_list.append(xb)                                    # record x̄ entering t
        xb = delta * xb + (1.0 - delta) * log_q_tensor[i]      # EWMA update

    return torch.stack(xbar_list)  # (N, G)


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: window feature matrix for WindowND
# ─────────────────────────────────────────────────────────────────────────────

def build_window_features(log_p_all, log_y_all, log_q_all,
                          window: int, store_ids=None):
    """Build the (N, in_dim) feature matrix for WindowND.

    For each observation i the feature vector is:
        [log_p_i, log_y_i,
         log_p_{i-1}, log_q_{i-1},
         log_p_{i-2}, log_q_{i-2},
         ...
         log_p_{i-L}, log_q_{i-L}]

    Missing history (store boundaries or beginning of sequence) is filled
    with the column-wise global mean of log_p / log_q.

    Parameters
    ----------
    log_p_all  : (N, G) ndarray — log prices in sequential order.
    log_y_all  : (N,)   ndarray — log incomes.
    log_q_all  : (N, G) ndarray — log quantities (or log-shares).
    window     : int — number of lag periods L.
    store_ids  : (N,) array or None — resets history at boundaries.

    Returns
    -------
    features : (N, G + 1 + window * 2 * G) ndarray
    """
    N, G = log_p_all.shape
    init_p = log_p_all.mean(0)    # (G,) fallback for missing history
    init_q = log_q_all.mean(0)    # (G,)
    out_dim = G + 1 + window * 2 * G
    features = np.zeros((N, out_dim), dtype=np.float32)

    for i in range(N):
        parts = [log_p_all[i], np.array([log_y_all[i]])]
        for lag in range(1, window + 1):
            j = i - lag
            if (j >= 0
                    and (store_ids is None or store_ids[j] == store_ids[i])):
                parts.extend([log_p_all[j], log_q_all[j]])
            else:
                parts.extend([init_p, init_q])
        features[i] = np.concatenate(parts)

    return features


# ─────────────────────────────────────────────────────────────────────────────
#  Model 2: WindowND
# ─────────────────────────────────────────────────────────────────────────────

class WindowND(nn.Module):
    """Free-form sequential IRL over a fixed lookback window.

    Conditions on the last *window* periods of (log price, log quantity)
    together with the current (log price, log income).  No parametric
    assumption is imposed on how history enters — the MLP learns whatever
    dynamic pattern (habit, brand loyalty, seasonal) exists in the data.

    Input layout (per observation):
        [log_p_t (G), log_y_t (1),
         log_p_{t-1} (G), log_q_{t-1} (G),
         ...
         log_p_{t-L} (G), log_q_{t-L} (G)]   total dim = G+1+2GL
    """
    name = "Neural Demand (window)"

    def __init__(self, n_goods: int = 3, hidden_dim: int = 256, window: int = 4):
        super().__init__()
        self.n_goods = n_goods
        self.window = window
        in_dim = n_goods + 1 + window * 2 * n_goods
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_goods),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    @property
    def beta(self):
        return torch.tensor(1.0)

    def forward(self, x):
        """x : (B, in_dim) — pre-built window feature vector."""
        return torch.softmax(self.net(x), dim=1)

    def _slutsky_symmetry_penalty(self, x):
        """True Slutsky symmetry penalty with income-effect correction.

        Penalizes asymmetry of M_ij = ∂w_i/∂log p_j + w_j · ∂w_i/∂log y,
        where log prices occupy the first G dims of x and log income is dim G.
        """
        G = self.n_goods
        x_d = x.detach().clone().requires_grad_(True)
        w = self.forward(x_d)  # (B, G)
        # Price Jacobian: J[b, i, j] = ∂w_i/∂log p_j  (j indexes first G dims)
        J = torch.stack([
            torch.autograd.grad(
                w[:, i].sum(), x_d, create_graph=True, retain_graph=True
            )[0][:, :G]
            for i in range(G)
        ], dim=1)  # (B, G, G)
        # Income derivatives: d[b, i] = ∂w_i/∂log y  (dim G of x)
        d = torch.stack([
            torch.autograd.grad(
                w[:, i].sum(), x_d, create_graph=True, retain_graph=True
            )[0][:, G]
            for i in range(G)
        ], dim=1)  # (B, G)
        M = J + d.unsqueeze(2) * w.unsqueeze(1)  # (B, G, G)
        return ((M - M.transpose(1, 2)) ** 2).mean()

    def slutsky_penalty(self, x):
        return self._slutsky_symmetry_penalty(x)

    def slutsky(self, x):
        return self._slutsky_symmetry_penalty(x)

# ─────────────────────────────────────────────────────────────────────────────
#  Fixed-δ habit trainer helper (backward-compatible name)
# ─────────────────────────────────────────────────────────────────────────────

def train_mdp_e2e(model, prices, income, w_expert, log_q_seq,
                  store_ids=None, epochs=3000, lr=5e-4, batch_size=256,
                  lam_mono=0.2, lam_slut=0.1, slut_start_frac=0.25,
                  xbar_recompute_every=1, delta_lr_scale=0.1, device="cpu",
                  verbose=False, tag="", store_idx=None, cache_dir=None,
                  force_retrain=False):
    """Train fixed-δ habit model with xbar computed from `log_q_seq`.

    Backward-compatible wrapper kept under the historical name
    ``train_mdp_e2e``.  With δ fixed, xbar is computed once before training.
    """
    from .simulation_train import train_neural_irl

    lq_t = torch.tensor(log_q_seq, dtype=torch.float32, device=device)
    d_t = torch.tensor(float(model.delta.item()), dtype=torch.float32, device=device)
    with torch.no_grad():
        xb = compute_xbar_e2e(d_t, lq_t, store_ids=store_ids).cpu().numpy()

    q_prev = np.roll(log_q_seq, 1, axis=0)
    q_prev[0] = log_q_seq[0]
    if store_ids is not None:
        for i in range(1, len(store_ids)):
            if store_ids[i] != store_ids[i - 1]:
                q_prev[i] = log_q_seq[i]

    model, hist = train_neural_irl(
        model, prices, income, w_expert,
        epochs=epochs, lr=lr, batch_size=batch_size,
        lam_mono=lam_mono, lam_slut=lam_slut, slut_start_frac=slut_start_frac,
        xb_prev_data=np.exp(xb),
        q_prev_data=np.exp(q_prev),
        v_hat_data=None,
        device=device, verbose=verbose, tag=tag,
        cache_dir=cache_dir, force_retrain=force_retrain,
    )
    return model, hist



# ─────────────────────────────────────────────────────────────────────────────
#  Training: WindowND
# ─────────────────────────────────────────────────────────────────────────────

def train_window_irl(model, window_feats, w_expert,
                     epochs=3000, lr=5e-4, batch_size=256,
                     lam_mono=0.2, lam_slut=0.1, slut_start_frac=0.25,
                     device="cpu", verbose=False, tag="", cache_dir=None,
                     force_retrain=False):
    """Train WindowND on pre-built window feature matrix.

    Includes caching: if a model with the same 'tag' exists in
    cache_dir, it is loaded instead of retrained.
    """
    model = model.to(device)

    # ── Cache Check ──────────────────────────────────────────────────────────
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        safe_tag = tag.replace(" ", "_").replace("=", "").replace("-", "_")
        cache_path = os.path.join(cache_dir, f"{safe_tag}.pt")

        if os.path.exists(cache_path) and not force_retrain:
            if verbose:
                print(f"    [Cache] Loading {tag} from {cache_path}")
            try:
                checkpoint = torch.load(cache_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                return model, checkpoint.get("history", [])
            except Exception as e:
                print(f"    [Cache] Failed to load {cache_path}: {e}. Retraining.")

    # Adam (not SGD): the scalar log_beta parameter benefits from adaptive
    # per-parameter step sizes, preventing the β oscillations seen with SGD.
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    G = model.n_goods
    N = len(w_expert)
    slut_start = int(epochs * slut_start_frac)

    X = torch.tensor(window_feats, dtype=torch.float32).to(device)
    W = torch.tensor(w_expert,     dtype=torch.float32).to(device)

    best_kl, best_state, history = float("inf"), None, []
    _last_full_kl = float("inf")   # cache full-data KL for history logging

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N, device=device)[:batch_size]
        x_b, w_b = X[idx], W[idx]

        opt.zero_grad()
        wp = model(x_b)
        loss_kl = nn.KLDivLoss(reduction="batchmean")(torch.log(wp + 1e-10), w_b)

        # Own-price monotonicity: penalize positive ∂w_g/∂log p_g for each good g.
        x_d = x_b.detach().clone().requires_grad_(True)
        wm = model(x_d)
        loss_mono = torch.tensor(0.0, device=device)
        for _g in range(G):
            _own = torch.autograd.grad(
                wm[:, _g].sum(), x_d, create_graph=True, retain_graph=True
            )[0][:, _g]
            loss_mono = loss_mono + torch.mean(torch.clamp(_own, min=0))
        loss_mono = loss_mono / G

        loss_slut = torch.tensor(0.0, device=device)
        if ep >= slut_start:
            sub = torch.randperm(N, device=device)[:64]
            loss_slut = model.slutsky_penalty(X[sub])

        loss = loss_kl + lam_mono * loss_mono + lam_slut * loss_slut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                kl_full = nn.KLDivLoss(reduction="batchmean")(
                    torch.log(model(X) + 1e-10), W).item()
            _last_full_kl = kl_full   # cache for history logging
            if kl_full < best_kl:
                best_kl = kl_full
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        # Log full-data KL (not noisy mini-batch KL) for clean convergence plots
        if ep % 100 == 0:
            history.append({"epoch": ep, "kl": _last_full_kl, "delta": float("nan")})
            if verbose:
                print(f"    [{tag}] ep {ep:4d} | KL={_last_full_kl:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    elif not np.isfinite(best_kl):
        model.eval()
        with torch.no_grad():
            best_kl = nn.KLDivLoss(reduction="batchmean")(
                torch.log(model(X) + 1e-10), W).item()
        model.train()

    # ── Save to Cache ────────────────────────────────────────────────────────
    if cache_dir is not None:
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": history,
            }, cache_path)
        except Exception as e:
            print(f"    [Cache] Failed to save {cache_path}: {e}")

    print(f"    [{tag}] KL_min={best_kl:.6f}")
    model.eval()
    return model, history
