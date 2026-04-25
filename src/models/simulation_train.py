"""Simulation neural training loop."""

import torch
import torch.nn as nn
import torch.optim as optim
import os


def _safe_tag(tag: str) -> str:
    return tag.replace(" ", "_").replace("=", "").replace("-", "_")


def _candidate_cache_paths(cache_dir: str, tag: str) -> list[str]:
    return [os.path.join(cache_dir, f"{_safe_tag(tag)}.pt")]


def _adapt_legacy_state_dict(sd: dict) -> dict:
    if "_delta" not in sd and "log_delta" in sd:
        with torch.no_grad():
            sd["_delta"] = torch.sigmoid(sd["log_delta"]).detach().clone()
        sd.pop("log_delta", None)
    return sd


def train_neural_irl(
    model,
    prices,
    income,
    w_expert,
    epochs=4000,
    lr=5e-3,
    batch_size=256,
    lam_mono=0.3,
    lam_slut=0.1,
    slut_start_frac=0.25,
    xb_prev_data=None,
    q_prev_data=None,
    # Legacy alias kept for backward compatibility: xbar_data → xb_prev_data
    xbar_data=None,
    # Control-function residuals (endogeneity correction)
    v_hat_data=None,
    device="cpu",
    verbose=False,
    tag="",
    cache_dir=None,
    force_retrain=False,
):
    """Train a StaticND or HabitND model.

    Includes caching: if a model with the same 'tag' exists in
    cache_dir, it is loaded instead of retrained.
    """
    model = model.to(device)

    # ── Cache Check ──────────────────────────────────────────────────────────
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_paths = _candidate_cache_paths(cache_dir, tag)
        cache_path = cache_paths[0]

        if not force_retrain and os.path.exists(cache_path):
            if verbose:
                print(f"    [Cache] Loading {tag} from {cache_path}")
            try:
                checkpoint = torch.load(cache_path, map_location=device, weights_only=False)
                sd = _adapt_legacy_state_dict(checkpoint["model_state_dict"])
                model.load_state_dict(sd, strict=False)
                model.eval()
                return model, checkpoint.get("history", [])
            except Exception as e:
                print(f"    [Cache] Failed to load {cache_path}: {e}. Retraining.")

    # ── Backward-compat shim ──────────────────────────────────────────────
    if xbar_data is not None and xb_prev_data is None:
        # Old single-array interface: treat xbar_data as xb_prev, derive
        # q_prev as a one-step lag of itself (approximation).
        import numpy as np
        xb_prev_data = xbar_data
        # q_prev ≈ xbar itself when we don't have explicit quantity data
        # (conservative fall-back: q_prev = xb_prev, so delta doesn't change the input)
        q_prev_data  = xbar_data

    mdp_mode = (xb_prev_data is not None) and (q_prev_data is not None)
    cf_mode  = (v_hat_data is not None) and hasattr(model, 'n_cf') and (model.n_cf > 0)

    model = model.to(device)
    # Adam gives stable updates across dense neural parameters.
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    N = len(income)
    slut_start = int(epochs * slut_start_frac)

    import numpy as np
    LP = torch.log(torch.tensor(prices,  dtype=torch.float32, device=device))
    LY = torch.log(torch.tensor(income,  dtype=torch.float32, device=device)).unsqueeze(1)
    W  = torch.tensor(w_expert, dtype=torch.float32, device=device)

    # Convert raw quantities to log-space (model always expects log-space inputs)
    if mdp_mode:
        XB_PREV = torch.log(torch.clamp(
            torch.tensor(xb_prev_data, dtype=torch.float32, device=device), min=1e-6))
        Q_PREV  = torch.log(torch.clamp(
            torch.tensor(q_prev_data,  dtype=torch.float32, device=device), min=1e-6))
    else:
        XB_PREV = Q_PREV = None

    # Control-function residuals tensor
    V_HAT = (torch.tensor(v_hat_data, dtype=torch.float32, device=device)
             if cf_mode else None)

    best_kl = float("inf")
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N, device=device)[:batch_size]
        lp_b, ly_b, w_b = LP[idx], LY[idx], W[idx]
        vh_b = V_HAT[idx] if cf_mode else None
        optimiser.zero_grad()

        if mdp_mode:
            xbp_b = XB_PREV[idx]
            qp_b  = Q_PREV[idx]
            w_pred = model(lp_b, ly_b, xbp_b, qp_b, v_hat=vh_b)
        else:
            w_pred = model(lp_b, ly_b, v_hat=vh_b)

        loss_kl = nn.KLDivLoss(reduction="batchmean")(torch.log(w_pred + 1e-10), w_b)

        lp_d = lp_b.detach().requires_grad_(True)
        if mdp_mode:
            w_mn = model(lp_d, ly_b, xbp_b, qp_b, v_hat=vh_b)
        else:
            w_mn = model(lp_d, ly_b, v_hat=vh_b)
        # Own-price monotonicity: penalise ∂w_g/∂log(p_g) > w_g.
        # This is equivalent to ∂q_g/∂p_g > 0 (upward-sloping quantity demand),
        # which is the correct economic constraint.  The weaker condition
        # ∂w_g/∂log(p_g) > 0 is wrong for demand systems where budget shares
        # legitimately rise with own price (Leontief, Stone–Geary), even though
        # quantity demand is still downward-sloping.
        loss_mono = torch.tensor(0.0, device=device)
        for _g in range(model.n_goods):
            _own = torch.autograd.grad(
                w_mn[:, _g].sum(), lp_d, create_graph=True, retain_graph=True
            )[0][:, _g]
            _threshold = w_mn[:, _g].detach()   # penalise only if ∂w/∂log p > w
            loss_mono = loss_mono + torch.mean(torch.clamp(_own - _threshold, min=0))
        loss_mono = loss_mono / model.n_goods

        loss_slut = torch.tensor(0.0, device=device)
        if ep >= slut_start:
            sub   = torch.randperm(N, device=device)[:64]
            vh_sub = V_HAT[sub] if cf_mode else None
            if mdp_mode:
                loss_slut = model.slutsky_penalty(
                    LP[sub], LY[sub], XB_PREV[sub], Q_PREV[sub], v_hat=vh_sub)
            else:
                loss_slut = model.slutsky_penalty(LP[sub], LY[sub], v_hat=vh_sub)

        loss = loss_kl + lam_mono * loss_mono + lam_slut * loss_slut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        if ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                if mdp_mode:
                    wp = model(LP, LY, XB_PREV, Q_PREV, v_hat=V_HAT)
                else:
                    wp = model(LP, LY, v_hat=V_HAT)
                kl_full = nn.KLDivLoss(reduction="batchmean")(
                    torch.log(wp + 1e-10), W).item()
            if kl_full < best_kl:
                best_kl = kl_full
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

        if ep % 400 == 0:
            delta_val = model.delta.item() if mdp_mode else float("nan")
            history.append({
                "epoch": ep,
                "kl":    loss_kl.item(),
                "delta": delta_val,
            })
            if verbose:
                delta_str = f" | delta(fixed)={delta_val:.3f}" if mdp_mode else ""
                print(f"    ep {ep:4d} | KL={loss_kl.item():.5f}{delta_str}")

    if best_state:
        model.load_state_dict(best_state)

    # ── Save to Cache ────────────────────────────────────────────────────────
    if cache_dir is not None:
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": history,
            }, cache_path)
        except Exception as e:
            print(f"    [Cache] Failed to save {cache_path}: {e}")

    model.eval()
    return model, history
