"""Dominicks neural training loop."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os


def _safe_tag(tag: str) -> str:
    return tag.replace(" ", "_").replace("=", "").replace("-", "_")


def _candidate_cache_paths(cache_dir: str, tag: str) -> list[str]:
    return [os.path.join(cache_dir, f"{_safe_tag(tag)}.pt")]


def _adapt_legacy_state_dict(sd: dict) -> dict:
    """Map legacy learned-delta checkpoints to fixed-delta format."""
    if "_delta" not in sd and "log_delta" in sd:
        with torch.no_grad():
            sd["_delta"] = torch.sigmoid(sd["log_delta"]).detach().clone()
        sd.pop("log_delta", None)
    return sd


def train_dominicks(model, p_tr, y_tr, w_tr, pfx, cfg,
                    xb_prev_tr=None, q_prev_tr=None, tag="",
                    store_idx_tr=None, v_hat_tr=None):
    """Shared training loop for StaticND / HabitND in Dominicks pipeline.
    
    Includes caching: if a model with the same 'tag' exists in
    cfg['model_cache_dir'], it is loaded instead of retrained.
    """

    dev = cfg["device"]
    verbose = bool(cfg.get("verbose", False))
    model = model.to(dev)

    # ── Cache Check ──────────────────────────────────────────────────────────
    cache_dir = cfg.get("model_cache_dir", "results/neural_demand/dominicks/models")
    os.makedirs(cache_dir, exist_ok=True)
    cache_paths = _candidate_cache_paths(cache_dir, tag)
    cache_path = cache_paths[0]

    if not cfg.get("force_retrain", False) and os.path.exists(cache_path):
        if verbose:
            print(f"    [Cache] Loading {tag} from {cache_path}")
        try:
            checkpoint = torch.load(cache_path, map_location=dev, weights_only=False)
            sd = _adapt_legacy_state_dict(checkpoint["model_state_dict"])
            model.load_state_dict(sd, strict=False)
            model.eval()
            return model, checkpoint.get("history", [])
        except Exception as e:
            print(f"    [Cache] Failed to load {cache_path}: {e}. Retraining.")

    # ── Training Setup ───────────────────────────────────────────────────────
    # Detect store-FE model: has an nn.Embedding 'store_emb' attribute
    _fe = (store_idx_tr is not None) and hasattr(model, 'store_emb')
    SI  = torch.tensor(store_idx_tr, dtype=torch.long).to(dev) if _fe else None
    # Detect control-function mode
    cf  = (v_hat_tr is not None) and hasattr(model, 'n_cf') and (model.n_cf > 0)
    V_HAT = torch.tensor(v_hat_tr, dtype=torch.float32).to(dev) if cf else None

    opt = optim.Adam(model.parameters(), lr=cfg[f"{pfx}_lr"], weight_decay=1e-5)
    ep_tot = cfg[f"{pfx}_epochs"]
    N, bs = len(y_tr), cfg[f"{pfx}_batch"]
    slut0 = int(ep_tot * cfg[f"{pfx}_slut_start"])
    mdp = (xb_prev_tr is not None) and (q_prev_tr is not None)

    LP = torch.log(torch.tensor(np.maximum(p_tr, 1e-8), dtype=torch.float32)).to(dev)
    LY = torch.log(torch.tensor(np.maximum(y_tr, 1e-8), dtype=torch.float32)).unsqueeze(1).to(dev)
    W  = torch.tensor(w_tr, dtype=torch.float32).to(dev)
    XB_PREV = torch.tensor(xb_prev_tr, dtype=torch.float32).to(dev) if mdp else None
    Q_PREV  = torch.tensor(q_prev_tr,  dtype=torch.float32).to(dev) if mdp else None

    best_kl, best_sd, hist = float("inf"), None, []
    _last_full_kl = float("inf")   # cache full-data KL for history logging
    _patience = int(cfg.get(f"{pfx}_early_stop_patience", 0))  # 0 = disabled
    _no_improve_eps = 0

    for ep in range(1, ep_tot + 1):
        model.train()
        idx = torch.randperm(N, device=dev)[:bs]
        lp_b, ly_b, w_b = LP[idx], LY[idx], W[idx]
        vh_b = V_HAT[idx] if cf else None

        opt.zero_grad()
        if mdp and _fe:
            xbp_b = XB_PREV[idx]; qp_b = Q_PREV[idx]
            wp = model(lp_b, ly_b, xbp_b, qp_b, SI[idx], v_hat=vh_b)
        elif mdp:
            xbp_b = XB_PREV[idx]; qp_b = Q_PREV[idx]
            wp = model(lp_b, ly_b, xbp_b, qp_b, v_hat=vh_b)
        elif _fe:
            wp = model(lp_b, ly_b, SI[idx], v_hat=vh_b)
        else:
            wp = model(lp_b, ly_b, v_hat=vh_b)

        lkl = nn.KLDivLoss(reduction="batchmean")(torch.log(wp + 1e-10), w_b)

        lp_d = lp_b.detach().requires_grad_(True)
        if mdp and _fe:
            wm = model(lp_d, ly_b, xbp_b, qp_b, SI[idx], v_hat=vh_b)
        elif mdp:
            wm = model(lp_d, ly_b, xbp_b, qp_b, v_hat=vh_b)
        elif _fe:
            wm = model(lp_d, ly_b, SI[idx], v_hat=vh_b)
        else:
            wm = model(lp_d, ly_b, v_hat=vh_b)
        # Own-price monotonicity: penalize positive ∂w_g/∂log p_g for each good g.
        # For large K, randomly subsample goods each step to avoid K backward passes.
        lmono = torch.tensor(0.0, device=dev)
        _max_mono = int(cfg.get("max_mono_goods", model.n_goods))
        _n_check = min(model.n_goods, _max_mono)
        _goods_check = (
            torch.randperm(model.n_goods, device=dev)[:_n_check].tolist()
            if _n_check < model.n_goods else range(model.n_goods)
        )
        for _g in _goods_check:
            _own = torch.autograd.grad(
                wm[:, _g].sum(), lp_d, create_graph=True, retain_graph=True
            )[0][:, _g]
            lmono = lmono + torch.mean(torch.clamp(_own, min=0))
        lmono = lmono / _n_check

        lslut = torch.tensor(0.0, device=dev)
        if ep >= slut0:
            sub    = torch.randperm(N, device=dev)[:64]
            vh_sub = V_HAT[sub] if cf else None
            _max_slut = int(cfg.get("max_slut_goods", model.n_goods))
            if mdp and _fe:
                lslut = model.slutsky(LP[sub], LY[sub], XB_PREV[sub], Q_PREV[sub],
                                      SI[sub], v_hat=vh_sub, max_goods=_max_slut)
            elif mdp:
                lslut = model.slutsky(LP[sub], LY[sub], XB_PREV[sub], Q_PREV[sub],
                                      v_hat=vh_sub, max_goods=_max_slut)
            elif _fe:
                lslut = model.slutsky(LP[sub], LY[sub], SI[sub], v_hat=vh_sub,
                                      max_goods=_max_slut)
            else:
                lslut = model.slutsky(LP[sub], LY[sub], v_hat=vh_sub,
                                      max_goods=_max_slut)

        loss = lkl + cfg[f"{pfx}_lam_mono"] * lmono + cfg[f"{pfx}_lam_slut"] * lslut
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Full-data eval every 10 epochs — also caches KL for history logging
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                if mdp and _fe:
                    wa = model(LP, LY, XB_PREV, Q_PREV, SI, v_hat=V_HAT)
                elif mdp:
                    wa = model(LP, LY, XB_PREV, Q_PREV, v_hat=V_HAT)
                elif _fe:
                    wa = model(LP, LY, SI, v_hat=V_HAT)
                else:
                    wa = model(LP, LY, v_hat=V_HAT)
                kl = nn.KLDivLoss(reduction="batchmean")(torch.log(wa + 1e-10), W).item()
            _last_full_kl = kl   # cache for history logging
            if kl < best_kl:
                best_kl = kl
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                _no_improve_eps = 0
            else:
                _no_improve_eps += 10
            model.train()
            if _patience > 0 and _no_improve_eps >= _patience:
                if verbose:
                    print(f"    [{tag}] Early stop at ep {ep} (no improvement for {_patience} eps)")
                break

        # Log full-data KL (not noisy mini-batch KL) for clean convergence plots
        if ep % 20 == 0:
            delta_val = model.delta.item() if mdp else float("nan")
            hist.append({
                "epoch": ep,
                "kl":    _last_full_kl,
                "delta": delta_val,
            })
            if verbose:
                delta_str = f" | delta(fixed)={delta_val:.3f}" if mdp else ""
                print(f"    [{tag}] ep {ep:4d} | KL={_last_full_kl:.5f}{delta_str}")

    # Ensure KL_min is always available, even for very short runs.
    if not np.isfinite(best_kl):
        model.eval()
        with torch.no_grad():
            if mdp and _fe:
                wa = model(LP, LY, XB_PREV, Q_PREV, SI, v_hat=V_HAT)
            elif mdp:
                wa = model(LP, LY, XB_PREV, Q_PREV, v_hat=V_HAT)
            elif _fe:
                wa = model(LP, LY, SI, v_hat=V_HAT)
            else:
                wa = model(LP, LY, v_hat=V_HAT)
            best_kl = nn.KLDivLoss(reduction="batchmean")(torch.log(wa + 1e-10), W).item()
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.train()

    if best_sd:
        model.load_state_dict(best_sd)

    print(f"    [{tag}] KL_min={best_kl:.6f}")
    
    # ── Save to Cache ────────────────────────────────────────────────────────
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "history": hist,
            "cfg": cfg,
        }, cache_path)
    except Exception as e:
        print(f"    [Cache] Failed to save {cache_path}: {e}")

    model.eval()
    return model, hist
