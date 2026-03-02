"""Unified linear-IRL routines with both Dominicks and simulation APIs."""

import numpy as np


def run_lirl(ff, p, y, w, cfg):
    """Dominicks API: feature fn + cfg dict."""

    F = ff(p, y)
    theta = np.zeros(F.shape[2])
    for ep in range(cfg["lirl_epochs"]):
        eta = cfg["lirl_lr"] / (1.0 + ep / 1000.0)
        lg = np.einsum("ngk,k->ng", F, theta)
        lg -= lg.max(1, keepdims=True)
        prob = np.exp(lg)
        prob /= prob.sum(1, keepdims=True)
        grad = np.mean(np.einsum("ngk,ng->nk", F, w - prob), 0) - cfg["lirl_l2"] * theta
        theta += eta * grad
    return theta


def pred_lirl(ff, theta, p, y):
    """Dominicks API prediction helper."""

    F = ff(p, y)
    lg = np.einsum("ngk,k->ng", F, theta)
    lg -= lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def run_linear_irl(features, expert_w, lr=0.05, epochs=3000, l2=1e-4):
    """Simulation API: already-materialized features."""

    n_feat = features.shape[2]
    theta = np.zeros(n_feat)
    for ep in range(epochs):
        logits = np.tensordot(features, theta, axes=([2], [0]))
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        diff = (expert_w - probs)[:, :, None]
        grad = np.mean((features * diff).sum(axis=1), axis=0) - l2 * theta
        theta += (lr / (1.0 + ep / 1000.0)) * grad
    return theta


def predict_linear_irl(features, theta):
    """Simulation API prediction helper."""

    logits = np.tensordot(features, theta, axes=([2], [0]))
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)

