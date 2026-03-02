"""Unified linear-feature maps with both Dominicks and simulation APIs."""

import numpy as np


# --- Dominicks API (numerically safe logs) ---
def feat_shared(p, y):
    g = p.shape[1]
    lp = np.log(np.maximum(p, 1e-8))
    ly = np.log(np.maximum(y, 1e-8))
    F = np.zeros((len(y), g, 3))
    for i in range(g):
        F[:, i] = np.c_[lp[:, i], lp[:, i] ** 2, ly]
    return F


def feat_good_specific(p, y):
    g = p.shape[1]
    lp = np.log(np.maximum(p, 1e-8))
    ly = np.log(np.maximum(y, 1e-8))
    F = np.zeros((len(y), g, g + 2))
    for i in range(g):
        F[:, i] = np.c_[lp, lp[:, i] ** 2, ly]
    return F


def feat_orth(p, y):
    g = p.shape[1]
    lp = np.log(np.maximum(p, 1e-8))
    ly = np.log(np.maximum(y, 1e-8))
    Q, _ = np.linalg.qr(lp - lp.mean(0))
    Q = Q[:, :g]
    F = np.zeros((len(y), g, 2 * g + 2))
    for i in range(g):
        F[:, i, i] = 1.0
        F[:, i, g : 2 * g] = Q
        F[:, i, 2 * g] = lp[:, i] ** 2
        F[:, i, 2 * g + 1] = ly
    return F


# --- Simulation API (original direct-log behavior) ---
def features_shared(p, y):
    N, G = p.shape
    F = np.zeros((N, G, 3))
    lp = np.log(p)
    for i in range(G):
        F[:, i, 0] = lp[:, i]
        F[:, i, 1] = lp[:, i] ** 2
        F[:, i, 2] = np.log(y)
    return F


def features_good_specific(p, y):
    N, G = p.shape
    F = np.zeros((N, G, G + 2))
    lp = np.log(p)
    for i in range(G):
        F[:, i, :G] = lp
        F[:, i, G] = lp[:, i] ** 2
        F[:, i, G + 1] = np.log(y)
    return F


def features_orthogonalised(p, y):
    N, G = p.shape
    lp = np.log(p)
    Q, _ = np.linalg.qr(lp - lp.mean(axis=0))
    F = np.zeros((N, G, G + G + 1 + 1))
    for i in range(G):
        F[:, i, i] = 1.0
        F[:, i, G : 2 * G] = Q
        F[:, i, 2 * G] = lp[:, i] ** 2
        F[:, i, 2 * G + 1] = np.log(y)
    return F

