"""Quasilinear consumer."""

import numpy as np


class QuasilinearConsumer:
    """U = x0 + a1 ln(x1+1) + a2 ln(x2+1)."""

    name = "Quasilinear"

    def __init__(self, a=None):
        self.a = np.array(a) if a is not None else np.array([1.5, 0.8])

    def solve_demand(self, prices, income):
        N, G = prices.shape
        shares = np.zeros((N, G))
        for i in range(N):
            p, y = prices[i], income[i]
            x1 = max(self.a[0] / p[1] - 1, 1e-6)
            x2 = max(self.a[1] / p[2] - 1, 1e-6)
            rem = y - p[1] * x1 - p[2] * x2
            x0 = max(rem / p[0], 1e-6) if rem > 0 else 1e-6
            shares[i] = np.array([x0, x1, x2]) * p / y
        return np.clip(shares, 1e-6, 1.0)

