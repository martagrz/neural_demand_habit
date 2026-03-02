"""Leontief consumer."""

import numpy as np


class LeontiefConsumer:
    """U = min(x_i / a_i)."""

    name = "Leontief"

    def __init__(self, a=None):
        self.a = np.array(a) if a is not None else np.array([1.0, 0.8, 1.5])

    def solve_demand(self, prices, income):
        denom = (prices * self.a[None, :]).sum(axis=1, keepdims=True)
        x = income[:, None] * self.a[None, :] / denom
        return np.clip(x * prices / income[:, None], 1e-6, 1.0)

