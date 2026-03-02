"""Stone-Geary consumer."""

import numpy as np


class StoneGearyConsumer:
    """LES with subsistence minima."""

    name = "Stone-Geary"

    def __init__(self, alpha=None, gamma=None):
        self.alpha = np.array(alpha) if alpha is not None else np.array([0.5, 0.3, 0.2])
        self.alpha /= self.alpha.sum()
        self.gamma = np.array(gamma) if gamma is not None else np.array([50.0, 30.0, 20.0])

    def solve_demand(self, prices, income):
        sub = (prices * self.gamma[None, :]).sum(axis=1)
        sup_ = np.maximum(income - sub, 1e-6)
        exp_v = prices * self.gamma[None, :] + self.alpha[None, :] * sup_[:, None]
        return np.clip(exp_v / income[:, None], 1e-6, 1.0)

