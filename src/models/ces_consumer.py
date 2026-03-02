"""CES consumer."""

import numpy as np


class CESConsumer:
    """CES: closed-form demand."""

    name = "CES"

    def __init__(self, alpha=None, rho=0.45):
        self.alpha = np.array(alpha) if alpha is not None else np.array([0.4, 0.4, 0.2])
        self.rho = rho

    def solve_demand(self, prices, income):
        sigma = 1.0 / (1.0 - self.rho)
        num = self.alpha[None, :] ** sigma * prices ** (1.0 - sigma)
        return num / num.sum(axis=1, keepdims=True)

