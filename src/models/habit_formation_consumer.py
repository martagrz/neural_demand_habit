"""Habit formation consumer."""

import numpy as np
from scipy.optimize import minimize


class HabitFormationConsumer:
    """Habit-adjusted CES."""

    name = "Habit Formation"

    def __init__(self, alpha=None, rho=0.45, theta=0.3, decay=0.7):
        self.alpha = np.array(alpha) if alpha is not None else np.array([0.4, 0.4, 0.2])
        self.rho = rho
        self.theta = theta
        self.decay = decay

    def solve_demand(self, prices, income, return_xbar=False):
        N, G = prices.shape
        shares = np.zeros((N, G))
        xbars = np.zeros((N, G))
        xbar = np.ones(G) * (np.mean(income) / (G * np.mean(prices)))
        for i in range(N):
            p, y = prices[i], income[i]
            floor = self.theta * xbar + 1e-6
            xbars[i] = xbar

            def neg_u(x):
                adj = x - self.theta * xbar
                if np.any(adj <= 0):
                    return 1e10
                return -(np.sum(self.alpha * adj**self.rho)) ** (1 / self.rho)

            x0 = np.maximum(y / (G * p), floor + 0.01)
            cons = {"type": "eq", "fun": lambda x, p=p, y=y: p @ x - y}
            res = minimize(
                neg_u, x0, bounds=[(floor[j], None) for j in range(G)], constraints=cons, method="SLSQP"
            )
            if res.success:
                shares[i] = res.x * p / y
                xbar = self.decay * xbar + (1 - self.decay) * res.x
            else:
                shares[i] = 1.0 / G
        if return_xbar:
            return np.clip(shares, 1e-6, 1.0), xbars
        return np.clip(shares, 1e-6, 1.0)

