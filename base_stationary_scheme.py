"""
Base Stationary Scheme class module.
"""

import numpy as np

from base_scheme import BaseScheme
from newton_solve import newton_solve
from wraps import timer


class BaseStationaryScheme(BaseScheme):
    """
    template - write later
    """
    @timer
    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
        """
        Main method to solve the scheme

        Args:
            tol: absolute tolerance of Newton's method.
            inner_tol: relative tolerance for bicgstab.
                Explicitly pass like keyword argument.
            u0_squared: start point for computing the result.
                Explicitly pass like keyword argument.
        Returns:
            The solution of the scheme.
        """
        inner_tol = kwargs.get("inner_tol", 5e-4)
        u0 = kwargs.get("u0_squared", 300.0 * np.ones(np.prod(self.square_shape)))
        u = u0.flatten() / self.w
        b = kwargs.get("b", self.b).flatten()
        operator = kwargs.get("operator", self.operator)
        jacobian = kwargs.get("jacobian", self.jacobian)

        sol = newton_solve(
            b,
            operator,
            jacobian,
            tol,
            inner_tol,
            x0=u
        )

        u = (self.w * sol).reshape(self.square_shape)
        return u