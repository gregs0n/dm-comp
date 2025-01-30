"""
Base Stationary Scheme class module.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab

from base_scheme import BaseScheme
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
        b = kwargs.get("b", self.b)
        u0 = kwargs.get("u0_squared", 300.0 * np.ones(np.prod(self.square_shape)))
        U = u0.flatten() / self.w

        A = LinearOperator(
            (U.size, U.size),
            matvec=lambda du: self.jacobian(U, du),
        )

        R = b - self.operator(U)
        dU, exit_code = bicgstab(
            A,
            R,
            rtol=inner_tol,
            atol=0.0,
            x0=R,
        )
        if exit_code:
            print(f"jacobian Failed with exit code: {exit_code} ON THE START")
            U = (self.w * U).reshape(self.square_shape)
            return U, exit_code

        err = np.abs(dU).max()
        print(f"\t{err:.3e}")
        while err > tol:
            U += dU
            R = b - self.operator(U)
            dU, exit_code = bicgstab(
                A,
                R,
                rtol=inner_tol,
                atol=0.0,
                x0=dU,
            )
            if exit_code:
                print(f"jacobian FAILED with exit code: {exit_code}")
                print(f"final error: {err:.3e}")
                U += dU
                U = (self.w * U).reshape(self.square_shape)
                return U, exit_code
            err = np.abs(dU).max()
            print(f"\t{err:.3e}")
        U = (self.w * U).reshape(self.square_shape)
        return U, exit_code
