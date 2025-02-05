"""
will appear later
"""

import logging
from typing import Callable
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab

def newton_solve(
        b: np.ndarray,
        operator: Callable[[np.ndarray], np.ndarray],
        jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray],
        tol: np.float64,
        inner_tol: np.ndarray = 5e-4,
        **kwargs,
    ) -> np.ndarray:
    """
    Solve non-linear equations system.

    Args:
        tol: absolute tolerance of Newton's method.
        inner_tol: relative tolerance for bicgstab.
            Explicitly pass like keyword argument.
        u0_squared: start point for computing the result.
            Explicitly pass like keyword argument.
    Returns:
        The solution of the scheme.
    """
    logger = logging.getLogger()

    x0 = kwargs.get("x0", np.ones_like(b))
    x = x0.copy()

    A = LinearOperator(
        (x.size, x.size),
        matvec=lambda dx: jacobian(x, dx),
    )

    r = b - operator(x)
    r_norm, r_norm_prev = np.abs(r).max(), np.inf
    dx = r.copy()

    err = 100.0
    while err > tol:
        dx, exit_code = bicgstab(
            A,
            r,
            rtol=inner_tol,
            atol=0.0,
            x0=dx,
        )
        err = np.abs(dx).max()
        r = b - operator(x + dx)
        r_norm, r_norm_prev = np.abs(r).max(), r_norm
        if exit_code:
            logger.warning(
                "\tNewton err: %.3e | r_norm: %.3e | BiCGstab FAILED(%d)",
                err, r_norm, exit_code
            )
            if r_norm > r_norm_prev:
                logger.error("\tBAD convergence")
                return x, exit_code
        else:
            logger.debug(
                "\tNewton err: %.3e | r_norm: %.3e",
                err, r_norm
            )
        x += dx
    return x
