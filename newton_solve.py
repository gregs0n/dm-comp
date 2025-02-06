"""
will appear later
"""

import logging
from typing import Callable
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from scipy.linalg import norm

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

    alpha = 0.875

    x0 = kwargs.get("x0", np.ones_like(b))
    x = x0.copy()

    A = LinearOperator(
        (x.size, x.size),
        matvec=lambda dx: jacobian(x, dx),
    )

    r = b - operator(x)
    r_norm, r_norm_prev = np.abs(r).max(), np.inf
    dx = np.ones_like(r)

    err = 100.0
    logger.debug("\tNewton err\tr_norm\tdr_before\tdr_after")
    while err > tol:
        dr_norm_before = norm(r - A(dx), ord=np.inf)
        dx, exit_code = bicgstab(
            A,
            r,
            rtol=inner_tol,
            atol=0.0,
            x0=dx,
        )
        dr_norm_after = norm(r - A(dx), ord=np.inf)
        dx *= alpha
        err = np.abs(dx).max()
        r = b - operator(x + dx)
        r_norm, r_norm_prev = np.abs(r).max(), r_norm
        if exit_code:
            logger.warning(
                "\t%.2e\t%.2e\t%.2e\t%.2e\tBiCGstab FAILED(%d)",
                err, r_norm, dr_norm_before, dr_norm_after, exit_code
            )
            if r_norm > r_norm_prev:
                logger.error("\tBAD convergence")
                return x
        else:
            logger.debug(
                "\t%.2e\t%.2e\t%.2e\t%.2e",
                err, r_norm, dr_norm_before, dr_norm_after,
            )
        x += dx

    dx /= alpha
    err = np.abs(dx).max()
    x += (1.0 - alpha) * dx
    r = b - operator(x)
    r_norm = np.abs(r).max()
    logger.debug(
        "\t%.2e\t%.2e\t%.2e\t%.2e",
        err, r_norm, dr_norm_before, dr_norm_after,
    )
    return x
