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

    n_iter = 0
    x0 = kwargs.get("x0", np.ones_like(b))
    x = x0.copy()

    A = LinearOperator(
        (x.size, x.size),
        matvec=lambda dx: jacobian(x, dx),
        dtype=np.float64
    )

    r = b - operator(x)
    r_norm, r_norm_prev = norm(r, ord=np.inf), np.inf
    dx = np.ones_like(r)

    logger.debug("\t[nn]\t_r_norm_\tdx_norm")
    while r_norm > tol:
        n_iter += 1
        dx, _ = bicgstab(
            A,
            r,
            rtol=inner_tol,
            atol=0.0,
            x0=dx,
        )
        alpha = 1.0
        r = b - operator(x + alpha*dx)
        r_norm = norm(r, ord=np.inf)
        k = 0
        while r_norm > r_norm_prev:
            k += 1
            logger.warning(
                "\t[  ]\t%.2e\tr_prev=%.2e\talpha=%.6f",
                r_norm, r_norm_prev, alpha
            )
            alpha *= 0.5
            r = b - operator(x + alpha*dx)
            r_norm = norm(r, ord=np.inf)
            if k > 10:
                logger.error("Tolerance not achieved")
                return x + alpha * dx, 1

        dx *= alpha
        r_norm_prev = r_norm
        dx_norm = norm(dx, ord=np.inf)
        logger.debug(
            "\t[%02d]\t%.2e\t%.2e",
            n_iter, r_norm, dx_norm
        )
        x += dx

    return x, 0
