"""
BaseNonStationaryScheme class module.
"""

import logging
import numpy as np

from base_scheme import BaseScheme
from utils import newton_solve

from utils import Material
from utils import timer


class BaseNonStationaryScheme(BaseScheme):
    """
    Base class for non-stationary schemes.
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple,
        material: Material,
        dt: np.float64,
        limits: list[
            np.float64, np.float64, np.float64
        ],  # [a, b, T] - square [a, b] x [a, b] x [0, T]
    ):
        """
        Base non-stat initializer,

        Args:
            F: The inner heat (3-dim)
            G: The bound heat (3-dim)
            square_shape: shape of the scheme. [n, n] for discrete methods
                and [n, n, m, m] for direct.
            material: namedtuple object for containing material properties
            limits: description of the computing area, [a, b, T] for non-stationary
        """
        super().__init__(F, G, square_shape, material, limits)
        self.cur_layer = 0
        self.dt = dt

    @timer
    def solve(self, tol: np.float64, **kwargs) -> np.ndarray:
        """
        Common method to solve non-stationary schemes
        layer by layer with `BaseNonStationaryScheme._solve_layer()`

        Args:
            tol: absolute tolerance of Newton's method.
            inner_tol: relative tolerance for bicgstab.
                Explicitly pass like keyword argument.
            u0_squared: start point of the result at T=0.0 seconds.
                Explicitly pass like keyword argument.
        Returns:
            The solution of the scheme.
        """
        logger = logging.getLogger()

        U = np.zeros_like(self.F)
        inner_tol = kwargs.get("inner_tol", 5e-4)
        U[0] = kwargs.get("u0_squared", 300.0 * np.ones(self.square_shape)) / self.w

        for self.cur_layer in range(1, U.shape[0]):
            logger.info("Compute layer [%03d]", self.cur_layer)
            U[self.cur_layer], exit_code = self._solve_layer(
                tol, u_prev_squared=U[self.cur_layer - 1], inner_tol=inner_tol
            )
        U *= self.w

        return U, exit_code

    def _solve_layer(
        self, tol: np.float64, u_prev_squared: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Non-stat layer solver.
        Almost identical to `BaseStationaryScheme.solve()`

        Args:
            tol: absolute tolerance of Newton's method.
            u_prev_squared: result array on the previous layer.
            inner_tol: relative tolerance for bicgstab.
                Explicitly pass like keyword argument.
        Returns:
            Solution's next layer.
        """

        inner_tol = kwargs.get("inner_tol", 5e-4)
        u_prev = u_prev_squared.flatten()
        b = self.b[self.cur_layer].flatten()

        U, exit_code = newton_solve(
            b,
            lambda u_linear: self.operator(u_linear, u_prev_linear=u_prev),
            self.jacobian,
            tol,
            inner_tol,
            x0=u_prev,
        )

        return U.reshape(self.square_shape), exit_code

    @staticmethod
    def flatten(
        u_squared: np.ndarray,
        limits: list,
        **kwargs,
    ) -> np.ndarray:
        """
        Method to change ndarray-s to the [cells, cells] format at every layer.
        Does nothing in DM scheme.

        Args:
            u_squared: self.square_shape-like np.ndarray object.
            limits: description of the computing area, [a, b, T] for non-stationary
            mod: 0 - returns max information. 1 - makes array compatible with DM solutions
        Returns:
            res: ndarray with shape (*self.square_shape[:2])
        """
        mod = kwargs.get("mod", 0)
        if mod > 1 or u_squared.ndims != 5:
            return u_squared

        cells = u_squared.shape[0]
        cell_size = u_squared.shape[2]
        if mod == 0:
            flat_size = cells * cell_size
            res = np.zeros((u_squared.shape[0], flat_size, flat_size))
        else:
            res = np.zeros(u_squared.shape[:3])
        for (i, layer) in enumerate(u_squared):
            res[i] = BaseScheme.flatten(layer, limits, mod=mod)

        return res
