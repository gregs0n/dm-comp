"""
Module template docstring
"""

from abc import abstractmethod
import logging
import numpy as np

from base_scheme import BaseScheme
from base_stationary_scheme import BaseStationaryScheme

from enviroment import Material
from wraps import timer


class BaseNonStationaryScheme(BaseScheme):
    """
    Class template docstring
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
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        super().__init__(F, G, square_shape, material, limits)
        self.cur_layer = 0
        self.dt = dt

    @timer
    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        logger = logging.getLogger()

        U = np.zeros_like(self.F)
        inner_tol = kwargs.get("inner_tol", 5e-4)
        U[0] = kwargs.get("u0_squared", 300.0 * np.ones(self.square_shape)) / self.w

        # view_data = self.flatten_layer(self.w*U[self.cur_layer], mod=0)
        # drawHeatmap(view_data, self.limits[:-1], f"images/direct_non_stat/plot_{self.cur_layer:03}", show_plot=0, zlim=[300, 600])

        for self.cur_layer in range(1, U.shape[0]):
            logger.info("Compute layer [%03d]", self.cur_layer)
            U[self.cur_layer], _ = self.solve_layer(
                tol, u_prev_squared=U[self.cur_layer - 1], inner_tol=inner_tol
            )
        U *= self.w

        return U

    def solve_layer(
        self, tol: np.float64, u_prev_squared: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """

        inner_tol = kwargs.get("inner_tol", 5e-4)
        u_prev = u_prev_squared
        b = self.b[self.cur_layer].flatten()

        U, exit_code = BaseStationaryScheme.solve(
            self,
            tol,
            inner_tol=inner_tol,
            u0_squared=self.w*u_prev,
            b=b,
            operator=lambda u_linear: self.operator(u_linear, u_prev_linear=u_prev.flatten()),
            jacobian=self.jacobian
        )

        return U / self.w, exit_code

    def flatten(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        mod = kwargs.get("mod", 0)
        if mod > 1 or len(u_squared.shape) != 5:
            return u_squared

        cells = self.square_shape[0]
        cell_size = self.square_shape[2]
        if mod == 0:
            flat_size = cells * cell_size
            flat_shape = (u_squared.shape[0], flat_size, flat_size)
            res = np.zeros(flat_shape)
        else:
            res = np.zeros((u_squared.shape[0], cells, cells))
        for i in range(u_squared.shape[0]):
            res[i] = self.flatten_layer(u_squared[i], mod=mod)

        return res

    @abstractmethod
    def flatten_layer(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
