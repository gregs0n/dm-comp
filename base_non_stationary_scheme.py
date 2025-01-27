"""
Module template docstring
"""

import abc
import numpy as np

from base_scheme import BaseScheme
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
        limits: list[np.float64, np.float64, np.float64],
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
        self.dt = self.limits[0] / self.F.shape[0]

    @timer
    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        U = np.zeros_like(self.F)
        inner_tol = kwargs.get("inner_tol", 5e-4)
        U[0] = kwargs.get("u0_squared", 300.0 * np.ones(self.square_shape)) / self.w

        for self.cur_layer in range(1, U.shape[0]):
            print(f"[{self.cur_layer:03}]", end="\t")
            U[self.cur_layer] = self.solve_layer(
                tol, U[self.cur_layer], inner_tol=inner_tol
            )
        U *= self.w

        return U

    @abc.abstractmethod
    def solve_layer(
        self, tol: np.float64, u_prev_squared: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """

    @abc.abstractmethod
    def flatten(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
