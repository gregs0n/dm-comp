"""
Module template docstring
"""

import abc
import numpy as np

from base_scheme import BaseScheme
from enviroment import Material

# from draw import drawHeatmap


class BaseStationaryScheme(BaseScheme):
    """
    Class template docstring
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple,
        material: Material,
        limits: list[np.float64, np.float64]
    ):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        super().__init__(F, G, square_shape, material, limits)

    def solve(
        self, tol: np.float64, *args, **kwargs
    ) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """

    @abc.abstractmethod
    def flatten(self, u_squared: np.ndarray, *args, **kwargs):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """

    # def drawHeatmap(self, data: np.ndarray):
    #     _data = self.flatten(data)
    #     drawHeatmap(_data, self.limits, "plot", show_plot=1)
