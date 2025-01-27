"""
Module template docstring
"""

import abc
import numpy as np

from base_scheme import BaseScheme


class BaseStationaryScheme(BaseScheme):
    """
    Class template docstring
    """

    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
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
