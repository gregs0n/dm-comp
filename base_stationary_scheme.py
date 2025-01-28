"""
Base Stationary Scheme class module.
"""

import abc
import numpy as np

from base_scheme import BaseScheme


class BaseStationaryScheme(BaseScheme):
    """
    Class for logic contingency of the project.
    For each Sationary scheme BaseScheme's interface is enough.
    May be removed soon.
    """

    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
