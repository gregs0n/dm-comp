"""
Base Scheme class module
"""

import abc
import numpy as np

from enviroment import Material


class BaseScheme(abc.ABC):
    """
    Class for containing common methods and attributes of
    different schemes from various diff equations tasks
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple,  # [cells, cells, cell_size, cell_size]
        material: Material,
        limits: list,
    ):
        """
        Base initializer of every derivative scheme.
        """
        self.F = F
        self.G = G

        self.square_shape = square_shape
        _temp_shape_size = 1
        for _shape_size in square_shape:
            _temp_shape_size *= _shape_size
        self.linear_shape = (_temp_shape_size,)

        self.material = material
        self.limits = limits

        self.__normed = 1
        self.w = 100.0
        self.stef_bolc = 5.67036713

    @abc.abstractmethod
    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
        """
        Abstarct method for solving schemes

        Args:
            tol: absolute tolerance of Newton's method.
            u0_squared: start point for computing the result
        Returns:
            The solution of the scheme.
        """

    @abc.abstractmethod
    def operator(self, u_linear: np.ndarray, **kwargs) -> np.ndarray:
        """
        Abstract method for operator of the scheme.

        Args:
            u_linear: 1d-shape array of U function.
        Returns:
            A(u), where A - is the differential system's scheme operator
        """

    @abc.abstractmethod
    def jacobian(self, u_linear: np.ndarray, du_linear: np.ndarray) -> np.ndarray:
        """
        Abstract method for operator's jacobian of the scheme.
        Use in Newton's method. For computing uses last newton's approx of the U-function

        Args:
            du_linear: 1d-shape array of dU function.
        Returns:
            Jac(du, U)
        """

    @staticmethod
    @abc.abstractmethod
    def GetBoundaries(
        f_func,
        g_func: list,
        square_shape: tuple,
        material: Material,
        limits: list,
        stef_bolc: np.float64,
    ) -> list[np.ndarray, np.ndarray]:
        """
        Abstract static function to obtain F and G arrays

        Args:
            f_func
        Returns:
            [F, G]
        """

    @property
    def normed(self):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        return self.__normed

    @normed.setter
    def normed(self, flag):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        self.__normed = flag
        self.w = 100 if self.__normed else 1
        self.stef_bolc = 5.67036713 if self.__normed else 5.67036713e-8
