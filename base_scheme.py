"""
Base Scheme class module.
"""

from abc import ABC, abstractmethod
import numpy as np

from enviroment import Material


class BaseScheme(ABC):
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

        Args:
            F: The inner heat
            G: The bound heat
            square_shape: shape of the scheme. [n, n] for discrete methods
                and [n, n, m, m] for direct.
            material: namedtuple object for containing material properties
            limits: description of the computing area, [a, b] for
                stationary schemes and [a, b, T] for non-stationary
        """
        self.F = F
        self.G = G

        self.b = self.F + self.G
        self.h = None

        self.square_shape = square_shape

        self.material = material
        self.limits = limits

        self.__normed = 1
        self.w = 100.0
        self.stef_bolc = 5.67036713

    @abstractmethod
    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
        """
        Abstarct method for solving schemes

        Args:
            tol: absolute tolerance of Newton's method.
            u0_squared: start point for computing the result.
                Explicitly pass like keyword argument.
        Returns:
            The solution of the scheme.
        """

    @abstractmethod
    def operator(self, u_linear: np.ndarray, **kwargs) -> np.ndarray:
        """
        Abstract method for operator of the scheme.

        Args:
            u_linear: 1d-shape array of U function.
        Returns:
            A(u), where A - is the differential system's scheme operator
        """

    @abstractmethod
    def jacobian(self, u_linear: np.ndarray, du_linear: np.ndarray) -> np.ndarray:
        """
        Abstract method for operator's jacobian of the scheme.
        Use in Newton's method by BiCGstab as matvec.
        For computing uses current newton's approx of the U-function

        Args:
            u_linear: 1d-shape array of U function.
            du_linear: 1d-shape array of dU function.
        Returns:
            Jac(U, dU)
        """

    @abstractmethod
    def flatten(self, u_squared: np.ndarray, *args, **kwargs):
        """
        Abstract method to change solutions to the [cells, cells] format.
        Does nothing in FDM and SDM schemes.

        Args:
            u_squared: self.square_shape-like np.ndarray object.
        Returns:
            res: ndarray with shape (*self.square_shape[:2])
        """

    @staticmethod
    @abstractmethod
    def GetBoundaries(
        f_func,
        g_func: list,
        square_shape: tuple,
        material: Material,
        limits: list,
        stef_bolc: np.float64,
        **kwargs
    ) -> list[np.ndarray, np.ndarray]:
        """
        Abstract static function to obtain F and G arrays.
        Parameters are the same as in __init__()

        Args:
            f_func: the temperature of the inner heat sources.
            g_func: list of 4 functions g(x, y) for the bound temperature:
                [
                    g(x=[a,b], y=a),

                    g(x=b, y=[a, b]),

                    g(x=[a,b], y=b),

                    g(x=a, y=[a,b])
                ]
            square_shape: shape of the scheme. [n, n] for discrete methods
                and [n, n, m, m] for direct.
            limits: description of the computing area, [a, b] for
                stationary schemes and [a, b, T] for non-stationary
        Returns:
            [F, G]
        """

    @property
    def normed(self):
        """
        `normed` property getter
        """
        return self.__normed

    @normed.setter
    def normed(self, flag):
        """
        `normed` property setter.
        """
        self.__normed = flag
        self.w = 100 if self.__normed else 1
        self.stef_bolc = 5.67036713 if self.__normed else 5.67036713e-8
