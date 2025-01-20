import numpy as np
import abc

from enviroment import Material


class BaseScheme(abc.ABC):
    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple,  # [cells, cells, cell_size, cell_size]
        material: Material,
        limits: list,
        *args,
        **kwargs
    ):
        self.F = F
        self.G = G
        self.U = np.zeros(square_shape)
        self.dU = np.zeros(square_shape)

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
    def solve(
        self, tol: np.float64, U0_squared: np.ndarray = None, *args, **kwargs
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def operator(self, u_linear: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def jacobian(self, du_linear: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def GetBoundaries(
        f_func,
        g_func: list,
        square_shape: tuple,
        material: Material,
        limits: list,
        stef_bolc: np.float64,
    ) -> list[np.ndarray, np.ndarray]:
        pass

    @property
    def normed(self):
        return self.__normed

    @normed.setter
    def normed(self, flag):
        self.__normed = flag
        self.w = 100 if self.__normed else 1
        self.stef_bolc = 5.67036713 if self.__normed else 5.67036713e-8
