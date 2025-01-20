import numpy as np
import abc

from BaseScheme import BaseScheme
from enviroment import Material
from draw import drawHeatmap

class BaseStationaryScheme(BaseScheme):
    def __init__(
            self,
            F: np.ndarray,
            G: np.ndarray,
            square_shape: tuple,
            material: Material,
            limits: list[np.float64, np.float64],
            *args,
            **kwargs
    ):
        super().__init__(F, G, square_shape, material, limits)

    def solve(self, tol: np.float64, U0_squared: np.ndarray = None, *args, **kwargs) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def flatten(self, u_squared: np.ndarray, *args, **kwargs):
        pass

    # def drawHeatmap(self, data: np.ndarray):
    #     _data = self.flatten(data)
    #     drawHeatmap(_data, self.limits, "plot", show_plot=1)