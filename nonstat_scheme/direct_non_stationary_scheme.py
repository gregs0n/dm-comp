"""
DirectNonStationaryScheme class module.
"""

import numpy as np
from numpy import float_power as fpower, fabs

from nonstat_scheme.base_non_stationary_scheme import BaseNonStationaryScheme
from stat_scheme import DirectStationaryScheme
from utils import Material


class DirectNonStationaryScheme(BaseNonStationaryScheme, DirectStationaryScheme):
    """
    Direct Stationary scheme - scheme to obtain the most exact solution
    of the complex heat conductivity non-sationary question in
    2d slice of the square rods pack.

    Used to compare other approx schemes' correctness
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple[int, int, int, int],
        material: Material,
        dt: np.float64,
        limits: list[np.float64, np.float64, np.float64],
        **kwargs,
    ):
        """
        Args:
            F: The inner heat
            G: The bound heat
            square_shape: shape of the scheme. [n, n] for discrete methods
                and [n, n, m, m] for direct.
            material: namedtuple object for containing material properties
            dt: time step.
            limits: description of the computing area, [a, b] for
                stationary schemes.
        Returns:
            the instance of the DirectNonStationaryScheme class.
        """
        super().__init__(F, G, square_shape, material, dt, limits)

        cells = self.square_shape[0]
        cell_size = self.square_shape[2]
        self.h = (self.limits[1] - self.limits[0]) / ((cell_size - 1) * cells)

        mask = (2 / self.h) * np.ones_like(self.G)

        mask[:, 0, 0, 0, 0] *= 2
        mask[:, -1, 0, -1, 0] *= 2
        mask[:, -1, -1, -1, -1] *= 2
        mask[:, 0, -1, 0, -1] *= 2

        self.b = self.F + mask * self.G

        self.HeatStream = lambda v: self.stef_bolc * fabs(v) * fpower(v, 3)
        self.dHeatStream = (
            lambda v, dv: 4 * self.stef_bolc * fabs(v) * fpower(v, 2) * dv
        )

    def operator(self, u_linear: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes c_rho * du/dt + `DirectStationaryScheme.operator(u_linear)`

        Args:
            u_linear: 1-d array of the u - temperature.
            u_prev_linear: U array on the previous layer.
        Returns:
            1-d array  of the c_rho * du/dt + A(u)
        """

        return (
            DirectStationaryScheme.operator(self, u_linear)
            + (u_linear - kwargs["u_prev_linear"]) * self.material.crho * self.w / self.dt
        )

    def jacobian(self, u_linear: np.ndarray, du_linear: np.ndarray) -> np.ndarray:
        """
        Computes gradient value in the U-point of A-operator.
        Use in Newton's method by BiCGstab as matvec to find newton's dU.
        For computing uses current newton's approx of the U-function.

        Args:
            u_linear: 1d-shape array of U function.
            du_linear: 1d-shape array of dU function.
        Returns:
            c_rho*du + `DirectStationaryScheme.jacobian(U, dU)`: 1d_shape array.
        """

        return (
            DirectStationaryScheme.jacobian(self, u_linear, du_linear)
            + self.material.crho * self.w * du_linear / self.dt
        )

    @staticmethod
    def GetBoundaries(
        f_func,
        g_func: list,
        square_shape: tuple,
        material: Material,
        limits: list[np.float64, np.float64, np.float64],
        stef_bolc: np.float64,
        **kwargs,
    ) -> list[np.ndarray, np.ndarray]:
        """
        Static function to obtain F and G arrays.
        Parameters are the same as in __init__()

        Args:
            f_func: the temperature of the inner heat sources.
            g_func: list of 4 functions g(x, y) for the bound temperature:
                [g(x=[a,b], y=a), g(x=b, y=[a, b]), g(x=[a,b], y=b), g(x=a, y=[a,b])]
            square_shape: shape of the scheme. [n, n] for discrete methods
                and [n, n, m, m] for direct.
            limits: description of the computing area, [a, b, T] for non-stationary
        Returns:
            [F, G]
        """
        HeatStream = lambda t: stef_bolc * np.power(t, 4)

        # f = lambda T, x, y: HeatStream(f_func(T, x, y))
        g = [
            lambda T, t: HeatStream(g_func[0](T, t)),
            lambda T, t: HeatStream(g_func[1](T, t)),
            lambda T, t: HeatStream(g_func[2](T, t)),
            lambda T, t: HeatStream(g_func[3](T, t)),
        ]

        ############

        cells = square_shape[0]
        cell_size = square_shape[2]
        h = (limits[1] - limits[0]) / ((cell_size - 1) * cells)
        dt = kwargs.get("dt", 0.1)
        t_array = np.arange(0, limits[2] + 0.5 * dt, dt)

        F: np.ndarray = np.zeros((t_array.size, *square_shape))
        G: np.ndarray = np.zeros((t_array.size, *square_shape))

        for layer, t_arg in enumerate(t_array):
            # for i in range(cells):
            #     for j in range(cells):
            #         for i2 in range(cell_size):
            #             for j2 in range(cell_size):
            #                 F[layer, i, j, i2, j2] = f(
            #                     t_arg,
            #                     (i * (cell_size - 1) + i2) * h,
            #                     (j * (cell_size - 1) + j2) * h,
            #                 )
            for k in range(cells):
                for k2 in range(cell_size):
                    G[layer, k, 0, k2, 0] = g[0](t_arg, (k * (cell_size - 1) + k2) * h)
                    G[layer, k, -1, k2, -1] = g[2](
                        t_arg, (k * (cell_size - 1) + k2) * h
                    )
                    G[layer, 0, k, 0, k2] = g[3](t_arg, (k * (cell_size - 1) + k2) * h)
                    G[layer, -1, k, -1, k2] = g[1](
                        t_arg, (k * (cell_size - 1) + k2) * h
                    )

        return [F, G]
