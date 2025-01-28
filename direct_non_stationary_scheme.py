"""
Module template docstring
"""

from sys import exit as sys_exit
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from numpy import float_power as fpower, fabs

from base_non_stationary_scheme import BaseNonStationaryScheme
from direct_stationary_scheme import DirectStationaryScheme
from enviroment import Material
from wraps import timer


class DirectNonStationaryScheme(BaseNonStationaryScheme, DirectStationaryScheme):
    """
    Class template docstring
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple[int, int, int, int],
        material: Material,
        dt: np.float64,
        limits: list[np.float64, np.float64, np.float64],
    ):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        super().__init__(F, G, square_shape, material, dt, limits)

        cells = self.square_shape[0]
        cell_size = self.square_shape[2]
        self.h = (self.limits[1] - self.limits[0]) / ((cell_size - 1) * cells)

        self.HeatStream = lambda v: self.stef_bolc * fabs(v) * fpower(v, 3)
        self.dHeatStream = (
            lambda v, dv: 4 * self.stef_bolc * fabs(v) * fpower(v, 2) * dv
        )

    @timer
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

        inner_tol = kwargs.get("inner_tol", 5e-4)
        u_prev = u_prev_squared.flatten()
        U = u_prev

        A = LinearOperator(
            (U.size, U.size),
            matvec=lambda du: self.jacobian(U, du),
        )

        self.G[self.cur_layer, 0, 0, 0, 0] *= 2
        self.G[self.cur_layer, -1, 0, -1, 0] *= 2
        self.G[self.cur_layer, -1, -1, -1, -1] *= 2
        self.G[self.cur_layer, 0, -1, 0, -1] *= 2

        b = (self.F[self.cur_layer] + (2 / self.h) * self.G[self.cur_layer]).flatten()
        R = b - self.operator(U, u_prev_linear=u_prev)
        dU, exit_code = bicgstab(
            A,
            R,
            rtol=inner_tol,
            atol=0.0,
            x0=R,
        )
        if exit_code:
            print(f"jacobian Failed with exit code: {exit_code} ON THE START")
            sys_exit()

        err = np.abs(dU).max()
        print(f"\t{err:.3e}")
        while err > tol:
            U += dU
            R = b - self.operator(U, u_prev_linear=u_prev)
            dU, exit_code = bicgstab(
                A,
                R,
                rtol=inner_tol,
                atol=0.0,
                x0=dU,
            )
            if exit_code:
                print(f"jacobian FAILED with exit code: {exit_code}")
                print(f"final error: {err:.3e}")
                sys_exit()
            err = np.abs(dU).max()
            print(f"\t{err:.3e}")
        return U.reshape(self.square_shape)

    def operator(self, u_linear: np.ndarray, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """

        return (
            DirectStationaryScheme.operator(self, u_linear)
            + (u_linear - kwargs["u_prev_linear"]) * self.material.crho / self.dt
        )

    def jacobian(self, u_linear: np.ndarray, du_linear: np.ndarray) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """

        return (
            DirectStationaryScheme.jacobian(self, u_linear, du_linear)
            + self.material.crho * du_linear / self.dt
        )

    @staticmethod
    def GetBoundaries(
        f_func,
        g_func: list,
        square_shape: tuple,
        material: Material,
        limits: list[np.float64, np.float64, np.float64],
        stef_bolc: np.float64,
        **kwargs
    ) -> list[np.ndarray, np.ndarray]:
        """
        Abstract static function to obtain F and G arrays

        Args:
            f_func
        Returns:
            [F, G]
        """
        HeatStream = lambda t: stef_bolc * np.power(t, 4)

        f = lambda T, x, y: HeatStream(f_func(T, x, y))
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
        t_array = np.arange(0, limits[2] + 0.5*dt, dt)

        F: np.ndarray = np.zeros((t_array.size, *square_shape))
        G: np.ndarray = np.zeros((t_array.size, *square_shape))

        for (layer, t_arg) in enumerate(t_array):
            for i in range(cells):
                for j in range(cells):
                    for i2 in range(cell_size):
                        for j2 in range(cell_size):
                            F[layer, i, j, i2, j2] = f(
                                t_arg,
                                (i * (cell_size - 1) + i2) * h,
                                (j * (cell_size - 1) + j2) * h,
                            )
            for k in range(cells):
                for k2 in range(cell_size):
                    G[layer, k, 0, k2, 0] = g[0](t_arg, (k * (cell_size - 1) + k2) * h)
                    G[layer, k, -1, k2, -1] = g[2](t_arg, (k * (cell_size - 1) + k2) * h)
                    G[layer, 0, k, 0, k2] = g[3](t_arg, (k * (cell_size - 1) + k2) * h)
                    G[layer, -1, k, -1, k2] = g[1](t_arg, (k * (cell_size - 1) + k2) * h)

        return [F, G]

    def flatten_layer(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        return DirectStationaryScheme.flatten(self, u_squared, kwargs["mod"])
