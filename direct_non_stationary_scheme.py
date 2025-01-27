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
        u_prev = u_prev_squared.reshape(self.linear_shape)
        U = u_prev

        A = LinearOperator(
            (*self.linear_shape, *self.linear_shape),
            matvec=lambda du: self.jacobian(U, du),
        )

        self.G[self.cur_layer, 0, 0, 0, 0] *= 2
        self.G[self.cur_layer, -1, 0, -1, 0] *= 2
        self.G[self.cur_layer, -1, -1, -1, -1] *= 2
        self.G[self.cur_layer, 0, -1, 0, -1] *= 2

        b = (self.F[self.cur_layer] + (2 / self.h) * self.G[self.cur_layer]).reshape(
            self.linear_shape
        )
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
        ## TODO

    def flatten(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        ## TODO

    def flatten_layer(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        ## TODO
