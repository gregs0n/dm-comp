"""
Module template docstring
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
from numpy import float_power as fpower, fabs

from base_stationary_scheme import BaseStationaryScheme
from enviroment import Material
from wraps import timer


class DirectStationaryScheme(BaseStationaryScheme):
    """
    Class template docstring
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple[int, int, int, int],
        material: Material,
        limits: list[np.float64, np.float64],
    ):
        """
        Calculate the square root of a number.

        Args:
            F: The density of the inner heat sources
            G: The density of the bound heat sources
            square_shape: shape of the scheme. [n, n] for discrete methods
                and [n, n, m, m] for direct.
            material: namedtuple object for containing material properties
            limits: description of the computing area, e.g. [0.0, 1.0]
        Returns:
            the instance of the DirectStationaryScheme class.
        """
        super().__init__(F, G, square_shape, material, limits)

        cells = self.square_shape[0]
        cell_size = self.square_shape[2]
        self.h = (self.limits[1] - self.limits[0]) / ((cell_size - 1) * cells)

        self.tcc_n = self.material.thermal_cond * self.w
        self.HeatStream = lambda v: self.stef_bolc * fabs(v) * fpower(v, 3)
        self.dHeatStream = (
            lambda v, dv: 4 * self.stef_bolc * fabs(v) * fpower(v, 2) * dv
        )

    @timer
    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """

        inner_tol = kwargs.get("inner_tol", 5e-4)
        u0 = kwargs.get("u0_squared", 300.0 * np.ones(np.prod(self.square_shape)))
        U = u0.flatten() / self.w

        A = LinearOperator(
            (U.size, U.size),
            matvec=lambda du: self.jacobian(U, du),
        )

        self.G[0, 0, 0, 0] *= 2
        self.G[-1, 0, -1, 0] *= 2
        self.G[-1, -1, -1, -1] *= 2
        self.G[0, -1, 0, -1] *= 2

        b = (self.F + (2 / self.h) * self.G).flatten()
        R = b - self.operator(U)
        dU, exit_code = bicgstab(
            A,
            R,
            rtol=inner_tol,
            atol=0.0,
            x0=R,
        )
        if exit_code:
            print(f"jacobian Failed with exit code: {exit_code} ON THE START")
            U = (self.w * U).reshape(self.square_shape)
            return U, exit_code

        err = np.abs(dU).max()
        print(f"\t{err:.3e}")
        while err > tol:
            U += dU
            R = b - self.operator(U)
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
                U += dU
                U = (self.w * U).reshape(self.square_shape)
                return U, exit_code
            err = np.abs(dU).max()
            print(f"\t{err:.3e}")
        U = (self.w * U).reshape(self.square_shape)
        return U, exit_code

    def operator(self, u_linear: np.ndarray, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        u = u_linear.reshape(self.square_shape)
        res = np.zeros_like(u)
        h = self.h
        h2 = h * h
        tcc_n_h2 = self.material.thermal_cond * self.w / h2
        two_div_h = 2 / h

        HeatStream = self.HeatStream

        res[::, ::, 1:-1, 1:-1] = -tcc_n_h2 * (
            u[::, ::, 2:, 1:-1]
            + u[::, ::, :-2, 1:-1]
            - 4 * u[::, ::, 1:-1, 1:-1]
            + u[::, ::, 1:-1, 2:]
            + u[::, ::, 1:-1, :-2]
        )

        res[::, ::, 0, 1:-1] = (
            -2 * tcc_n_h2 * (u[::, ::, 1, 1:-1] - u[::, ::, 0, 1:-1])
            + two_div_h * HeatStream(u[::, ::, 0, 1:-1])
            - tcc_n_h2 * (u[::, ::, 0, 2:] - 2 * u[::, ::, 0, 1:-1] + u[::, ::, 0, :-2])
        )
        res[::, ::, 1:-1, 0] = (
            -2 * tcc_n_h2 * (u[::, ::, 1:-1, 1] - u[::, ::, 1:-1, 0])
            + two_div_h * HeatStream(u[::, ::, 1:-1, 0])
            - tcc_n_h2 * (u[::, ::, 2:, 0] - 2 * u[::, ::, 1:-1, 0] + u[::, ::, :-2, 0])
        )
        res[::, ::, -1, 1:-1] = (
            2 * tcc_n_h2 * (u[::, ::, -1, 1:-1] - u[::, ::, -2, 1:-1])
            + two_div_h * HeatStream(u[::, ::, -1, 1:-1])
            - tcc_n_h2
            * (u[::, ::, -1, 2:] - 2 * u[::, ::, -1, 1:-1] + u[::, ::, -1, :-2])
        )
        res[::, ::, 1:-1, -1] = (
            2 * tcc_n_h2 * (u[::, ::, 1:-1, -1] - u[::, ::, 1:-1, -2])
            + two_div_h * HeatStream(u[::, ::, 1:-1, -1])
            - tcc_n_h2
            * (u[::, ::, 2:, -1] - 2 * u[::, ::, 1:-1, -1] + u[::, ::, :-2, -1])
        )

        res[::, ::, 0, 0] = 2 * two_div_h * HeatStream(
            u[::, ::, 0, 0]
        ) - 2 * tcc_n_h2 * (u[::, ::, 0, 1] - 2 * u[::, ::, 0, 0] + u[::, ::, 1, 0])
        res[::, ::, 0, -1] = 2 * two_div_h * HeatStream(
            u[::, ::, 0, -1]
        ) - 2 * tcc_n_h2 * (u[::, ::, 0, -2] - 2 * u[::, ::, 0, -1] + u[::, ::, 1, -1])
        res[::, ::, -1, -1] = 2 * two_div_h * HeatStream(
            u[::, ::, -1, -1]
        ) - 2 * tcc_n_h2 * (
            u[::, ::, -1, -2] - 2 * u[::, ::, -1, -1] + u[::, ::, -2, -1]
        )
        res[::, ::, -1, 0] = 2 * two_div_h * HeatStream(
            u[::, ::, -1, 0]
        ) - 2 * tcc_n_h2 * (u[::, ::, -1, 1] - 2 * u[::, ::, -1, 0] + u[::, ::, -2, 0])

        # inside joints
        res[1:, ::, 0, 1:-1] -= two_div_h * HeatStream(u[:-1, ::, -1, 1:-1])
        res[:-1, ::, -1, 1:-1] -= two_div_h * HeatStream(u[1:, ::, 0, 1:-1])
        res[::, 1:, 1:-1, 0] -= two_div_h * HeatStream(u[::, :-1, 1:-1, -1])
        res[::, :-1, 1:-1, -1] -= two_div_h * HeatStream(u[::, 1:, 1:-1, 0])

        # inside corners
        res[:-1, :-1, -1, -1] -= two_div_h * (
            HeatStream(u[1:, :-1, 0, -1]) + HeatStream(u[:-1, 1:, -1, 0])
        )
        res[1:, :-1, 0, -1] -= two_div_h * (
            HeatStream(u[:-1, :-1, -1, -1]) + HeatStream(u[1:, 1:, 0, 0])
        )
        res[1:, 1:, 0, 0] -= two_div_h * (
            HeatStream(u[1:, :-1, 0, -1]) + HeatStream(u[:-1, 1:, -1, 0])
        )
        res[:-1, 1:, -1, 0] -= two_div_h * (
            HeatStream(u[1:, 1:, 0, 0]) + HeatStream(u[:-1, :-1, -1, -1])
        )

        # side corners
        res[0, :-1, 0, -1] -= two_div_h * HeatStream(u[0, 1:, 0, 0])
        res[0, 1:, 0, 0] -= two_div_h * HeatStream(u[0, :-1, 0, -1])
        res[-1, :-1, -1, -1] -= two_div_h * HeatStream(u[-1, 1:, -1, 0])
        res[-1, 1:, -1, 0] -= two_div_h * HeatStream(u[-1, :-1, -1, -1])
        res[:-1, 0, -1, 0] -= two_div_h * HeatStream(u[1:, 0, 0, 0])
        res[1:, 0, 0, 0] -= two_div_h * HeatStream(u[:-1, 0, -1, 0])
        res[:-1, -1, -1, -1] -= two_div_h * HeatStream(u[1:, -1, 0, -1])
        res[1:, -1, 0, -1] -= two_div_h * HeatStream(u[:-1, -1, -1, -1])

        return res.flatten()

    def jacobian(self, u_linear: np.ndarray, du_linear: np.ndarray) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        u = u_linear.reshape(self.square_shape)
        du = du_linear.reshape(self.square_shape)
        res = np.zeros_like(du)
        h = self.h
        h2 = self.h * self.h
        tcc_n_h2 = self.material.thermal_cond * self.w / h2
        two_div_h = 2 / h
        dHeatStream = self.dHeatStream

        # internal area
        res[::, ::, 1:-1, 1:-1] = tcc_n_h2 * (
            -(du[::, ::, 2:, 1:-1] + du[::, ::, :-2, 1:-1] - 2 * du[::, ::, 1:-1, 1:-1])
            - (
                du[::, ::, 1:-1, 2:]
                + du[::, ::, 1:-1, :-2]
                - 2 * du[::, ::, 1:-1, 1:-1]
            )
        )

        # all sides
        res[::, ::, 0, 1:-1] = (
            -2 * tcc_n_h2 * (du[::, ::, 1, 1:-1] - du[::, ::, 0, 1:-1])
            + two_div_h * dHeatStream(u[::, ::, 0, 1:-1], du[::, ::, 0, 1:-1])
            - tcc_n_h2
            * (du[::, ::, 0, 2:] - 2 * du[::, ::, 0, 1:-1] + du[::, ::, 0, :-2])
        )
        res[::, ::, 1:-1, 0] = (
            -2 * tcc_n_h2 * (du[::, ::, 1:-1, 1] - du[::, ::, 1:-1, 0])
            + two_div_h * dHeatStream(u[::, ::, 1:-1, 0], du[::, ::, 1:-1, 0])
            - tcc_n_h2
            * (du[::, ::, 2:, 0] - 2 * du[::, ::, 1:-1, 0] + du[::, ::, :-2, 0])
        )
        res[::, ::, -1, 1:-1] = (
            2 * tcc_n_h2 * (du[::, ::, -1, 1:-1] - du[::, ::, -2, 1:-1])
            + two_div_h * dHeatStream(u[::, ::, -1, 1:-1], du[::, ::, -1, 1:-1])
            - tcc_n_h2
            * (du[::, ::, -1, 2:] - 2 * du[::, ::, -1, 1:-1] + du[::, ::, -1, :-2])
        )
        res[::, ::, 1:-1, -1] = (
            2 * tcc_n_h2 * (du[::, ::, 1:-1, -1] - du[::, ::, 1:-1, -2])
            + two_div_h * dHeatStream(u[::, ::, 1:-1, -1], du[::, ::, 1:-1, -1])
            - tcc_n_h2
            * (du[::, ::, 2:, -1] - 2 * du[::, ::, 1:-1, -1] + du[::, ::, :-2, -1])
        )

        # inner sides
        res[1:, ::, 0, 1:-1] -= two_div_h * dHeatStream(
            u[:-1, ::, -1, 1:-1], du[:-1, ::, -1, 1:-1]
        )
        res[:-1, ::, -1, 1:-1] -= two_div_h * dHeatStream(
            u[1:, ::, 0, 1:-1], du[1:, ::, 0, 1:-1]
        )
        res[::, 1:, 1:-1, 0] -= two_div_h * dHeatStream(
            u[::, :-1, 1:-1, -1], du[::, :-1, 1:-1, -1]
        )
        res[::, :-1, 1:-1, -1] -= two_div_h * dHeatStream(
            u[::, 1:, 1:-1, 0], du[::, 1:, 1:-1, 0]
        )

        # all corners
        res[::, ::, 0, 0] = 2 * two_div_h * dHeatStream(
            u[::, ::, 0, 0], du[::, ::, 0, 0]
        ) - 2 * tcc_n_h2 * (du[::, ::, 0, 1] - 2 * du[::, ::, 0, 0] + du[::, ::, 1, 0])
        res[::, ::, 0, -1] = 2 * two_div_h * dHeatStream(
            u[::, ::, 0, -1], du[::, ::, 0, -1]
        ) - 2 * tcc_n_h2 * (
            du[::, ::, 0, -2] - 2 * du[::, ::, 0, -1] + du[::, ::, 1, -1]
        )
        res[::, ::, -1, -1] = 2 * two_div_h * dHeatStream(
            u[::, ::, -1, -1], du[::, ::, -1, -1]
        ) - 2 * tcc_n_h2 * (
            du[::, ::, -1, -2] - 2 * du[::, ::, -1, -1] + du[::, ::, -2, -1]
        )
        res[::, ::, -1, 0] = 2 * two_div_h * dHeatStream(
            u[::, ::, -1, 0], du[::, ::, -1, 0]
        ) - 2 * tcc_n_h2 * (
            du[::, ::, -1, 1] - 2 * du[::, ::, -1, 0] + du[::, ::, -2, 0]
        )

        # inner corners
        res[:-1, :-1, -1, -1] -= two_div_h * (
            dHeatStream(u[1:, :-1, 0, -1], du[1:, :-1, 0, -1])
            + dHeatStream(u[:-1, 1:, -1, 0], du[:-1, 1:, -1, 0])
        )
        res[1:, :-1, 0, -1] -= two_div_h * (
            dHeatStream(u[:-1, :-1, -1, -1], du[:-1, :-1, -1, -1])
            + dHeatStream(u[1:, 1:, 0, 0], du[1:, 1:, 0, 0])
        )
        res[1:, 1:, 0, 0] -= two_div_h * (
            dHeatStream(u[1:, :-1, 0, -1], du[1:, :-1, 0, -1])
            + dHeatStream(u[:-1, 1:, -1, 0], du[:-1, 1:, -1, 0])
        )
        res[:-1, 1:, -1, 0] -= two_div_h * (
            dHeatStream(u[1:, 1:, 0, 0], du[1:, 1:, 0, 0])
            + dHeatStream(u[:-1, :-1, -1, -1], du[:-1, :-1, -1, -1])
        )

        # outer corners
        res[0, :-1, 0, -1] -= two_div_h * dHeatStream(u[0, 1:, 0, 0], du[0, 1:, 0, 0])
        res[0, 1:, 0, 0] -= two_div_h * dHeatStream(u[0, :-1, 0, -1], du[0, :-1, 0, -1])
        res[-1, :-1, -1, -1] -= two_div_h * dHeatStream(
            u[-1, 1:, -1, 0], du[-1, 1:, -1, 0]
        )
        res[-1, 1:, -1, 0] -= two_div_h * dHeatStream(
            u[-1, :-1, -1, -1], du[-1, :-1, -1, -1]
        )
        res[:-1, 0, -1, 0] -= two_div_h * dHeatStream(u[1:, 0, 0, 0], du[1:, 0, 0, 0])
        res[1:, 0, 0, 0] -= two_div_h * dHeatStream(u[:-1, 0, -1, 0], du[:-1, 0, -1, 0])
        res[:-1, -1, -1, -1] -= two_div_h * dHeatStream(
            u[1:, -1, 0, -1], du[1:, -1, 0, -1]
        )
        res[1:, -1, 0, -1] -= two_div_h * dHeatStream(
            u[:-1, -1, -1, -1], du[:-1, -1, -1, -1]
        )

        return res.flatten()

    @staticmethod
    def GetBoundaries(
        f_func,
        g_func: list,
        square_shape: tuple[int, int],
        material: Material,
        limits: list[np.float64, np.float64],
        stef_bolc: np.float64,
    ):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        HeatStream = lambda t: stef_bolc * np.power(t, 4)

        f = lambda x, y: HeatStream(f_func(x, y))
        g = [
            lambda t: HeatStream(g_func[0](t)),
            lambda t: HeatStream(g_func[1](t)),
            lambda t: HeatStream(g_func[2](t)),
            lambda t: HeatStream(g_func[3](t)),
        ]

        ############

        cells = square_shape[0]
        cell_size = square_shape[2]
        h = (limits[1] - limits[0]) / ((cell_size - 1) * cells)

        F: np.ndarray = np.zeros(square_shape)
        G: np.ndarray = np.zeros(square_shape)

        for i in range(cells):
            for j in range(cells):
                for i2 in range(cell_size):
                    for j2 in range(cell_size):
                        F[i, j, i2, j2] = f(
                            (i * (cell_size - 1) + i2) * h,
                            (j * (cell_size - 1) + j2) * h,
                        )
        for k in range(cells):
            for k2 in range(cell_size):
                G[k, 0, k2, 0] = g[0]((k * (cell_size - 1) + k2) * h)
                G[k, -1, k2, -1] = g[2]((k * (cell_size - 1) + k2) * h)
                G[0, k, 0, k2] = g[3]((k * (cell_size - 1) + k2) * h)
                G[-1, k, -1, k2] = g[1]((k * (cell_size - 1) + k2) * h)

        return [F, G]

    def flatten(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        mod = kwargs.get("mod", 0)
        cells = self.square_shape[0]
        cell_size = self.square_shape[2]
        if mod == 0:
            flat_size = cells * cell_size
            flat_shape = (flat_size, flat_size)

            u_flat = np.zeros(flat_shape)

            for i in range(cells):
                for j in range(cells):
                    for i2 in range(cell_size):
                        for j2 in range(cell_size):
                            u_flat[i * cell_size + i2, j * cell_size + j2] = u_squared[
                                i, j, i2, j2
                            ]
        else:
            h2 = self.h * self.h
            u_flat = np.zeros(shape=(cells, cells))
            for i_cell in range(cells):
                for j_cell in range(cells):
                    cur_cell = u_squared[i_cell, j_cell]
                    u_flat[i_cell, j_cell] += np.sum(cur_cell[1:-1, 1:-1]) * h2
                    u_flat[i_cell, j_cell] += (
                        h2
                        * 0.5
                        * (
                            np.sum(cur_cell[1:-1, 0])
                            + np.sum(cur_cell[1:-1, -1])
                            + np.sum(cur_cell[0, 1:-1])
                            + np.sum(cur_cell[-1, 1:-1])
                        )
                    )
                    u_flat[i_cell, j_cell] += (
                        h2
                        * 0.25
                        * (
                            cur_cell[0, 0]
                            + cur_cell[0, -1]
                            + cur_cell[-1, 0]
                            + cur_cell[-1, -1]
                        )
                    )
                    u_flat[i_cell, j_cell] *= (
                        cells / (self.limits[1] - self.limits[0])
                    ) ** 2
        return u_flat
