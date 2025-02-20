"""
Direct Stationary Scheme class module.
"""

import numpy as np
from numpy import float_power as fpower, fabs

from stat_scheme.base_stationary_scheme import BaseStationaryScheme
from utils import Material


class DirectStationaryScheme(BaseStationaryScheme):
    """
    Direct Stationary scheme - scheme to obtain the most exact solution
    of the complex heat conductivity sationary question in
    2d slice of the square rods pack.

    Used to compare other approx schemes' correctness
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple[int, int, int, int],
        material: Material,
        limits: list[np.float64, np.float64],
        **kwargs,
    ):
        """
        Args:
            F: The inner heat
            G: The bound heat
            square_shape: shape of the scheme. [n, n] for discrete methods
                and [n, n, m, m] for direct.
            material: namedtuple object for containing material properties
            limits: description of the computing area, [a, b] for
                stationary schemes.
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

        mask = (2 / self.h) * np.ones_like(self.G)

        mask[0, 0, 0, 0] *= 2
        mask[-1, 0, -1, 0] *= 2
        mask[-1, -1, -1, -1] *= 2
        mask[0, -1, 0, -1] *= 2

        self.b = self.F + mask * self.G

    def operator(self, u_linear: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes A(u), where A - is the differential equations system's scheme operator
        (non linear).

        Use in solve to get (R)esidual = b - A(u)

        Args:
            u_linear: 1-d array of the u - temperature.
        Returns:
            res: 1-d array of the A(u).
        """
        u = u_linear.reshape(self.square_shape)
        res = np.zeros_like(u)
        h = self.h
        h2 = h * h
        tcc_n_h2 = self.material.thermal_cond * self.w / h2
        two_div_h = 2 / h

        HeatStream = self.HeatStream

        res[..., 1:-1, 1:-1] = -tcc_n_h2 * (u[..., 2:, 1:-1] + u[..., :-2, 1:-1]
            - 4 * u[..., 1:-1, 1:-1]
            + u[..., 1:-1, 2:]
            + u[..., 1:-1, :-2]
        )

        res[..., 0, 1:-1] = (
            -2 * tcc_n_h2 * (u[..., 1, 1:-1] - u[..., 0, 1:-1])
            + two_div_h * HeatStream(u[..., 0, 1:-1])
            - tcc_n_h2 * (u[..., 0, 2:] - 2 * u[..., 0, 1:-1] + u[..., 0, :-2])
        )
        res[..., 1:-1, 0] = (
            -2 * tcc_n_h2 * (u[..., 1:-1, 1] - u[..., 1:-1, 0])
            + two_div_h * HeatStream(u[..., 1:-1, 0])
            - tcc_n_h2 * (u[..., 2:, 0] - 2 * u[..., 1:-1, 0] + u[..., :-2, 0])
        )
        res[..., -1, 1:-1] = (
            2 * tcc_n_h2 * (u[..., -1, 1:-1] - u[..., -2, 1:-1])
            + two_div_h * HeatStream(u[..., -1, 1:-1])
            - tcc_n_h2
            * (u[..., -1, 2:] - 2 * u[..., -1, 1:-1] + u[..., -1, :-2])
        )
        res[..., 1:-1, -1] = (
            2 * tcc_n_h2 * (u[..., 1:-1, -1] - u[..., 1:-1, -2])
            + two_div_h * HeatStream(u[..., 1:-1, -1])
            - tcc_n_h2
            * (u[..., 2:, -1] - 2 * u[..., 1:-1, -1] + u[..., :-2, -1])
        )

        res[..., 0, 0] = 2 * two_div_h * HeatStream(
            u[..., 0, 0]
        ) - 2 * tcc_n_h2 * (u[..., 0, 1] - 2 * u[..., 0, 0] + u[..., 1, 0])
        res[..., 0, -1] = 2 * two_div_h * HeatStream(
            u[..., 0, -1]
        ) - 2 * tcc_n_h2 * (u[..., 0, -2] - 2 * u[..., 0, -1] + u[..., 1, -1])
        res[..., -1, -1] = 2 * two_div_h * HeatStream(
            u[..., -1, -1]
        ) - 2 * tcc_n_h2 * (
            u[..., -1, -2] - 2 * u[..., -1, -1] + u[..., -2, -1]
        )
        res[..., -1, 0] = 2 * two_div_h * HeatStream(
            u[..., -1, 0]
        ) - 2 * tcc_n_h2 * (u[..., -1, 1] - 2 * u[..., -1, 0] + u[..., -2, 0])

        # inside joints
        res[1:, :, 0, 1:-1] -= two_div_h * HeatStream(u[:-1, :, -1, 1:-1])
        res[:-1, :, -1, 1:-1] -= two_div_h * HeatStream(u[1:, :, 0, 1:-1])
        res[:, 1:, 1:-1, 0] -= two_div_h * HeatStream(u[:, :-1, 1:-1, -1])
        res[:, :-1, 1:-1, -1] -= two_div_h * HeatStream(u[:, 1:, 1:-1, 0])

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
        Computes gradient value in the U-point of A-operator.
        Use in Newton's method by BiCGstab as matvec to find newton's dU.
        For computing uses current newton's approx of the U-function.

        Args:
            u_linear: 1d-shape array of U function.
            du_linear: 1d-shape array of dU function.
        Returns:
            Jac(U, dU): 1d_shape array.
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
        res[..., 1:-1, 1:-1] = tcc_n_h2 * (
            -(du[..., 2:, 1:-1] + du[..., :-2, 1:-1] - 2 * du[..., 1:-1, 1:-1])
            - (
                du[..., 1:-1, 2:]
                + du[..., 1:-1, :-2]
                - 2 * du[..., 1:-1, 1:-1]
            )
        )

        # all sides
        res[..., 0, 1:-1] = (
            -2 * tcc_n_h2 * (du[..., 1, 1:-1] - du[..., 0, 1:-1])
            + two_div_h * dHeatStream(u[..., 0, 1:-1], du[..., 0, 1:-1])
            - tcc_n_h2
            * (du[..., 0, 2:] - 2 * du[..., 0, 1:-1] + du[..., 0, :-2])
        )
        res[..., 1:-1, 0] = (
            -2 * tcc_n_h2 * (du[..., 1:-1, 1] - du[..., 1:-1, 0])
            + two_div_h * dHeatStream(u[..., 1:-1, 0], du[..., 1:-1, 0])
            - tcc_n_h2
            * (du[..., 2:, 0] - 2 * du[..., 1:-1, 0] + du[..., :-2, 0])
        )
        res[..., -1, 1:-1] = (
            2 * tcc_n_h2 * (du[..., -1, 1:-1] - du[..., -2, 1:-1])
            + two_div_h * dHeatStream(u[..., -1, 1:-1], du[..., -1, 1:-1])
            - tcc_n_h2
            * (du[..., -1, 2:] - 2 * du[..., -1, 1:-1] + du[..., -1, :-2])
        )
        res[..., 1:-1, -1] = (
            2 * tcc_n_h2 * (du[..., 1:-1, -1] - du[..., 1:-1, -2])
            + two_div_h * dHeatStream(u[..., 1:-1, -1], du[..., 1:-1, -1])
            - tcc_n_h2
            * (du[..., 2:, -1] - 2 * du[..., 1:-1, -1] + du[..., :-2, -1])
        )

        # inner sides
        res[1:, :, 0, 1:-1] -= two_div_h * dHeatStream(
            u[:-1, :, -1, 1:-1], du[:-1, :, -1, 1:-1]
        )
        res[:-1, :, -1, 1:-1] -= two_div_h * dHeatStream(
            u[1:, :, 0, 1:-1], du[1:, :, 0, 1:-1]
        )
        res[:, 1:, 1:-1, 0] -= two_div_h * dHeatStream(
            u[:, :-1, 1:-1, -1], du[:, :-1, 1:-1, -1]
        )
        res[:, :-1, 1:-1, -1] -= two_div_h * dHeatStream(
            u[:, 1:, 1:-1, 0], du[:, 1:, 1:-1, 0]
        )

        # all corners
        res[..., 0, 0] = 2 * two_div_h * dHeatStream(
            u[..., 0, 0], du[..., 0, 0]
        ) - 2 * tcc_n_h2 * (du[..., 0, 1] - 2 * du[..., 0, 0] + du[..., 1, 0])
        res[..., 0, -1] = 2 * two_div_h * dHeatStream(
            u[..., 0, -1], du[..., 0, -1]
        ) - 2 * tcc_n_h2 * (
            du[..., 0, -2] - 2 * du[..., 0, -1] + du[..., 1, -1]
        )
        res[..., -1, -1] = 2 * two_div_h * dHeatStream(
            u[..., -1, -1], du[..., -1, -1]
        ) - 2 * tcc_n_h2 * (
            du[..., -1, -2] - 2 * du[..., -1, -1] + du[..., -2, -1]
        )
        res[..., -1, 0] = 2 * two_div_h * dHeatStream(
            u[..., -1, 0], du[..., -1, 0]
        ) - 2 * tcc_n_h2 * (
            du[..., -1, 1] - 2 * du[..., -1, 0] + du[..., -2, 0]
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
        **kwargs,
    ):
        """
        Static function to obtain F and G arrays.
        Parameters are the same as in __init__()

        Args:
            f_func: the temperature of the inner heat sources.
            g_func: list of 4 functions g(x, y) for the bound temperature:
                [g(x=[a,b], y=a), g(x=b, y=[a, b]), g(x=[a,b], y=b), g(x=a, y=[a,b])]
            square_shape: shape of the scheme. [n, n] for discrete methods
                and [n, n, m, m] for direct.
            limits: description of the computing area, [a, b] for
                stationary schemes and [a, b, T] for non-stationary
        Returns:
            [F, G]
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

    @staticmethod
    def flatten(
        u_squared: np.ndarray,
        limits: list[np.float64, np.float64],
        **kwargs,
    ) -> np.ndarray:
        """
        Method to change ndarray-s to the [cells, cells] format.

        Args:
            u_squared: self.square_shape-like np.ndarray object.
            limits: description of the computing area, [a, b] for stationary
            mod: 0 - returns max information. 1 - makes array compatible with DM solutions
        Returns:
            res: ndarray with shape (*self.square_shape[:2])
        """
        mod = kwargs.get("mod", 0)
        cells = u_squared.shape[0]
        cell_size = u_squared.shape[2]
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
            h = (limits[1] - limits[0]) / ((cell_size - 1) * cells)
            h2 = h * h
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
                        cells / (limits[1] - limits[0])
                    ) ** 2
        return u_flat
