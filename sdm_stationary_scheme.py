"""
Module template docstring
"""

import numpy as np
from scipy.integrate import quad  # , nquad
from scipy.sparse.linalg import LinearOperator, bicgstab

from base_stationary_scheme import BaseStationaryScheme
from enviroment import Material
from wraps import timer


class SDMStationaryScheme(BaseStationaryScheme):
    """
    Class template docstring
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple[int, int],
        material: Material,
        limits: list[np.float64, np.float64],
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
        self.h = (self.limits[1] - self.limits[0]) / cells

        self.H, self.dH = SDMStationaryScheme.createH(
            self.h, self.material.thermal_cond, self.stef_bolc
        )
        self.B, self.dB = SDMStationaryScheme.createH(
            self.h, 2.0 * self.material.thermal_cond, self.stef_bolc
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

        self.H, self.dH = SDMStationaryScheme.createH(
            self.h, self.material.thermal_cond, self.stef_bolc
        )
        self.B, self.dB = SDMStationaryScheme.createH(
            self.h, 2.0 * self.material.thermal_cond, self.stef_bolc
        )

        inner_tol = kwargs.get("inner_tol", 5e-4)
        u0 = kwargs.get("u0_squared", 300.0 * np.ones(np.prod(self.square_shape)))
        U = u0.flatten() / self.w

        A = LinearOperator(
            (U.size, U.size),
            matvec=lambda du: self.jacobian(U, du),
        )
        b = (self.F + self.G).flatten()
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
        print(f"{err:.3e}")
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
            print(f"{err:.3e}")
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
        H, B = self.H, self.B

        # internal cells
        res[1:-1, 1:-1] = (
            -1
            / self.h
            * (
                H(u[2:, 1:-1])
                + H(u[:-2, 1:-1])
                + H(u[1:-1, 2:])
                + H(u[1:-1, :-2])
                - 4.0 * H(u[1:-1, 1:-1])
            )
        )

        # edge cells
        res[1:-1, 0] = (
            H(u[1:-1, 0])
            - H(u[1:-1, 1])
            - (H(u[2:, 0]) - 2.0 * H(u[1:-1, 0]) + H(u[:-2, 0]))
            + B(u[1:-1, 0])
        )
        res[-1, 1:-1] = (
            H(u[-1, 1:-1])
            - H(u[-2, 1:-1])
            - (H(u[-1, 2:]) - 2.0 * H(u[-1, 1:-1]) + H(u[-1, :-2]))
            + B(u[-1, 1:-1])
        )
        res[1:-1, -1] = (
            H(u[1:-1, -1])
            - H(u[1:-1, -2])
            - (H(u[2:, -1]) - 2.0 * H(u[1:-1, -1]) + H(u[:-2, -1]))
            + B(u[1:-1, -1])
        )
        res[0, 1:-1] = (
            H(u[0, 1:-1])
            - H(u[1, 1:-1])
            - (H(u[0, 2:]) - 2.0 * H(u[0, 1:-1]) + H(u[0, :-2]))
            + B(u[0, 1:-1])
        )

        # corner cells
        res[0, 0] = H(u[0, 0]) - 0.5 * (H(u[1, 0]) + H(u[0, 1])) + B(u[0, 0])
        res[-1, 0] = H(u[-1, 0]) - 0.5 * (H(u[-2, 0]) + H(u[-1, 1])) + B(u[-1, 0])
        res[-1, -1] = H(u[-1, -1]) - 0.5 * (H(u[-2, -1]) + H(u[-1, -2])) + B(u[-1, -1])
        res[0, -1] = H(u[0, -1]) - 0.5 * (H(u[1, -1]) + H(u[0, -2])) + B(u[0, -1])

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
        res = np.zeros_like(u)
        dH, dB = self.dH, self.dB
        w = self.w

        # internal cells
        res[1:-1, 1:-1] = (
            -1
            / self.h
            * (
                dH(u[2:, 1:-1]) * w * du[2:, 1:-1]
                + dH(u[:-2, 1:-1]) * w * du[:-2, 1:-1]
                + dH(u[1:-1, 2:]) * w * du[1:-1, 2:]
                + dH(u[1:-1, :-2]) * w * du[1:-1, :-2]
                - 4.0 * dH(u[1:-1, 1:-1]) * w * du[1:-1, 1:-1]
            )
        )

        # edge cells
        res[1:-1, 0] = (
            dH(u[1:-1, 0]) * w * du[1:-1, 0]
            - dH(u[1:-1, 1]) * w * du[1:-1, 1]
            - (
                dH(u[2:, 0]) * w * du[2:, 0]
                - 2.0 * dH(u[1:-1, 0]) * w * du[1:-1, 0]
                + dH(u[:-2, 0]) * w * du[:-2, 0]
            )
            + dB(u[1:-1, 0]) * w * du[1:-1, 0]
        )
        res[-1, 1:-1] = (
            dH(u[-1, 1:-1]) * w * du[-1, 1:-1]
            - dH(u[-2, 1:-1]) * w * du[-2, 1:-1]
            - (
                dH(u[-1, 2:]) * w * du[-1, 2:]
                - 2.0 * dH(u[-1, 1:-1]) * w * du[-1, 1:-1]
                + dH(u[-1, :-2]) * w * du[-1, :-2]
            )
            + dB(u[-1, 1:-1]) * w * du[-1, 1:-1]
        )
        res[1:-1, -1] = (
            dH(u[1:-1, -1]) * w * du[1:-1, -1]
            - dH(u[1:-1, -2]) * w * du[1:-1, -2]
            - (
                dH(u[2:, -1]) * w * du[2:, -1]
                - 2.0 * dH(u[1:-1, -1]) * w * du[1:-1, -1]
                + dH(u[:-2, -1]) * w * du[:-2, -1]
            )
            + dB(u[1:-1, -1]) * w * du[1:-1, -1]
        )
        res[0, 1:-1] = (
            dH(u[0, 1:-1]) * w * du[0, 1:-1]
            - dH(u[1, 1:-1]) * w * du[1, 1:-1]
            - (
                dH(u[0, 2:]) * w * du[0, 2:]
                - 2.0 * dH(u[0, 1:-1]) * w * du[0, 1:-1]
                + dH(u[0, :-2]) * w * du[0, :-2]
            )
            + dB(u[0, 1:-1]) * w * du[0, 1:-1]
        )

        # corner cells
        res[0, 0] = (
            dH(u[0, 0]) * w * du[0, 0]
            - 0.5 * (dH(u[1, 0]) * w * du[1, 0] + dH(u[0, 1]) * w * du[0, 1])
            + dB(u[0, 0]) * w * du[0, 0]
        )
        res[-1, 0] = (
            dH(u[-1, 0]) * w * du[-1, 0]
            - 0.5 * (dH(u[-2, 0]) * w * du[-2, 0] + dH(u[-1, 1]) * w * du[-1, 1])
            + dB(u[-1, 0]) * w * du[-1, 0]
        )
        res[-1, -1] = (
            dH(u[-1, -1]) * w * du[-1, -1]
            - 0.5 * (dH(u[-2, -1]) * w * du[-2, -1] + dH(u[-1, -2]) * w * du[-1, -2])
            + dB(u[-1, -1]) * w * du[-1, -1]
        )
        res[0, -1] = (
            dH(u[0, -1]) * w * du[0, -1]
            - 0.5 * (dH(u[1, -1]) * w * du[1, -1] + dH(u[0, -2]) * w * du[0, -2])
            + dB(u[0, -1]) * w * du[0, -1]
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
        h: np.float64 = (limits[1] - limits[0]) / square_shape[0]

        # HeatStream, _ = SDMStationaryScheme.createH(h, material.thermal_cond, stef_bolc)
        BoundaryHeatStream, _ = SDMStationaryScheme.createH(
            h, 2.0 * material.thermal_cond, stef_bolc
        )

        # f = lambda x, y: HeatStream(f_func(x, y))
        g = [
            lambda t: BoundaryHeatStream(g_func[0](t)),
            lambda t: BoundaryHeatStream(g_func[1](t)),
            lambda t: BoundaryHeatStream(g_func[2](t)),
            lambda t: BoundaryHeatStream(g_func[3](t)),
        ]

        F: np.ndarray = np.zeros(square_shape)
        G: np.ndarray = np.zeros(square_shape)

        # for i in range(square_shape[0]):
        #     x1 = i * h
        #     for j in range(square_shape[0]):
        #         x2 = j * h
        #         F[i, j], _ = nquad(f, [[x1, x1 + h], [x2, x2 + h]])

        for k in range(1, square_shape[0] - 1):
            x = k * h
            G[k, 0], _ = quad(g[0], x, x + h)
            G[-1, k], _ = quad(g[1], x, x + h)
            G[k, -1], _ = quad(g[2], x, x + h)
            G[0, k], _ = quad(g[3], x, x + h)

        G[0, 0] = 0.5 * (
            quad(g[0], limits[0], limits[0] + h)[0]
            + quad(g[3], limits[0], limits[0] + h)[0]
        )
        G[-1, 0] = 0.5 * (
            quad(g[0], limits[1] - h, limits[1])[0]
            + quad(g[1], limits[0], limits[0] + h)[0]
        )
        G[-1, -1] = 0.5 * (
            quad(g[1], limits[1] - h, limits[1])[0]
            + quad(g[2], limits[1] - h, limits[1])[0]
        )
        G[0, -1] = 0.5 * (
            quad(g[2], limits[0], limits[0] + h)[0]
            + quad(g[3], limits[1] - h, limits[1])[0]
        )

        F /= h**2
        G /= h

        return [F, G]

    @staticmethod
    def createH(h: np.float64, thermal_cond: np.float64, stef_bolc: np.float64):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        w = 100.0 if stef_bolc > 1.0 else 1.0
        a = h / thermal_cond
        b = np.float_power(4.0 * stef_bolc / w * a, 1.0 / 3.0)
        b2 = b * b
        sq3 = np.sqrt(3.0)
        h_0 = np.pi / (6.0 * sq3 * b)

        H = (
            lambda v: w
            / a
            * (
                v
                + np.log(b2 * v**2 - b * v + 1.0) / (6.0 * b)
                - np.arctan((2.0 * b * v - 1.0) / sq3) / (sq3 * b)
                - np.log(b * v + 1.0) / (3.0 * b)
                - h_0
            )
        )

        dH = (
            lambda v: 4.0
            * stef_bolc
            / w
            * v**3
            / (1.0 + 4.0 * stef_bolc / w * a * v**3)
        )

        return H, dH

    def flatten(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        return u_squared
