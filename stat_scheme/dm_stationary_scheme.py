"""
DMStationaryScheme class module.
"""

import numpy as np
from scipy.integrate import quad  # , nquad

from stat_scheme.base_stationary_scheme import BaseStationaryScheme
from utils import Material


class DMStationaryScheme(BaseStationaryScheme):
    """
    Discrete Method's Scheme.
    Helps to compute the result faster than Direct scheme.
    """

    def __init__(
        self,
        F: np.ndarray,
        G: np.ndarray,
        square_shape: tuple[int, int],
        material: Material,
        limits: list[np.float64, np.float64],
        **kwargs,
    ):
        """
        Args:
            F: The inner heat
            G: The bound heat
            square_shape: shape of the scheme, e.g. [cells, cells].
            material: namedtuple object for containing material properties
            limits: description of the computing area.
                e.g. [a, b] = [0.0, 1.0].
            use_sdm: tells wether account material's thermal conductivity or not.
        Returns:
            the instance of the DMStationaryScheme class.
        """
        super().__init__(F, G, square_shape, material, limits)

        cells = self.square_shape[0]
        self.h = (self.limits[1] - self.limits[0]) / cells

        use_sm = kwargs.get("use_sm", True)

        self.H, self.dH = DMStationaryScheme.createH(
            self.h, self.material.thermal_cond, self.stef_bolc, use_sm
        )
        self.B, self.dB = DMStationaryScheme.createH(
            self.h, 2.0 * self.material.thermal_cond, self.stef_bolc, use_sm
        )


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
        h: np.float64 = (limits[1] - limits[0]) / square_shape[0]
        use_sm = kwargs.get("use_sm", False)

        # HeatStream, _ = DMStationaryScheme.createH(h, material.thermal_cond, stef_bolc, use_sm)
        BoundaryHeatStream, _ = DMStationaryScheme.createH(
            h, 2.0 * material.thermal_cond, stef_bolc, use_sm
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
    def createH(
        h: np.float64,
        thermal_cond: np.float64,
        stef_bolc: np.float64,
        use_sm: bool = False
        ):
        """
        Creates function H and its derivative to compute heat stream of the cell

        Args:
            h: cell's side length
            thermal_cond: material's thermal cond.
            stef_bolc: Stefan-Boltzman constant (may be normed)
            use_sm: tells wether account material's thermal conductivity or not.
        Returns:
            H, dH
        """
        w = 100.0 if stef_bolc > 1.0 else 1.0

        if not use_sm:
            H = lambda v: stef_bolc * np.power(v, 4)

            dH = lambda v: 4.0 * stef_bolc / w * v**3
        else:
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

    @staticmethod
    def flatten(
        u_squared: np.ndarray,
        limits: list[np.float64, np.float64],
        **kwargs,
    ) -> np.ndarray:
        """
        does nothing, used nowhere.
        May be removed soon.
        """
        return u_squared
