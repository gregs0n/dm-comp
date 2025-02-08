"""
DMNonStationaryScheme class module.
"""

import numpy as np
from scipy.integrate import quad # , nquad

from nonstat_scheme.base_non_stationary_scheme import BaseNonStationaryScheme
from stat_scheme import DMStationaryScheme
from utils import Material


class DMNonStationaryScheme(BaseNonStationaryScheme, DMStationaryScheme):
    """
    Discrete Method's Scheme.
    Helps to compute the result faster than Direct scheme.
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
            use_sdm: tells wether account material's thermal conductivity or not.
        Returns:
            the instance of the DMNonStationaryScheme class.
        """
        super().__init__(F, G, square_shape, material, dt, limits)
        DMStationaryScheme.__init__(self, F, G, square_shape, material, limits, use_sdm=kwargs.get("use_sdm", False))

        self.mask = np.ones(self.square_shape)

        self.mask[:, 0] *= self.h
        self.mask[:, -1] *= self.h
        self.mask[0, 1:-1] *= self.h
        self.mask[-1, 1:-1] *= self.h

        self.mask[0, 0] *= 0.5
        self.mask[-1, 0] *= 0.5
        self.mask[-1, -1] *= 0.5
        self.mask[0, -1] *= 0.5

        self.mask = self.mask.flatten()


    def operator(self, u_linear: np.ndarray, **kwargs) -> np.ndarray:
        """
        TComputes c_rho * du/dt + `DMStationaryScheme.operator(u_linear)`

        Args:
            u_linear: 1-d array of the u - temperature.
            u_prev_linear: U array on the previous layer.
        Returns:
            1-d array  of the c_rho * du/dt + A(u)
        """

        return (
            DMStationaryScheme.operator(self, u_linear)
            + self.mask * (u_linear - kwargs["u_prev_linear"]) * self.material.crho *  self.w / self.dt
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
            c_rho*du + `DMStationaryScheme.jacobian(U, dU)`: 1d_shape array.
        """
        return (
            DMStationaryScheme.jacobian(self, u_linear, du_linear)
            + self.material.crho * self.mask * self.w * du_linear / self.dt
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
            use_sdm: tells wether account material's thermal conductivity or not.
        Returns:
            [F, G]
        """
        h: np.float64 = (limits[1] - limits[0]) / square_shape[0]
        dt = kwargs.get("dt", 0.1)

        use_sdm = kwargs.get("use_sdm", True)
        t_array = np.arange(0, limits[2] + 0.5 * dt, dt)

        # HeatStream, _ = DMStationaryScheme.createH(h, material.thermal_cond, stef_bolc, use_sdm)
        BoundaryHeatStream, _ = DMStationaryScheme.createH(
            h, 2.0 * material.thermal_cond, stef_bolc, use_sdm
        )

        # f = lambda x, y, T: HeatStream(f_func(T, x, y))
        g = [
            lambda t, T: BoundaryHeatStream(g_func[0](T, t)),
            lambda t, T: BoundaryHeatStream(g_func[1](T, t)),
            lambda t, T: BoundaryHeatStream(g_func[2](T, t)),
            lambda t, T: BoundaryHeatStream(g_func[3](T, t)),
        ]

        ############

        F: np.ndarray = np.zeros((t_array.size, *square_shape))
        G: np.ndarray = np.zeros((t_array.size, *square_shape))

        for layer, t_arg in enumerate(t_array):
            # for i in range(square_shape[0]):
            #     x1 = i * h
            #     for j in range(square_shape[0]):
            #         x2 = j * h
            #         F[i, j], _ = nquad(f, [[x1, x1 + h], [x2, x2 + h]], args=(t_arg,))

            for k in range(1, square_shape[0] - 1):
                x = k * h
                G[layer, k, 0], _ = quad(g[0], x, x + h, args=(t_arg,))
                G[layer, -1, k], _ = quad(g[1], x, x + h, args=(t_arg,))
                G[layer, k, -1], _ = quad(g[2], x, x + h, args=(t_arg,))
                G[layer, 0, k], _ = quad(g[3], x, x + h, args=(t_arg,))

            G[layer, 0, 0] = 0.5 * (
                quad(g[0], limits[0], limits[0] + h, args=(t_arg,))[0]
                + quad(g[3], limits[0], limits[0] + h, args=(t_arg,))[0]
            )
            G[layer, -1, 0] = 0.5 * (
                quad(g[0], limits[1] - h, limits[1], args=(t_arg,))[0]
                + quad(g[1], limits[0], limits[0] + h, args=(t_arg,))[0]
            )
            G[layer, -1, -1] = 0.5 * (
                quad(g[1], limits[1] - h, limits[1], args=(t_arg,))[0]
                + quad(g[2], limits[1] - h, limits[1], args=(t_arg,))[0]
            )
            G[layer, 0, -1] = 0.5 * (
                quad(g[2], limits[0], limits[0] + h, args=(t_arg,))[0]
                + quad(g[3], limits[1] - h, limits[1], args=(t_arg,))[0]
            )

        F /= h**2
        G /= h

        return [F, G]
