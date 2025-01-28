"""
FDMStationaryScheme class module.
"""

import numpy as np
from scipy.integrate import quad  # , nquad
from scipy.sparse.linalg import LinearOperator, bicgstab

from base_stationary_scheme import BaseStationaryScheme
from enviroment import Material
from wraps import timer


class FDMStationaryScheme(BaseStationaryScheme):
    """
    First Discrete Method's Scheme.
    Helps to compute the result much faster than Direct scheme.
    Does not use tcc (thermal conductivity coefficient),
    what makes the result much worse with small tcc-s.
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
        Args:
            F: The inner heat
            G: The bound heat
            square_shape: shape of the scheme, e.g. [cells, cells].
            material: namedtuple object for containing material properties
            limits: description of the computing area.
                e.g. [a, b] = [0.0, 1.0].
        Returns:
            the instance of the FDMStationaryScheme class.
        """
        super().__init__(F, G, square_shape, material, limits)
        cells = self.square_shape[0]
        self.h = (self.limits[1] - self.limits[0]) / cells

    @timer
    def solve(self, tol: np.float64, *args, **kwargs) -> np.ndarray:
        """
        Main method to solve the scheme

        Args:
            tol: absolute tolerance of Newton's method. 
            u0_squared: start point for computing the result.
                Explicitly pass like keyword argument.
        Returns:
            The solution of the scheme.
        """
        # inner_tol = kwargs.get("inner_tol", 5e-4)
        u0 = kwargs.get("u0_squared", 300.0 * np.ones(np.prod(self.square_shape)))
        H0_linear = self.stef_bolc * np.power(u0.flatten() / self.w, 4)
        A = LinearOperator(
            (H0_linear.size, H0_linear.size),
            matvec=self.operator
        )
        b = (self.F + self.G).flatten()
        R = b - self.operator(H0_linear)
        res, exit_code = bicgstab(
            A,
            b,
            rtol=0.0,
            atol=tol,
            x0=R,
        )
        if exit_code:
            print(f"operator failed with exit code: {exit_code}")
            return np.zeros_like(self.F), exit_code
        U = (self.w * np.power(res / self.stef_bolc, 0.25)).reshape(self.square_shape)
        return U, exit_code

    def operator(self, u_linear: np.ndarray, **kwargs) -> np.ndarray:
        """
        Main scheme's operator. The whole scheme solves the equation
        in H-values - heat streams. An after that the whole
        H-array comes to the temperature values (at the end of teh solve-method).

        Args:
            u_linear: 1-d array of the H - heat stream of the cell.
        Returns:
            A(H)
        """
        H = u_linear.reshape(self.square_shape)
        res = np.zeros_like(H)

        # internal cells
        res[1:-1, 1:-1] = (
            -1
            / self.h
            * (
                H[2:, 1:-1]
                + H[:-2, 1:-1]
                + H[1:-1, 2:]
                + H[1:-1, :-2]
                - 4.0 * H[1:-1, 1:-1]
            )
        )

        # edge cells
        res[1:-1, 0] = (
            H[1:-1, 0]
            - H[1:-1, 1]
            - (H[2:, 0] - 2.0 * H[1:-1, 0] + H[:-2, 0])
            + H[1:-1, 0]
        )
        res[-1, 1:-1] = (
            H[-1, 1:-1]
            - H[-2, 1:-1]
            - (H[-1, 2:] - 2.0 * H[-1, 1:-1] + H[-1, :-2])
            + H[-1, 1:-1]
        )
        res[1:-1, -1] = (
            H[1:-1, -1]
            - H[1:-1, -2]
            - (H[2:, -1] - 2.0 * H[1:-1, -1] + H[:-2, -1])
            + H[1:-1, -1]
        )
        res[0, 1:-1] = (
            H[0, 1:-1]
            - H[1, 1:-1]
            - (H[0, 2:] - 2.0 * H[0, 1:-1] + H[0, :-2])
            + H[0, 1:-1]
        )

        # corner cells
        res[0, 0] = H[0, 0] - 0.5 * (H[1, 0] + H[0, 1]) + H[0, 0]
        res[-1, 0] = H[-1, 0] - 0.5 * (H[-2, 0] + H[-1, 1]) + H[-1, 0]
        res[-1, -1] = H[-1, -1] - 0.5 * (H[-2, -1] + H[-1, -2]) + H[-1, -1]
        res[0, -1] = H[0, -1] - 0.5 * (H[1, -1] + H[0, -2]) + H[0, -1]

        return res.flatten()

    def jacobian(self, u_linear: np.ndarray, du_linear: np.ndarray) -> np.ndarray:
        """
        does nothing, used nowhere.
        May be removed soon.
        """
        return du_linear

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
        Static function to obtain F and G arrays.
        Parameters are the same as in __init__()

        Args:
            f_func: the temperature of the inner heat sources.
            g_func: list of 4 functions g(x, y) for the bound temperature:
                [
                    g(x=[a,b], y=a),
                    
                    g(x=b, y=[a, b]),
                    
                    g(x=[a,b], y=b),
                    
                    g(x=a, y=[a,b])
                ]
            square_shape: shape of the scheme.
                Template - [cells, cells, cell_size, cell_size].
            limits: description of the computing area, [a, b] for
                stationary schemes.
        Returns:
            [F, G]
        """
        HeatStream = lambda t: stef_bolc * np.power(t, 4)
        # f = lambda x, y: HeatStream(f_func(x, y))
        g = [
            lambda t: HeatStream(g_func[0](t)),
            lambda t: HeatStream(g_func[1](t)),
            lambda t: HeatStream(g_func[2](t)),
            lambda t: HeatStream(g_func[3](t)),
        ]

        h: np.float64 = (limits[1] - limits[0]) / square_shape[0]
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

    def flatten(self, u_squared: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        does nothing, used nowhere.
        May be removed soon.
        """
        return u_squared
