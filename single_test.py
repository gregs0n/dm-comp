#!/home/gregs0n/venvs/numpy-venv/bin/python3
"""
pass
"""

from os import environ
import logging

environ["OMP_NUM_THREADS"] = "4"

import numpy as np

from direct_stationary_scheme import DirectStationaryScheme
from fdm_stationary_scheme import FDMStationaryScheme
from sdm_stationary_scheme import SDMStationaryScheme

from base_non_stationary_scheme import BaseNonStationaryScheme
from direct_non_stationary_scheme import DirectNonStationaryScheme

from enviroment import Material  # , TestParams, Test
from draw import draw1D, drawHeatmap, drawGif

logger = logging.getLogger("single_test")
logging.basicConfig(
    #filename='direct_test.log',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s\t%(message)s',
    datefmt='%d.%m.%Y %H:%M:%S'
)

stat_schemes = [DirectStationaryScheme, FDMStationaryScheme, SDMStationaryScheme]
non_stat_schemes = [DirectNonStationaryScheme]


def GetStatFunc():
    """
    returns f(x, y) and list of 4 g(t) NORMED
    """
    w = 100.0
    tmax = 600.0 / w
    tmin = 300.0 / w
    coef = tmax - tmin
    d = tmin

    L = 1.0
    a, b = 0.0, 0.5
    g_kernel = lambda t: 0.5 + 0.5 * np.sin(np.pi * t)

    left_g = lambda t: (
        g_kernel((2 * t - 0.5 * (b - a)) / (b - a)) if a <= t <= b else 0.0
    )

    right_g = lambda t: (
        g_kernel((2 * (L - t) - 0.5 * (b - a)) / (b - a))
        if (1.0 - b) <= t <= (1.0 - a)
        else 0.0
    )

    f = lambda x, y: 0.0
    g = [
        lambda t: (d + coef * left_g(t)),
        # lambda t: tmax,
        lambda t: tmin,
        lambda t: (d + coef * right_g(t)),
        lambda t: tmin,
        # lambda t: tmin,
    ]
    return f, g


def GetNonStatFunc(timeBnd: np.float64, dt: np.float64):
    """
    pass
    """
    w = 100.0
    tmax = 600.0 / w
    tmin = 300.0 / w
    coef = tmax - tmin
    d = tmin

    L = 1.0
    a, b = 0.0, 0.5

    t_window = 0.75 * timeBnd

    activation_kernel = lambda t: 0.5 + 0.5 * np.sin(np.pi * t)
    # activation_kernel = lambda t: t

    activation = lambda t: (
        activation_kernel((t - 0.5 * t_window) / t_window) if t < t_window else 1.0
    )
    # activation = lambda t: activation_kernel(t / t_window) if t < t_window else 1.0

    g_kernel = lambda t: 0.5 + 0.5 * np.sin(np.pi * t)

    left_g = lambda t: (
        g_kernel((2 * t - 0.5 * (b - a)) / (b - a)) if a <= t <= b else 0.0
    )

    right_g = lambda t: (
        g_kernel((2 * (L - t) - 0.5 * (b - a)) / (b - a))
        if (1.0 - b) <= t <= (1.0 - a)
        else 0.0
    )

    f = lambda T, x, y: 0.0
    g = [
        lambda T, t: (d + coef * activation(T) * left_g(t)),
        # lambda T, t: tmax,
        lambda T, t: tmin,
        lambda T, t: (d + coef * activation(T) * right_g(t)),
        lambda T, t: tmin,
        # lambda T, t: tmin,
    ]

    # arg = np.arange(0, timeBnd + 0.5*dt, dt)

    # draw1D(
    #     [np.array([activation(x) for x in arg])],
    #     [0, timeBnd],
    #     "activation & g_kernel"
    # )

    return f, g


def main_stat():
    """
    template docstring
    """
    k = 0
    Scheme = stat_schemes[k]

    f, g = GetStatFunc()

    cell = 10
    cell_size = 11

    square_shape = (cell, cell, cell_size, cell_size) if k == 0 else (cell, cell)

    material = Material(
        "template",
        thermal_cond=20.0,
        tmin=0.0, tmax=1.0,
        crho=1.0
    )
    limits = (0.0, 1.0)
    stef_bolc = 5.67036713

    # param = TestParams(square_shape, material.thermal_cond, limits)
    # test = Test(param, {"FDM" : np.zeros(square_shape)})
    # test.save()
    # return

    F, G = Scheme.GetBoundaries(f, g, square_shape, material, limits, stef_bolc)
    scheme = Scheme(F, G, square_shape, material, limits)
    scheme.normed = 1
    res, _ = scheme.solve(1e-6, inner_tol=5e-4)
    res = scheme.flatten(res, mod=0)

    # drawHeatmap(res, limits, "images/plot", show_plot=1, zlim=[300, 600])
    return F, G, res


def main_non_stat():
    """
    template docstring
    """

    # F_stat, G_stat, stat_res = main_stat()

    k = 0
    Scheme = non_stat_schemes[k]

    T = 1.0
    dt = 0.01
    f, g = GetNonStatFunc(T, dt)

    cell = 10
    cell_size = 6

    square_shape = (cell, cell, cell_size, cell_size) if k == 0 else (cell, cell)

    material = Material("template", 1.0, 0.0, 1.0, 1.0)
    limits = (0.0, 1.0, T)
    stef_bolc = 5.67036713

    # param = TestParams(square_shape, material.thermal_cond, limits)
    # test = Test(param, {"FDM" : np.zeros(square_shape)})
    # test.save()
    # return

    F, G = Scheme.GetBoundaries(f, g, square_shape, material, limits, stef_bolc, dt=dt)
    scheme = Scheme(F, G, square_shape, material, dt, limits)
    # scheme.normed = 1
    logger.info("")
    res = scheme.solve(1e-6, inner_tol=5e-4)  # , u0_squared=stat_res)
    res = scheme.flatten(res, mod=1)

    n_plots = 10
    n_plots_step = max(1, res.shape[0] // n_plots)
    for i, layer in enumerate(res[::n_plots_step]):
        drawHeatmap(
            layer,
            scheme.limits[:-1],
            f"images/direct_non_stat/plot_{i*n_plots_step:03}",
            show_plot=0,
            zlim=[300, 600],
        )

    # drawGif(res)


def TestBndFuncs(a=0.0, b=0.3, L=1.0):
    arg = np.linspace(0, L, 100)

    activation = lambda t: 0.5 + 0.5 * np.sin(np.pi * t)

    g_kernel = lambda t: 0.5 + 0.5 * np.sin(np.pi * t)

    left_g = lambda t: (
        g_kernel((2 * t - 0.5 * (b - a)) / (b - a)) if a <= t <= b else 0.0
    )

    right_g = lambda t: (
        g_kernel((2 * (L - t) - 0.5 * (b - a)) / (b - a)) if 0.7 <= t <= 1.0 else 0.0
    )
    draw1D(
        [
            activation((arg - 0.5 * L) / L),
            g_kernel((2 * arg - 0.5 * L) / L),
            np.array([left_g(x) for x in arg]),
            np.array([right_g(x) for x in arg]),
        ],
        [0, L],
        "activation & g_kernel",
    )


if __name__ == "__main__":
    # main_stat()
    main_non_stat()
    # TestBndFuncs()
