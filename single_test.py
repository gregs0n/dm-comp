#!/home/gregs0n/venvs/numpy-venv/bin/python3
"""
pass
"""

from os import environ
import numpy as np

from direct_stationary_scheme import DirectStationaryScheme
from fdm_stationary_scheme import FDMStationaryScheme
from sdm_stationary_scheme import SDMStationaryScheme

from base_non_stationary_scheme import BaseNonStationaryScheme
from direct_non_stationary_scheme import DirectNonStationaryScheme

from enviroment import Material  # , TestParams, Test
from draw import draw1D, drawHeatmap, drawGif

environ["OMP_NUM_THREADS"] = "4"

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
    f = lambda x, y: 0.0
    g = [
        lambda t: ((d + coef * np.sin(np.pi * t / 0.3)) if 0.0 <= t <= 0.3 else tmin),
        # lambda t: tmax,
        lambda t: tmin,
        lambda t: (
            (d + coef * np.sin(np.pi * (1.0 - t) / 0.3)) if 0.7 <= t <= 1.0 else tmin
        ),
        lambda t: tmin,
        # lambda t: tmin,
    ]
    return f, g


def GetNonStatFunc(T: np.float64):
    """
    pass
    """

    arg = np.linspace(0, T, 100)
    activation = lambda t: 0.5 + 0.5 * np.sin(np.pi * (t - 0.5 * T) / T)
    data = activation(arg)
    draw1D([data], [0, T], "activation", ylim=[-0.05, 1.05])
    f_stat, g_stat = GetStatFunc()
    f = lambda T, x, y: activation(T) * f_stat(x, y)

    g = [
        lambda T, t: activation(T) * g_stat[0](t),
        lambda T, t: activation(T) * g_stat[1](t),
        lambda T, t: activation(T) * g_stat[2](t),
        lambda T, t: activation(T) * g_stat[3](t),
    ]
    return f, g


def main_stat():
    """
    template docstring
    """
    k = 2
    Scheme = stat_schemes[k]

    f, g = GetStatFunc()

    cell = 50
    cell_size = 11

    square_shape = (cell, cell, cell_size, cell_size) if k == 0 else (cell, cell)

    material = Material("template", 1.0, 0.0, 1.0, 1.0)
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

    drawHeatmap(res, limits, "plot", show_plot=1)  # , zlim=[300, 600])


def main_non_stat():
    """
    template docstring
    """
    k = 0
    Scheme = non_stat_schemes[k]

    T = 5.0
    dt = 0.1
    f, g = GetNonStatFunc(T)

    cell = 5
    cell_size = 6

    square_shape = (cell, cell, cell_size, cell_size) if k == 0 else (cell, cell)

    material = Material("template", 1.0, 0.0, 1.0, 10.0)
    limits = (0.0, 1.0, T)
    stef_bolc = 5.67036713

    # param = TestParams(square_shape, material.thermal_cond, limits)
    # test = Test(param, {"FDM" : np.zeros(square_shape)})
    # test.save()
    # return

    F, G = Scheme.GetBoundaries(f, g, square_shape, material, limits, stef_bolc, dt=dt)
    scheme = Scheme(F, G, square_shape, material, dt, limits)
    # scheme.normed = 1
    print(G.shape)
    data = BaseNonStationaryScheme.flatten(scheme, G, mod=1)
    print(data.shape)
    drawGif(data)
    exit()
    res, _ = scheme.solve(1e-6, inner_tol=5e-4)
    res = scheme.flatten(res, mod=0)

    drawGif(res)


if __name__ == "__main__":
    # main_stat()
    main_non_stat()
