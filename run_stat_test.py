#!/home/gregs0n/venvs/numpy-venv/bin/python3
"""
Module for single launch of a stationary scheme.
Saves nothing, only shows.
"""

from os import environ
import logging

environ["OMP_NUM_THREADS"] = "4"

import numpy as np

from stat_scheme import DirectStationaryScheme, DMStationaryScheme, AsymptoticStationaryScheme
from utils import Material, drawHeatmap

stat_schemes = [DirectStationaryScheme, DMStationaryScheme, AsymptoticStationaryScheme]

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

def test_stat(scheme_no, use_sdm, cell, cell_size, tcc, crho):
    """
    template docstring
    """
    logger = logging.getLogger()

    Scheme = stat_schemes[scheme_no]

    f, g = GetStatFunc()

    square_shape = (cell, cell, cell_size, cell_size) if scheme_no == 0 else (cell, cell)

    material = Material(
        "template",
        thermal_cond=tcc,
        tmin=0.0, tmax=1.0,
        crho=crho
    )
    limits = (0.0, 1.0)
    stef_bolc = 5.67036713

    F, G = Scheme.GetBoundaries(f, g, square_shape, material, limits, stef_bolc, use_sdm=use_sdm)
    scheme = Scheme(np.copy(F), np.copy(G), square_shape, material, limits, use_sdm=use_sdm)
    logger.info("Started test (%d, %d) %.2f with %d equations", cell, cell_size, tcc, F.size)
    res, _ = scheme.solve(1e-6, inner_tol=5e-4)
    # data = Scheme.flatten(res, square_shape, limits, mod=0)
    data = Scheme.flatten(res, limits, mod=1)
    if True or scheme_no == 0:
        drawHeatmap(
            data,
            limits,
            "direct_flatten",
            # "asymptotic",
            show_plot=0,
            # zlim=[300, 600],
        )
    logger.info("Test over")
    return F, G, res

if __name__ == "__main__":
    logging.basicConfig(
        filename='stat_test.log',
        filemode='w',
        encoding='utf-8',
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s\t%(message)s',
        datefmt='%d.%m.%Y %H:%M:%S'
    )
    test_stat(
        scheme_no=0,
        use_sdm=False,
        cell=5,
        cell_size=51,
        tcc=1.0,
        crho=20.0
    )
