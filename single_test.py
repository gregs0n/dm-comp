#!/home/gregs0n/venvs/numpy-venv/bin/python3
"""
pass
"""

from os import environ
import logging

environ["OMP_NUM_THREADS"] = "4"

import numpy as np

from direct_stationary_scheme import DirectStationaryScheme
from dm_stationary_scheme import DMStationaryScheme

from direct_non_stationary_scheme import DirectNonStationaryScheme
from dm_non_stationary_scheme import DMNonStationaryScheme

from enviroment import Material  # , TestParams, Test
from draw import draw1D, drawHeatmap, drawGif

logger = logging.getLogger("single_test")
logging.basicConfig(
    #filename='direct_test.log',
    # filename='logs.txt',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s\t%(message)s',
    datefmt='%d.%m.%Y %H:%M:%S'
)

stat_schemes = [DirectStationaryScheme, DMStationaryScheme]
non_stat_schemes = [DirectNonStationaryScheme, DMNonStationaryScheme]


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

    t_window = 0.8 * timeBnd

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

    arg = np.arange(0, timeBnd + 0.5*dt, dt)

    # draw1D(
    #     [np.array([activation(x) for x in arg])],
    #     [0, timeBnd],
    #     "g(t)",
    #     show_plot=0,
    #     ylim=[-0.05, 1.05],
    # )

    # draw1D(
    #     [
    #         np.array([activation(x) for x in arg]),
    #         np.array([left_g(x) for x in arg]),
    #         np.array([right_g(x) for x in arg])
    #     ],
    #     [0, timeBnd],
    #     "activation & g_kernel",
    #     show_plot=0,
    #     ylim=[-0.05, 1.05],
    # )

    return f, g


def test_stat(scheme_no, use_sdm, cell, cell_size, tcc, crho):
    """
    template docstring
    """
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
    res, _ = scheme.solve(1e-6, inner_tol=5e-4)
    res = scheme.flatten(res, mod=0)

    drawHeatmap(
        res,
        limits,
        "images/plot",
        show_plot=1,
        zlim=[300, 600],
    )
    return F, G, res


def test_non_stat(scheme_no, use_sdm, cell, cell_size, tcc, crho, T, dt):
    """
    template docstring
    """

    # F_stat, G_stat, stat_res = test_stat()

    Scheme = non_stat_schemes[scheme_no]
    f, g = GetNonStatFunc(T, dt)

    square_shape = (cell, cell, cell_size, cell_size) if scheme_no == 0 else (cell, cell)

    material = Material(
        "template",
        thermal_cond=tcc,
        tmin=0.0, tmax=1.0,
        crho=crho
    )
    limits = (0.0, 1.0, T)
    stef_bolc = 5.67036713

    F, G = Scheme.GetBoundaries(f, g, square_shape, material, limits, stef_bolc, dt=dt, use_sdm=use_sdm)
    scheme = Scheme(np.copy(F), np.copy(G), square_shape, material, dt, limits, use_sdm=use_sdm)
    res = scheme.solve(1e-6, inner_tol=5e-4) # , u0_squared=600.0*np.ones_like(F[0]))
    res = scheme.flatten(res, mod=1)

    if scheme_no == 0:
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
            logger.info("Draw layer [%03d]", i*n_plots_step)

    # drawGif(res)
    filename = "DMNonStationaryScheme"
    if scheme_no == 0:
        filename = "DirectNonStationaryScheme"
    elif use_sdm:
        filename += "_SDM"
    else:
        filename += "_FDM"
    print(filename)
    np.save(filename, res)


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
        show_plot=0,
        ylim=[-0.05, 1.05]
    )

def norm_L2(x: np.ndarray, h: np.float64) -> np.float64:
    if x.ndim != 2:
        return -1
    else:
        h2 = h*h
        x *= x
        res = h2 * np.sum(x)
        return res

def main():
    cell = 30
    cell_size = 6
    tcc = 1.0
    crho = 2000.0
    T = 50.0
    dt = 1.0
    logger.info("Num layers - %d", int(T/dt))
    logger.info("crho/dt = %f", crho / dt)
    test_non_stat(0, 0, cell, cell_size, tcc, crho, T, dt)
    test_non_stat(1, 0, cell, cell_size, tcc, crho, T, dt)
    test_non_stat(1, 1, cell, cell_size, tcc, crho, T, dt)

def check_non_stat():
    suffix = ""#"_lin"
    folder = f"nonstat{suffix}/"
    direct = np.load(folder + "DirectNonStationaryScheme.npy")
    print(direct.shape)
    fdm = np.load(folder + "DMNonStationaryScheme_FDM.npy")
    print(fdm.shape)
    sdm = np.load(folder + "DMNonStationaryScheme_SDM.npy")
    print(sdm.shape)
    if (fdm.shape != direct.shape
        or sdm.shape != direct.shape):
        exit()

    err_fdm_raw = np.abs(direct - fdm)
    err_sdm_raw = np.abs(direct - sdm)

    err_fdm = np.zeros(direct.shape[0]-1)
    err_sdm = np.zeros(direct.shape[0]-1)

    for i in range(direct.shape[0]-1):
        err_fdm[i] = err_fdm_raw[i+1].max() / direct[i+1].max()
        err_sdm[i] = err_sdm_raw[i+1].max() / direct[i+1].max()
        # err_fdm[i] = norm_L2(err_fdm_raw[i+1], 1.0/30.0)
        # err_sdm[i] = norm_L2(err_sdm_raw[i+1], 1.0/30.0)

        drawHeatmap(
            err_fdm_raw[i],
            [0.0, 1.0],
            f"images/non_stat_err/fdm_plot_{i+1:03}",
            show_plot=0,
        )
        drawHeatmap(
            err_sdm_raw[i],
            [0.0, 1.0],
            f"images/non_stat_err/sdm_plot_{i+1:03}",
            show_plot=0,
        )
        logger.info("Draw layer [%03d]", i+1)

    draw1D(
        [
            err_fdm,
            err_sdm
        ],
        [1, direct.shape[0]],
        f"nonstat_err{suffix}",
        yscale='linear',
        legends=["FDM", "SDM"],
        show_plot=0
    )

if __name__ == "__main__":
    TestBndFuncs()
    # main()
    # check_non_stat()
