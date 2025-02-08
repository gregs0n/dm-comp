#!/home/gregs0n/venvs/numpy-venv/bin/python3
"""
Runs test from `nonstat_test_data.py`
Single test launches all three schemes and compares them between each other.
Then saves everything to this test's folder.
"""

import os
import logging

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from scipy.linalg import norm

from nonstat_scheme import DirectNonStationaryScheme, DMNonStationaryScheme

from utils import Material, NonStatTest, draw1D, drawHeatmap
from nonstat_test_data import nonstat_tests

logger = logging.getLogger()

non_stat_schemes = [DirectNonStationaryScheme, DMNonStationaryScheme]

def runtest(test: NonStatTest):
    os.chdir(".tests/nonstat_tests")
    test.init_test_folder()
    os.chdir(test.name)
    params = test.params

    logger.info("Start test `%s`", test.name)
    logger.info("Num layers - %d", int(params.T/params.dt))
    logger.info("crho/dt = %f", params.c_rho / params.dt)

    run_nonstat_scheme(test, 0, 0, *params)
    run_nonstat_scheme(test, 1, 0, *params)
    run_nonstat_scheme(test, 1, 1, *params)

    check_non_stat(test)

    logger.info("End test `%s`", test.name)

    os.chdir("../../..")

def run_nonstat_scheme(test: NonStatTest, scheme_no, use_sdm, cell, cell_size, tcc, crho, T, dt):
    Scheme = non_stat_schemes[scheme_no]
    f, g = test.f, test.g

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
    res, _ = scheme.solve(1e-6, inner_tol=5e-4) # , u0_squared=600.0*np.ones_like(F[0]))
    data = scheme.flatten(res, mod=0)

    if scheme_no == 0:
        n_plots = 10
        n_plots_step = max(1, data.shape[0] // n_plots)
        for i, layer in enumerate(data[::n_plots_step]):
            drawHeatmap(
                layer,
                scheme.limits[:-1],
                f"{test.direct_folder}/plot_{i*n_plots_step:03}",
                show_plot=0,
                zlim=[data.min(), data.max()],
            )
            logger.info("Draw layer [%03d]", i*n_plots_step)

    filename = "DMNonStationaryScheme"
    if scheme_no == 0:
        filename = "DirectNonStationaryScheme"
    elif use_sdm:
        filename += "_SDM"
    else:
        filename += "_FDM"
    logger.info("Scheme %s finished", filename)
    np.save(filename, scheme.flatten(res, mod=1))

def check_non_stat(test):

    direct = np.load("DirectNonStationaryScheme.npy")
    fdm = np.load("DMNonStationaryScheme_FDM.npy")
    sdm = np.load("DMNonStationaryScheme_SDM.npy")
    if (fdm.shape != direct.shape
        or sdm.shape != direct.shape):
        exit()

    err_fdm_raw = direct - fdm
    err_sdm_raw = direct - sdm

    err_fdm_L_inf = np.zeros(direct.shape[0]-1)
    err_sdm_L_inf = np.zeros(direct.shape[0]-1)
    err_fdm_L_2 = np.zeros(direct.shape[0]-1)
    err_sdm_L_2 = np.zeros(direct.shape[0]-1)

    n_plots = 10
    n_plots_step = max(1, direct.shape[0] // n_plots)
    indexes = [1] + list(range(n_plots_step, direct.shape[0], n_plots_step))

    for i in range(direct.shape[0]-1):
        err_fdm_L_inf[i] = norm(err_fdm_raw[i+1].flatten(), ord=np.inf) / norm(direct[i+1].flatten(), ord=np.inf)
        err_sdm_L_inf[i] = norm(err_sdm_raw[i+1].flatten(), ord=np.inf) / norm(direct[i+1].flatten(), ord=np.inf)
        err_fdm_L_2[i] = norm(err_fdm_raw[i+1].flatten()) / norm(direct[i+1].flatten())
        err_sdm_L_2[i] = norm(err_sdm_raw[i+1].flatten()) / norm(direct[i+1].flatten())

    os.chdir(test.err_fdm_folder)
    for i in indexes:
        drawHeatmap(
            err_fdm_raw[i],
            [0.0, 1.0],
            f"plot_{i:03}",
            show_plot=0,
        )

    os.chdir("..")
    os.chdir(test.err_sdm_folder)
    for i in indexes:
        drawHeatmap(
            err_sdm_raw[i],
            [0.0, 1.0],
            f"plot_{i:03}",
            show_plot=0,
        )
    os.chdir("..")
    draw1D(
        [
            err_fdm_L_inf,
            err_sdm_L_inf
        ],
        [0, test.params.T],
        "Относительная (L_inf) ошибка методов по времени",
        yscale='linear',
        legends=["FDM", "SDM"],
        show_plot=0
    )
    draw1D(
        [
            err_fdm_L_2,
            err_sdm_L_2
        ],
        [0, test.params.T],
        "Относительная (L_2) ошибка методов по времени",
        yscale='linear',
        legends=["FDM", "SDM"],
        show_plot=0
    )

if __name__ == "__main__":
    logging.basicConfig(
        filename='nonstat_test.log',
        filemode='w',
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%d.%m.%Y %H:%M:%S'
    )
    # runtest(nonstat_tests[0])
    for _test in nonstat_tests:
        runtest(_test)
