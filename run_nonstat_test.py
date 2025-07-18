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
import matplotlib.pyplot as plt
from scipy.linalg import norm

from nonstat_scheme import DirectNonStationaryScheme, DMNonStationaryScheme, AsymptoticNonStationaryScheme

from utils import Material, NonStatTest, draw1D, drawHeatmap
from nonstat_test_data import nonstat_tests, dt_changes_tests, lambda_changes_tests

logger = logging.getLogger()

non_stat_schemes = [DirectNonStationaryScheme, DMNonStationaryScheme, AsymptoticNonStationaryScheme]

def runtest(test: NonStatTest):
    os.chdir(".tests/nonstat_tests")
    test.init_test_folder()
    os.chdir(test.name)
    params = test.params

    logger.info("Start test `%s`", test.name)
    logger.info("Num layers - %d", int(params.T/params.dt))
    logger.info("crho/dt = %f", params.c_rho / params.dt)

    # run_nonstat_scheme(test, 0, 0, *params)
    # run_nonstat_scheme(test, 1, 0, *params)
    # run_nonstat_scheme(test, 1, 1, *params)
    # run_nonstat_scheme(test, 2, 0, *params)
    # run_nonstat_scheme(test, 2, 1, *params)

    # [err_fdm_L_2, err_sdm_L_2] = check_non_stat(1, test)
    [err_fdm_L_2, err_sdm_L_2] = check_non_stat(2, test)

    logger.info("End test `%s`", test.name)

    os.chdir("../../..")

    return [err_fdm_L_2, err_sdm_L_2]

def run_nonstat_scheme(test: NonStatTest, scheme_no, use_sm, cell, cell_size, tcc, crho, T, dt):
    Scheme = non_stat_schemes[scheme_no]
    f, g = test.f, test.g

    square_shape = (cell, cell, cell_size, cell_size)
    if scheme_no == 1:
        square_shape = (cell, cell)
    elif scheme_no == 2:
        square_shape = (min(9*cell, 200), min(9*cell, 200)) # (250, 250) # 

    material = Material(
        "template",
        thermal_cond=tcc,
        tmin=0.0, tmax=1.0,
        crho=crho
    )
    limits = (0.0, 1.0, T)
    stef_bolc = 5.67036713

    F, G = Scheme.GetBoundaries(f, g, square_shape, material, limits, stef_bolc, dt=dt, use_sm=use_sm)
    scheme = Scheme(np.copy(F), np.copy(G), square_shape, material, dt, limits, use_sm=use_sm)
    res, _ = scheme.solve(1e-6, inner_tol=5e-4) # , u0_squared=600.0*np.ones_like(F[0]))
    data = scheme.flatten(res, limits, mod=0, square_shape=(cell, cell))

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
            logger.debug("Draw layer [%03d]", i*n_plots_step)

    filename = Scheme.__name__

    if scheme_no != 0:
        filename += "_SM" if use_sm else "_FM"
    logger.info("Scheme %s finished", filename)
    if scheme_no == 2:
        np.save(filename+"_full", res)
    np.save(filename, Scheme.flatten(res, limits, mod=1, square_shape=(cell, cell)))

def check_non_stat(scheme_no, test):

    direct = np.load("DirectNonStationaryScheme.npy")
    fm = np.zeros_like(direct)
    sm = np.zeros_like(direct)
    err_fm_folder = "./"
    err_sm_folder = "./"
    if scheme_no == 1:
        fm = np.load("DMNonStationaryScheme_FM.npy")
        err_fm_folder = test.err_fdm_folder
        err_sm_folder = test.err_sdm_folder
        sm = np.load("DMNonStationaryScheme_SM.npy")
    elif scheme_no == 2:
        fm = np.load("AsymptoticNonStationaryScheme_FM.npy")
        sm = np.load("AsymptoticNonStationaryScheme_SM.npy")
        err_fm_folder = test.err_fam_folder
        err_sm_folder = test.err_sam_folder
    if (fm.shape != direct.shape
        or sm.shape != direct.shape):
        exit()

    err_fm_raw = direct - fm
    err_sm_raw = direct - sm

    # err_fm_L_inf = np.zeros(direct.shape[0]-1)
    # err_sm_L_inf = np.zeros(direct.shape[0]-1)
    err_fm_L_2 = np.zeros(direct.shape[0]-1)
    err_sm_L_2 = np.zeros(direct.shape[0]-1)

    n_plots = 10
    n_plots_step = max(1, direct.shape[0] // n_plots)
    indexes = [1] + list(range(n_plots_step, direct.shape[0], n_plots_step))

    for i in range(direct.shape[0]-1):
        # err_fm_L_inf[i] = norm(err_fm_raw[i+1].flatten(), ord=np.inf) / norm(direct[i+1].flatten(), ord=np.inf)
        # err_sm_L_inf[i] = norm(err_sm_raw[i+1].flatten(), ord=np.inf) / norm(direct[i+1].flatten(), ord=np.inf)
        err_fm_L_2[i] = norm(err_fm_raw[i+1].flatten()) / norm(direct[i+1].flatten())
        err_sm_L_2[i] = norm(err_sm_raw[i+1].flatten()) / norm(direct[i+1].flatten())

    os.chdir(err_fm_folder)
    for i in indexes:
        drawHeatmap(
            err_fm_raw[i],
            [0.0, 1.0],
            f"plot_{i:03}",
            show_plot=0,
        )

    os.chdir("..")
    os.chdir(err_sm_folder)
    for i in indexes:
        drawHeatmap(
            err_sm_raw[i],
            [0.0, 1.0],
            f"plot_{i:03}",
            show_plot=0,
        )
    os.chdir("..")
    # draw1D(
    #     [
    #         err_fm_L_inf,
    #         err_sm_L_inf
    #     ],
    #     [0, test.params.T],
    #     f"Относительная (L_inf) ошибка методов по времени ({scheme_no})",
    #     yscale='linear',
    #     legends=["FM", "SM"],
    #     show_plot=0,
    #     filename=non_stat_schemes[scheme_no].__name__
    # )
    draw1D(
        [
            err_fm_L_2,
            err_sm_L_2
        ],
        [0, test.params.T],
        f"Относительная (L_2) ошибка методов по времени ({scheme_no})",
        yscale='linear',
        legends=["FM", "SM"],
        show_plot=0,
        filename=non_stat_schemes[scheme_no].__name__
    )

    return [err_fm_L_2, err_sm_L_2]

def loc_draw1D(
    data: list,
    limits: list,
    plot_name: str,
    yscale="linear",
    show_plot=True,
    ylim=[],
    legends=[],
):

    arg = np.linspace(limits[0], limits[1], data[0].size)
    fig, ax = plt.subplots()
    ax.set_title(plot_name)
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]

    for (i, single_data) in enumerate(data):
        lab = legends[i] if legends else f"plot_{i+1}"
        ax.plot(
            arg,
            single_data,
            label=lab,
            color=colors[i],
            #marker="o" if i%2 == 0 else "s",
            linestyle='-' if 0 and i%2 == 0 else '--',
            #linewidth=1.25,
        )
    if not ylim:
        ylim = [min([i.min() for i in data]), max([i.max() for i in data])]
    ax.set_yscale(yscale)
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    # ax.set_xlim(xmin=1.0 / limits[0], xmax=1.0 / limits[1])
    ax.grid(True)
    ax.legend()
    if show_plot:
        plt.show()
    else:
        fig.savefig(plot_name + ".png")
    plt.close()
    del fig, ax


def dt_changes():
    errs_L2 = dict()

    dts = [0.02, 0.1, 0.5, 1.0]

    for _test in dt_changes_tests:
        errs_L2[_test.params.dt] = runtest(_test)
        logger.info(
            "Data shape %d, %d",
            errs_L2[_test.params.dt][0].size,
            errs_L2[_test.params.dt][1].size
        )

    etl = [
        errs_L2[0.02][0][::50],
        errs_L2[0.02][1][::50]
    ]
    data_L2 = [
        np.abs(errs_L2[0.1][0][::10] - etl[0]),
        np.abs(errs_L2[0.1][1][::10] - etl[1]),
        np.abs(errs_L2[0.5][0][::2] - etl[0]),
        np.abs(errs_L2[0.5][1][::2] - etl[1]),
        np.abs(errs_L2[1.0][0] - etl[0]),
        np.abs(errs_L2[1.0][1] - etl[1]),
    ]
    
    # data_L2 = [
    #     errs_L2[0.02][0][::50],
    #     errs_L2[0.02][1][::50],
    #     errs_L2[0.1][0][::10],
    #     errs_L2[0.1][1][::10],
    #     errs_L2[0.5][0][::2],
    #     errs_L2[0.5][1][::2],
    #     errs_L2[1.0][0],
    #     errs_L2[1.0][1],
    # ]

    plot_legends = []

    for dt in dts[1:]:
        plot_legends += [rf"FDM:$dt$={dt}", rf"SDM:$dt$={dt}"]

    loc_draw1D(
        data_L2[1::2],
        [0, dt_changes_tests[0].params.T],
        "dt L2-errors(3)",
        legends=plot_legends[1::2],
        yscale="linear",
        show_plot=0,
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
    # dt_changes()
    # runtest(nonstat_tests[11])
    # runtest(nonstat_tests[12])
    # for _test in :
        # runtest(_test)

    thermal_conds = [1.0, 5.0, 10.0, 20.0]
    errs_L2 = {tcc: [[], []] for tcc in thermal_conds}
    # master_tests = [nonstat_tests[-2], nonstat_tests[0], nonstat_tests[-1], nonstat_tests[1]]

    for _test in lambda_changes_tests:
        errs_L2[_test.params.thermal_cond] = runtest(_test)

    data_L2 = []
    # data_L_inf = []
    plot_legends = []
    for tcc in thermal_conds:
        data_L2 += list(map(np.array, errs_L2[tcc]))
        plot_legends += [rf"FAM:$\lambda$={tcc}", rf"SAM:$\lambda$={tcc}"]

    loc_draw1D(
        data_L2[1::2],
        [0, nonstat_tests[0].params.T],
        "fam & sam L2-errors (3)",
        legends=plot_legends[1::2],
        yscale="linear",
        show_plot=0,
    )
