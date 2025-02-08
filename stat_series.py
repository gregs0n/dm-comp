#!/home/gregs0n/venvs/numpy-venv/bin/python3

from os import environ
environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from itertools import product
import logging

from stat_scheme import DirectStationaryScheme, DMStationaryScheme

from run_stat_test import GetStatFunc
from utils import Material

direct_dots = [
    0,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 10, 10, 10, 10,
    11, 10, 11, 11, 11, 11, 11,  9,  9, 10,
    11, 10,  9, 10, 10, 10, 10,  10, 10, 10,
    11, 10, 10, 11,  9,  9,  9,  9,  9,  9
]

logger = logging.getLogger()

def loadTest():
    start, finish = 10, 50
    cells = list(range(start, finish+1))
    thermal_conds = [1.0, 5.0, 10.0, 20.0]
    limits = (0.0, 1.0)
    stef_bolc = 5.67036713

    f, g = GetStatFunc()
    material = Material("template", 20.0, 0.0, 1.0, 1.0)

    for tcc, cell in product(thermal_conds[-1:], cells):

        result_folder = f".tests/stat_results/tcc{int(tcc)}/"

        material = Material("template", tcc, 0.0, 1.0, 1.0)

        square_shape = [cell, cell]

        F, G = DMStationaryScheme.GetBoundaries(
            f, g,
            square_shape,
            material,
            limits,
            stef_bolc,
            use_sdm=False
        )
        fdm_scheme = DMStationaryScheme(
            F, G,
            square_shape,
            material,
            limits,
            use_sdm=False
        )

        logger.info(
            "Started FDM (%d) %.2f with %d equations",
            cell, tcc, F.size
        )
        fdm_res, exit_code = fdm_scheme.solve(tol=1e-6, inner_tol=5e-4)
        if not exit_code:
            np.save(result_folder + f"fdm/{cell:02d}", fdm_res)

        F, G = DMStationaryScheme.GetBoundaries(
            f, g,
            square_shape,
            material,
            limits,
            stef_bolc,
            use_sdm=True
        )
        sdm_scheme = DMStationaryScheme(
            F, G,
            square_shape,
            material,
            limits,
            use_sdm=True
        )

        logger.info(
            "Started SDM (%d) %.2f with %d equations",
            cell, tcc, F.size
        )
        sdm_res, exit_code = sdm_scheme.solve(tol=1e-6, inner_tol=5e-4)
        if not exit_code:
            np.save(result_folder + f"sdm/{cell:02d}", sdm_res)

        square_shape = [cell, cell, direct_dots[cell], direct_dots[cell]]
        F, G = DirectStationaryScheme.GetBoundaries(
            f, g,
            square_shape,
            material,
            limits,
            stef_bolc
        )
        direct_scheme = DirectStationaryScheme(
            F, G,
            square_shape,
            material,
            limits
        )

        logger.info(
            "Started Direct (%d, %d) %.2f with %d equations",
            cell, direct_dots[cell], tcc, F.size
        )
        direct_res, exit_code = direct_scheme.solve(tol=1e-6, inner_tol=5e-4)
        if not exit_code:
            np.save(result_folder + f"direct/{cell:02d}", direct_res)


def checkTest():
    start, finish = 10, 50
    cells = list(range(start, finish+1))#
    thermal_conds = [1.0, 5.0, 10.0, 20.0]
    limits = (0.0, 1.0)

    errs_L2 = {tcc: [[], []] for tcc in thermal_conds}
    # errs_L_inf = {tcc: [[], []] for tcc in thermal_conds}

    for tcc, cell in product(thermal_conds, cells):

        result_folder = f".tests/stat_results/tcc{int(tcc)}/"

        direct_res = np.load(result_folder + f"direct/{cell:02d}" + ".npy")
        direct_res = DirectStationaryScheme.flatten(direct_res, limits, mod=1)

        fdm_res = np.load(result_folder + f"fdm/{cell:02d}" + ".npy")
        sdm_res = np.load(result_folder + f"sdm/{cell:02d}" + ".npy")

        errs_L2[tcc][0].append(
            norm((direct_res - fdm_res).flatten()) / norm(direct_res.flatten())
        )
        errs_L2[tcc][1].append(
            norm((direct_res - sdm_res).flatten()) / norm(direct_res.flatten())
        )
        # errs_L_inf[tcc][0].append(
        #     norm((direct_res - fdm_res).flatten(), ord=np.inf)
        #     / norm(direct_res.flatten(), ord=np.inf)
        # )
        # errs_L_inf[tcc][1].append(
        #     norm((direct_res - sdm_res).flatten(), ord=np.inf)
        #     / norm(direct_res.flatten(), ord=np.inf)
        # )
        print("Ready cell", cell)

    data_L2 = []
    # data_L_inf = []
    legends = []
    for tcc in thermal_conds:
        data_L2 += list(map(np.array, errs_L2[tcc]))
        # data_L_inf += list(map(np.array, errs_L_inf[tcc]))
        legends += [rf"FDM:$\lambda$={tcc}", rf"SDM:$\lambda$={tcc}"]

    draw1D(
        data_L2,
        [start, finish],
        "fdm & sdm L2-errors",
        legends=legends,
        yscale="log",
        show_plot=1,
    )
    # draw1D(
    #     data_L_inf,
    #     [start, finish],
    #     "fdm & sdm L_inf-errors",
    #     legends=legends,
    #     yscale="log",
    #     show_plot=1,
    # )

def draw1D(
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
            (1.0 / arg),
            single_data,
            label=lab,
            color=colors[i//2],
            #marker="o" if i%2 == 0 else "s",
            linestyle='-' if i%2 == 0 else '--',
            #linewidth=1.25,
        )
    if not ylim:
        ylim = [min([i.min() for i in data]), max([i.max() for i in data])]
    ax.set_yscale(yscale)
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    ax.set_xlim(xmin=1.0 / limits[0], xmax=1.0 / limits[1])
    ax.grid(True)
    ax.legend()
    if show_plot:
        plt.show()
    else:
        fig.savefig(plot_name + ".png")
    plt.close()
    del fig, ax

if __name__ == "__main__":
    logging.basicConfig(
        filename='stat_test.log',
        filemode='w',
        encoding='utf-8',
        level=logging.INFO,
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%d.%m.%Y %H:%M:%S'
    )
    # loadTest()
    checkTest()
