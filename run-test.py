#!/home/gregs0n/venvs/numpy-venv/bin/python3

from os import environ
environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from itertools import product
import logging

from direct_stationary_scheme import DirectStationaryScheme

from single_test import GetStatFunc
from enviroment import Material, TestParams, Test
from draw import drawHeatmap

direct_dots = [
    0,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 10, 10, 10, 10,
    11, 10, 11, 11, 11, 11, 11,  9,  9, 10,
    11, 10,  9, 10, 10,  4, 10,  8,  5, 40,
    11,  6,  6,  7, 45, 46, 47, 48,  9, 50
]

logger = logging.getLogger()
logging.basicConfig(
    filename='direct_test.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s\t%(levelname)s::%(message)s',
    datefmt='%d.%m.%Y %H:%M:%S'
)


def loadTest():
    test_folder = "tests/stat/"
    result_folder = "direct_results/"
    image_folder = "images/"

    cells = list(range(10, 51))# [40, 45, 46, 47, 48, 50]
    thermal_conds = [1.0, 5.0, 10.0, 20.0]
    limits = (0.0, 1.0)
    stef_bolc = 5.67036713

    f, g = GetStatFunc()

    material = Material("template", 20.0, 0.0, 1.0, 1.0)

    cell_size_start = 11

    for cell, tcc in product(cells, thermal_conds[-1:]):

        param = TestParams((cell, cell), tcc, limits)
        test = Test(param, {"FDM": np.zeros((cell, cell))})
        material = Material("template", tcc, 0.0, 1.0, 1.0)
        logging.info("Test %s %s", str(test), test.get_hash())

        res = np.load(result_folder + test.get_hash() + ".npy")
        print(res.shape)

        # cell_size = cell_size_start
        # exit_code = 1
        # while cell_size > 2 and exit_code != 0:
        #     logging.info("Start computations with cell size %d", cell_size)

        #     square_shape = (cell, cell, cell_size, cell_size)
        #     test.data["DirectStationaryScheme"] = DirectStationaryScheme.GetBoundaries(
        #         f, g, square_shape, material, limits, stef_bolc
        #     )
        #     scheme = DirectStationaryScheme(
        #         *test.data["DirectStationaryScheme"],
        #         square_shape,
        #         material,
        #         limits
        #     )
        #     res, exit_code = scheme.solve(1e-6, inner_tol=1e-4)
        #     if exit_code:
        #         cell_size -= 1

        # direct_dots[cell] = cell_size
        # data_0 = scheme.flatten(res, mod=0)
        # data_1 = scheme.flatten(res, mod=1)
        # drawHeatmap(data_0, limits, image_folder + str(test) + "_mod=0", show_plot=0, zlim=[300, 600])
        # drawHeatmap(data_1, limits, image_folder + str(test) + "_mod=1", show_plot=0, zlim=[300, 600])

        # np.save(result_folder + test.get_hash(), res)
        # test.save(test_folder)

        # logging.info("SUCCESS - (%d, %d) - %d", cell, cell, cell_size)


if __name__ == "__main__":
    loadTest()
