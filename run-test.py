#!/home/gregs0n/venvs/numpy-venv/bin/python3

import numpy as np
from itertools import product
import pickle

from DirectStationaryScheme import DirectStationaryScheme

from enviroment import Material, TestParams, Test
from draw import drawHeatmap


def GetFunc():
    ## returns f(x, y) and list of 4 g(t) NORMED
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


def loadTest():
    path = "./tests/stat/"

    cells = list(range(10, 51))
    thermal_conds = [1.0, 5.0, 10.0, 20.0]
    limits = (0.0, 1.0)
    stef_bolc = 5.67036713

    f, g = GetFunc()

    material = Material("template", 20.0, 0.0, 1.0, 1.0)

    cell_size_start = 10

    for cell, tcc in product(cells, thermal_conds[-1:]):
        param = TestParams((cell, cell), tcc, limits)
        test = Test(param, {"FDM": np.zeros((cell, cell))})
        material = Material("template", tcc, 0.0, 1.0, 1.0)
        print(cell, tcc, test, test.getHash(), end=" ")

        file = open(path + test.getHash() + ".bin", "rb")
        bnd_data = pickle.load(file)
        file.close()

        print(bnd_data["DirectStationaryScheme"][0].shape)

        # cell_size = cell_size_start
        # square_shape = (cell, cell, cell_size, cell_size)
        # exit_code = 1
        # while cell_size > 2 and exit_code == 1:
        #     cell_size -= 1
        #     square_shape = (cell, cell, cell_size, cell_size)
        #     if 1 or cell_size < cell_size_start:
        #         print(f"RESTART WITH REDUCED CELL SIZE - {cell_size}")
        #         bnd_data["DirectStationaryScheme"] = DirectStationaryScheme.GetBoundaries(
        #             f, g, square_shape, material, limits, stef_bolc
        #         )
        #     scheme = DirectStationaryScheme(
        #         *bnd_data["DirectStationaryScheme"],
        #         square_shape,
        #         material,
        #         limits
        #     )
        #     res, exit_code = scheme.solve(1e-6, inner_tol=1e-6)
        # res = scheme.flatten(res, mod=1)
        # np.save("direct_results/" + test.getHash(), res)
        # drawHeatmap(res, limits, "images/" + test.__str__(), show_plot=0)
        # cell_size_start = cell_size + 1
        # if cell_size < 11:
        #     test.data = bnd_data
        #     test.save(path)


if __name__ == "__main__":
    loadTest()
