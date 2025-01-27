#!/home/gregs0n/venvs/numpy-venv/bin/python3

import numpy as np
from itertools import product

from FDMStationaryScheme import FDMStationaryScheme
from SDMStationaryScheme import SDMStationaryScheme
from DirectStationaryScheme import DirectStationaryScheme

from enviroment import Material, TestParams, Test
from draw import drawHeatmap

schemes = [DirectStationaryScheme, FDMStationaryScheme, SDMStationaryScheme]


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


def GenerateStationaryTests():
    path = "./tests/stat/"

    cells = list(range(10, 50))
    thermal_conds = [1.0, 5.0, 10.0, 20.0]
    limits = (0.0, 1.0)
    stef_bolc = 5.67036713

    f, g = GetFunc()

    material = Material("template", 20.0, 0.0, 1.0, 1.0)

    for cell, tcc in product(cells, thermal_conds):
        param = TestParams((cell, cell), tcc, limits)
        test = Test(param, {"FDM": np.zeros((cell, cell))})
        material._replace(thermal_cond=tcc)
        # print(test, end="\t")
        for SchemeType in schemes:
            square_shape = (
                (cell, cell, 11, 11)
                if SchemeType == DirectStationaryScheme
                else (cell, cell)
            )
            F, G = SchemeType.GetBoundaries(
                f, g, square_shape, material, limits, stef_bolc
            )
            test.data[SchemeType.__name__] = F, G
        test.save(path)


if __name__ == "__main__":
    GenerateStationaryTests()
