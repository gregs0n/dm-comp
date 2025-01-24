#!/home/gregs0n/venvs/numpy-venv/bin/python3
"""
pass
"""

from os import environ
import numpy as np

from fdm_stationary_scheme import FDMStationaryScheme
from sdm_stationary_scheme import SDMStationaryScheme
from direct_stationary_scheme import DirectStationaryScheme

from enviroment import Material#, TestParams, Test
from draw import drawHeatmap

environ['OMP_NUM_THREADS'] = '4'

schemes = [
    DirectStationaryScheme,
    FDMStationaryScheme,
    SDMStationaryScheme
]

def GetFunc():
    ## returns f(x, y) and list of 4 g(t) NORMED
    w = 100.0
    tmax = 600.0 / w
    tmin = 300.0 / w
    coef = tmax - tmin
    d = tmin
    f = lambda x, y: 0.0
    g = [
        lambda t: (
            (d + coef * np.sin(np.pi * t / 0.3))
            if 0.0 <= t <= 0.3
            else tmin
        ),
        # lambda t: tmax,
        lambda t: tmin,
        lambda t: (
            (d + coef * np.sin(np.pi * (1.0 - t) / 0.3))
            if 0.7 <= t <= 1.0
            else tmin
        ),
        lambda t: tmin,
        # lambda t: tmin,
    ]
    return f, g

def main():
    k = 2
    Scheme = schemes[k]

    f, g = GetFunc()

    cell = 10
    cell_size = 6

    square_shape = (cell, cell, cell_size, cell_size) if k == 0 else (cell, cell)

    material = Material("template", 20.0, 0.0, 1.0, 1.0)
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
    
    drawHeatmap(res, limits, "plot", show_plot=1)#, zlim=[300, 600])

if __name__ == "__main__":
    main()
