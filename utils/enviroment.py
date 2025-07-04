"""
Module template docstring
"""

from typing import Callable
from collections import namedtuple
from dataclasses import dataclass
import os

import numpy as np
from utils.draw import draw1D

Material = namedtuple("Material", ["name", "thermal_cond", "tmin", "tmax", "crho"])

NonStatTestParams = namedtuple(
    "NonStatTestParams",
    [
        "cell",
        "cell_size",
        "thermal_cond",
        "c_rho",
        "T",
        "dt"
    ],
    defaults=[50.0, 1.0]
)

@dataclass
class NonStatTest:
    """
    Later
    """
    name: str
    description: str
    params: NonStatTestParams
    f: Callable[[np.float64, np.float64, np.float64], np.float64]
    g: list[Callable[[np.float64, np.float64, np.float64], np.float64]]
    __test_describe_filename: str = "Описание теста.txt"
    __heat_src_folder: str = "0.Описание теплового источника"
    direct_folder: str = "1.Решение задачи - Прямой метод"
    err_fdm_folder: str = "2.Ошибка - Первый дискретный метод"
    err_sdm_folder: str = "3.Ошибка - Второй дискретный метод"
    err_fam_folder: str = "4.Ошибка - Первый асимптотический метод"
    err_sam_folder: str = "5.Ошибка - Второй асимптотический метод"

    def init_test_folder(self):
        param_description: dict[str, Callable] = {
            "cell" : lambda param: f"{param} x {param} = {param*param} стержней",
            "thermal_cond" : lambda param: f"Коэффициент теплопроводности - {param}",
            "c_rho" : lambda param: f"Коэффициент c_rho - {param}",
            "T" : lambda param: f"Временной промежуток от 0 до {param} сек",
            "dt" : lambda param: f"Шаг по времени - {param} сек",
        }

        if self.name not in os.listdir():
            os.mkdir(self.name)

        os.chdir(self.name)

        if self.__heat_src_folder not in os.listdir():
            os.mkdir(self.__heat_src_folder)
        if self.direct_folder not in os.listdir():
            os.mkdir(self.direct_folder)
        if self.err_fdm_folder not in os.listdir():
            os.mkdir(self.err_fdm_folder)
        if self.err_sdm_folder not in os.listdir():
            os.mkdir(self.err_sdm_folder)
        if self.err_fam_folder not in os.listdir():
            os.mkdir(self.err_fam_folder)
        if self.err_sam_folder not in os.listdir():
            os.mkdir(self.err_sam_folder)

        with open(self.__test_describe_filename, "w", encoding='utf8') as file:
            file.write(self.description)
            for field, value in self.params._asdict().items():
                if field != "cell_size":
                    file.write(param_description[field](value) + '\n')
        os.chdir(self.__heat_src_folder)
        self.g_preview()
        os.chdir("../..")

    def g_preview(self):
        arg = np.linspace(0, 1, 1000)
        titles = [
            "(x,y)=[0,1]x[0]",
            "(x,y)=[1]x[0,1]",
            "(x,y)=[0,1]x[1]",
            "(x,y)=[0]x[0,1]",
        ]
        legends = [
            "g(x, 0)",
            "g(1, y)",
            "g(x, 1)",
            "g(0, y)",
        ]
        for (i, g_side) in enumerate(self.g):
            draw1D(
                [np.array([100.0*g_side(self.params.T, x) for x in arg])],
                [0, 1],
                titles[i],
                legends=[legends[i]],
                show_plot=0,
            )

if __name__ == "__main__":
    t = NonStatTestParams(30, 6, 1.0, 20.0)
    test = NonStatTest(
        "example",
        """
Всем привет, это строкове описание теста
вот его параметры:

""",
        t,
        lambda t: 0.5 + 0.5 * np.sin(np.pi * t),
        [
            lambda t: 0.5 + 0.5 * np.sin(np.pi * t),
            lambda t: 0.5 + 0.5 * np.sin(np.pi * t),
            lambda t: 0.5 + 0.5 * np.sin(np.pi * t),
            lambda t: 0.5 + 0.5 * np.sin(np.pi * t)
        ]
    )
    test.init_test_folder()
