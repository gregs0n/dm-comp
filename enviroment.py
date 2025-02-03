"""
Module template docstring
"""

import pickle
from typing import Callable
from collections import namedtuple
from dataclasses import dataclass
import os

import numpy as np
from draw import draw1D

Material = namedtuple("Material", ["name", "thermal_cond", "tmin", "tmax", "crho"])

TestParams = namedtuple(
    "TestParams",
    [
        "square_shape",
        "thermal_cond",
        "limits",
    ],
)


@dataclass
class Test:
    """
    Class template docstring
    """

    params: TestParams
    data: dict[str, np.ndarray]

    def __repr__(self):
        return f"tcc({self.params.thermal_cond:04.1f})_shape{self.params.square_shape}"

    def get_hash(self):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        return f"{abs(hash(self.params)):0>16x}"

    def save(self, folder: str = ""):
        """
        Template docstring (EDIT)

        Args:
            arg1: arg1 decsription
        Returns:
            what function returns
        """
        file_path = folder + self.get_hash() + ".bin"
        # print(file_path)
        with open(file_path, "wb") as file:
            pickle.dump(self.data, file)

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

    def init_test_folder(self):
        if self.name not in os.listdir():
            os.mkdir(self.name)
            os.chdir(self.name)
            os.mkdir(self.__heat_src_folder)
            os.mkdir(self.direct_folder)
            os.mkdir(self.err_fdm_folder)
            os.mkdir(self.err_sdm_folder)
        else:
            os.chdir(self.name)
        with open(self.__test_describe_filename, "w", encoding='utf8') as file:
            file.write(self.description)
            for field, value in self.params._asdict().items():
                if field != "cell_size":
                    file.write(f"{field} = {value}\n")
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
                [np.array([g_side(50.0, x) for x in arg])],
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
