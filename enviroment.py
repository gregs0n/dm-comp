"""
Module template docstring
"""

import pickle
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

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
