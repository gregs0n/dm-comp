import pickle
from hashlib import sha256
import numpy as np
from collections import namedtuple
from dataclasses import dataclass

Material = namedtuple(
    "Material",
    [
        "name",
        "thermal_cond",
        "tmin",
        "tmax",
        "crho"
    ]
)

TestParams = namedtuple(
    "TestParams",
    [
        "square_shape",
        "thermal_cond",
        "limits",
    ]
)

@dataclass
class Test:
    params: TestParams
    data: dict[str, np.ndarray]

    def __str__(self):
        return f"square_shape{self.params.square_shape}_tcc({self.params.thermal_cond:04.1f})"

    def getHash(self):
        return f"{abs(hash(self.params)):0>16x}"

    def save(self, path: str = ""):
        file_path = path + self.getHash() + ".bin"
        print(file_path)
        with open(file_path, "wb") as file:
            pickle.dump(self.data, file)