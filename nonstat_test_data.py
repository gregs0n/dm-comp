"""
template
"""

import numpy as np
from utils import NonStatTestParams, NonStatTest

ACTIVATION_L = 0.6
TMIN = 3.0
TMAX = 6.0

def get_activation(T):
    t_window = ACTIVATION_L * T
    return lambda t: 0.5 + 0.5 * np.sin(np.pi * (t - 0.5*t_window) / t_window) if t < t_window else 1.0

g_kernel = lambda t: 0.5 + 0.5 * np.sin(np.pi * t)
def get_g(a, b):
    L = b - a
    return lambda t: g_kernel((2 * (t - a) - 0.5 * L) / L) if a <= t <= b else 0.0

nonstat_tests = [
    NonStatTest(
        name="test_00_two_side_sin",
        description="Два синуса на противоположных сторонах, в отдалении от углов\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=5.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.0, 0.5)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.0)(t),
            lambda T, t: TMIN,
        ]
    ),
        NonStatTest(
        name="test_01_two_side_sin",
        description="Два синуса на противоположных сторонах, в отдалении от углов.\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=20.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.0, 0.5)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.0)(t),
            lambda T, t: TMIN,
        ]
    ),
    NonStatTest(
        name="test_02_four_corner_sin",
        description="Четыре синуса, расположены в углах таким образом, чтобы максимум располагался точно в углу.\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=5.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.7, 1.3)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-0.3, 0.3)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-0.3, 0.3)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.7, 1.3)(t),
        ]
    ),
    NonStatTest(
        name="test_03_four_corner_sin",
        description="Четыре синуса, расположены в углах таким образом, чтобы максимум располагался точно в углу.\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=20.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.7, 1.3)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-0.3, 0.3)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-0.3, 0.3)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.7, 1.3)(t),
        ]
    ),
        NonStatTest(
        name="test_04_point_heat_source",
        description="Имитация источника, близкого к точечному.\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=1.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5 - 0.05, 0.5 + 0.05)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN,
            lambda T, t: TMIN,
        ]
    ),
            NonStatTest(
        name="test_05_point_heat_source",
        description="Имитация источника, близкого к точечному.\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=5.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5 - 0.05, 0.5 + 0.05)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN,
            lambda T, t: TMIN,
        ]
    ),
    NonStatTest(
        name="test_06_two_side_sin_dt=0.1",
        description="Два синуса на противоположных сторонах, в отдалении от углов\nУменьшенный шаг по времени (0.1)\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=5.0,
            c_rho=20.0,
            dt=0.1
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.0, 0.5)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.0)(t),
            lambda T, t: TMIN,
        ]
    ),
    NonStatTest(
        name="test_07_two_side_sin_dt=0.02",
        description="Два синуса на противоположных сторонах, в отдалении от углов\n Самый мелкий шаг по времени (0.02)\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=5.0,
            c_rho=20.0,
            dt=0.02
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.0, 0.5)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.0)(t),
            lambda T, t: TMIN,
        ]
    ),
    NonStatTest(
        name="test_08_two_side_sin_cell=25",
        description="Два синуса на противоположных сторонах, в отдалении от углов\n 25 стержней\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=25,
            cell_size=11,
            thermal_cond=1.0,
            c_rho=20.0,
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.0, 0.5)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.0)(t),
            lambda T, t: TMIN,
        ]
    ),
    NonStatTest(
        name="test_09_two_side_sin_cell=50",
        description="Два синуса на противоположных сторонах, в отдалении от углов\n 50 стержней\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=50,
            cell_size=4,
            thermal_cond=1.0,
            c_rho=20.0,
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.0, 0.5)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.0)(t),
            lambda T, t: TMIN,
        ]
    ),
    NonStatTest(
        name="test_10_two_side_sin_abrupt_heat_change",
        description="Два синуса на противоположных сторонах, в отдалении от углов\nФункция g перестает быть непрерывной по времени. Резкий нагрев с 10-й секунды.\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=15,
            cell_size=6,
            thermal_cond=5.0,
            c_rho=50.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * (0.0 if T < 0.2 * 50.0 else 1.0) * get_g(0.0, 0.5)(t),
            lambda T, t: TMIN,
            lambda T, t: TMIN + (TMAX - TMIN) * (0.0 if T < 0.2 * 50.0 else 1.0) * get_g(0.5, 1.0)(t),
            lambda T, t: TMIN,
        ]
    ),
]
