import numpy as np
from enviroment import NonStatTestParams, NonStatTest

ACTIVATION_L = 0.8
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
        name="test_0_two_side_sin",
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
        name="test_1_two_side_sin",
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
        name="test_2_four_corner_sin",
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
        name="test_3_four_corner_sin",
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
        name="test_4_point_heat_source",
        description="Имитация источника, близкого к точечному\n",
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
        name="test_5_point_heat_source",
        description="Имитация источника, близкого к точечному\n",
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
]

if __name__ == "__main__":
    nonstat_tests[0].init_test_folder()
