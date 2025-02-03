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
        name="test_0",
        description="Исторически первый тест.\nДва синуса.\n\nПараметры теста:\n\n",
        params=NonStatTestParams(
            cell=30,
            cell_size=6,
            thermal_cond=1.0,
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
        name="test_1",
        description="Новый тест с синусами в углах\n",
        params=NonStatTestParams(
            cell=30,
            cell_size=6,
            thermal_cond=1.0,
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
        name="test_2",
        description="Новый тест с синусами в углах, но шире, чем в предыдущем тесте\n",
        params=NonStatTestParams(
            cell=30,
            cell_size=6,
            thermal_cond=1.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.5)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-0.5, 0.5)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-0.5, 0.5)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.5)(t),
        ]
    ),
        NonStatTest(
        name="test_3",
        description="Новый тест с синусами в углах, но уже, чем в предыдущем тесте\n",
        params=NonStatTestParams(
            cell=30,
            cell_size=6,
            thermal_cond=1.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(1.0 - 1.0/30, 1.0 + 1.0/30)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-1.0/30, 1.0/30)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-1.0/30, 1.0/30)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(1.0 - 1.0/30, 1.0 + 1.0/30)(t),
        ]
    ),
    NonStatTest(
        name="test_4",
        description="Новый тест с синусами в углах, но шире, чем во втором тесте\n",
        params=NonStatTestParams(
            cell=10,
            cell_size=11,
            thermal_cond=20.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.5)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-0.5, 0.5)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(-0.5, 0.5)(t),
            lambda T, t: TMIN + (TMAX - TMIN) * get_activation(50.0)(T) * get_g(0.5, 1.5)(t),
        ]
    ),
    NonStatTest(
        name="test_5",
        description="Узкий синус в угле, но с высокой температурой\n",
        params=NonStatTestParams(
            cell=10,
            cell_size=11,
            thermal_cond=20.0,
            c_rho=20.0
        ),
        f=lambda T, x, y: 0.0,
        g=[
            lambda T, t: TMIN + 4*(TMAX - TMIN) * get_activation(50.0)(T) * get_g(1.0 - 1.0/5, 1.0 + 1.0/5)(t),
            lambda T, t: TMIN + 4*(TMAX - TMIN) * get_activation(50.0)(T) * get_g(-1.0/5, 1.0/5)(t),
            lambda T, t: TMIN + 4*(TMAX - TMIN) * get_activation(50.0)(T) * get_g(-1.0/5, 1.0/5)(t),
            lambda T, t: TMIN + 4*(TMAX - TMIN) * get_activation(50.0)(T) * get_g(1.0 - 1.0/5, 1.0 + 1.0/5)(t),
        ]
    ),
]

if __name__ == "__main__":
    nonstat_tests[0].init_test_folder()
