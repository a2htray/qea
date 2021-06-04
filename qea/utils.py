import numpy as np

def init_solution(rs: np.random.RandomState, num_dimensions, bound_lows, bound_highs):
    return init_vector(rs, num_dimensions, bound_lows, bound_highs)


def init_velocity(rs: np.random.RandomState, num_dimensions, init_velocity_lows, init_velocity_highs):
    return init_vector(rs, num_dimensions, init_velocity_lows, init_velocity_highs)


def init_vector(rs: np.random.RandomState, n, lows, highs):
    return lows + rs.uniform(0, 1, n) * (highs - lows)