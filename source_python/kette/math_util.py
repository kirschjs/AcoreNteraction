import numpy as np


def log_fact(n):
    """Calculates `ln(n!)` by adding `ln(n) + ln(n-1) + ...`

    Raises `ValueError` if non-integral or negative values are passed.
    """
    # TODO: Consider benchmarking vs naive implementation of `np.log(...)`.
    if n < 0:
        raise ValueError('Factorial not defined for negative numbers')
    if not isinstance(n, int):
        raise ValueError('Factorial only defined for integral values')

    result = 0
    while n > 0:
        result += np.log(n)
        n -= 1

    return result