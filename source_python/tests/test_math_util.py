import os
import sys

import numpy as np
import pytest

# This is a horrible hack and should never, ever (!) leave this file. Replace ASAP.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from kette.math_util import log_fact


def test_log_fact_integer():
    assert log_fact(0) == 0
    assert log_fact(1) == 0
    assert log_fact(2) == np.log(2)


def test_log_fact_negative():
    # Note: `pytest.raises(e)` is a context manager that will fail if no exception
    # of type `e` was thrown in its inner block.
    with pytest.raises(ValueError):
        log_fact(-1)


def test_log_fact_non_integer():
    with pytest.raises(ValueError):
        log_fact(1.0)