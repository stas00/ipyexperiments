import pytest
from ipyexperiments.utils.mem import *
from math import isclose

def test_leave_free_mbs():
    delta = 10
    free_before = gpu_mem_get_free_mbs()
    assert free_before > delta, "have some free gpu memory"

    left = free_before - delta
    buf = gpu_mem_leave_free_mbs(left)
    assert buf is not None, f"able to allocate {delta} mem"

    free_after = gpu_mem_get_free_mbs()
    assert isclose(free_after, left, abs_tol=1), f"allocated {delta}, before {free_before}, after {free_after}"
