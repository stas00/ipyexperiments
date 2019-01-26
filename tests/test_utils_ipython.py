import pytest
from ipyexperiments.utils.ipython import *

# at the moment just a syntax check, the test would be useless w/o ipython env

@ipython_tb_clear_frames
def do_something(): return True

def test_decorator():
    assert do_something() is True, "decorator test"

def test_ctx():
    with ipython_tb_clear_frames_ctx():
        x = 10
