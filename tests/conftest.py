# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import sys
from os.path import abspath, dirname

# make sure we test against the checked out git version of the repo and
# not the pre-installed version. With 'pip install -e .' it's not
# needed, but it's better to be safe and ensure the git path comes
# second in sys.path (the first path is the test dir path)
git_repo_path = abspath(dirname(dirname(__file__)))
sys.path.insert(1, git_repo_path)

# make sure we have pytest-ipynb installed - otherwise half the test suite will be not collected and run
try:
    import pytest_ipynb
except:
    raise ModuleNotFoundError("to run the test suite, run: "
                              "pip install git+https://github.com/stas00/pytest-ipynb.git")
