# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

# note that to run ipynb tests run:
#
# pip install nbmake
# pytest --nbmake


import sys, os
from pathlib import Path

# make sure we test against the checked out git version of the repo and
# not the pre-installed version. With 'pip install -e .' it's not
# needed, but it's better to be safe and ensure the git path comes
# second in sys.path (the first path is the test dir path)
git_repo_path_str = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, git_repo_path_str)
