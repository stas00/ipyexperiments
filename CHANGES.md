# Changes


## 0.1.5 (2018-12-28)

- revert the attempt to account for modified variables - it doesn't work if some function during the experiment modified a variable introduced before the experiment - we surely must not delete it.

## 0.1.4 (2018-12-28)

- fixed the method for detecting which variables to delete. Moved from comparing a list of variables in the ipython user namespace before and after the experiment (which was missing on variables that were previously defined) via `ipython`'s `namespace.who_ls()` to saving the objects for each variable and then comparing whether they have changed at the end of the experiment. So, now, all variables defined during the experiment get correctly deleted (unless explicitly told otherwise).

## 0.1.3 (2018-12-19)

- replace human size function to do 1024, instead of 1000-computations

## 0.1.2 (2018-12-18)

- create a test suite - cover the main cpu tests

## 0.1.1 (2018-12-18)

- add conda/pypi releases

## 0.1.0 (2018-11-19)

- the core of the project implemented
