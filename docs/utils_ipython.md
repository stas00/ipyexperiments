# ipython/jupyter helper tools

## About

This module includes helper utilities for memory management in the ipython/jupyter environment.

Currently, the main functions are to provide workaround solutions for [memory leakage on exception under ipython/jupyter](https://github.com/ipython/ipython/pull/11572).

## API
```
from ipyexperiments.utils.ipython import *
```

### is_in_ipython

`is_in_ipython()`

Is the code running in the ipython environment (jupyter-notebook/ipython/ipython-notebook).

Returns `True` or `False`.

### ipython_tb_clear_frames

`ipython_tb_clear_frames()`

Reclaim general/GPU RAM on any exception under ipython environment (decorator).

ipython has a feature where it stores tb with all the locals() tied in, which prevents gc.collect from freeing those variables, therefore we cleanse the tb before handing it over to ipython. For more information see [this answer](https://stackoverflow.com/a/54295910/9201239).

This is a decorator to be used with any function that needs this workaround: *) under non-ipython environment it doesn't do anything. *) under ipython it strips tb of `locals()`

This workaround has a side-effect of not being able to use `%debug` or `%pdb` magic, because the debugger needs the `locals()` of each involved frame to be there.

Usage:

```
@ipython_tb_clear_frames
def myfunc(...)
```

### ipython_tb_clear_frames_ctx

`ipython_tb_clear_frames_ctx()`

Reclaim general/GPU RAM on any exception under ipython environment (context manager).

See the details in the `ipython_tb_clear_frames` docstring.

If the code to be protected is not a function, use this context manager.

For example:
```
with ipython_tb_clear_frames_ctx():
    my_code()
```
