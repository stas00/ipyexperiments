""" ipython helper tools """

############# ipython on exception memory leak prevention helpers ############

import os, functools, traceback

def is_in_ipython():
    "Is the code running in the ipython environment (jupyter including)"

    program_name = os.path.basename(os.getenv('_', ''))

    if ('jupyter-notebook' in program_name or # jupyter-notebook
        'ipython'          in program_name or # ipython
        'JPY_PARENT_PID'   in os.environ):    # ipython-notebook
        return True
    else:
        return False

IS_IN_IPYTHON = is_in_ipython()

def ipython_tb_clear_frames(func):
    """Reclaim general/GPU RAM on any exception under ipython environment (decorator)

    ipython has a feature where it stores tb with all the locals() tied in, which
    prevents gc.collect from freeing those variables, therefore we cleanse the tb
    before handing it over to ipython.

    This is a decorator to be used with any function that needs this workaround:
    *) under non-ipython environment it doesn't do anything.
    *) under ipython it strips tb of `locals()`

    This workaround has a side-effect of not being able to use `%debug` or
    `%pdb` magic, because the debugger needs the `locals()` of each involved
    frame to be there.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not IS_IN_IPYTHON:
            return func(*args, **kwargs)

        try:
            return func(*args, **kwargs)
        except:
            type, val, tb = sys.exc_info()
            traceback.clear_frames(exc_tb)
            raise type(val).with_traceback(tb) from None
    return wrapper


class ipython_tb_clear_frames_ctx():
    """Reclaim general/GPU RAM on any exception under ipython environment (context manager).

    See the details in the `ipython_tb_clear_frames` docstring.

    If the code to be protected is not a function, use this context manager.

    For example:
    ```
    with ipython_tb_clear_frames_ctx():
        my_code()
    ```

    """
    def __enter__(self): return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_val: return True
        traceback.clear_frames(exc_tb)
        raise exc_type(exc_val).with_traceback(exc_tb) from None
