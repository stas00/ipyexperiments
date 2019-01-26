# Helper utility functions for memory management

# About

This module includes helper utilities for memory diagnostics and debug.

# API
```
from ipyexperiments.utils.mem import *
```
## gpu_mem_allocate_mbs

`gpu_mem_allocate_mbs(n)`

Allocate `n` MBs. Return the variable holding the allocated memory on success, `None` on failure. There exact allocated memory may be slightly different in size, with 1MB tolerance.

## gpu_mem_leave_free_mbs

`gpu_mem_leave_free_mbs(n)`

Consume whatever memory is needed so that `n` MBs are left free (approximately).

On success it returns a variable that holds the allocated memory, which needs to be kept alive as long as it's needed to hold that memory. Call `del` to release the memory when that memory is no longer needed.

This is very useful in tests/debug situations where the code needs to hit OOM, so this function will leave just the requested amount of GPU RAM free, regardless of GPU utilization or size of the card.

Since typically NVIDIA reports more free memory than it can actually allocate (often as little as 80% of it is actually allocatable) give your code enough of a margin not to get hit by this mismatch of the promised free memory and the reality.

Example:

Create a reproducible test case for *CUDA Out Of Memory* that will work regardless of which GPU card it will run on:

```
from ipyexperiments.utils.mem import *
gpu_mem_leave_free_mbs(100)
# now run some code that uses up 100MB and causes CUDA OOM
```
