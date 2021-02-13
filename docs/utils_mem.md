# Helper utility functions for memory management

# About

This module includes helper utilities for memory diagnostics and debug.


If you want to measure relative usages, always preload pytorch kernels first:
```
from ipyexperiments.utils.mem import preload_pytorch
preload_pytorch()
```

Currently these functions rely on pytorch, but can be ported to support other backends.

# API
```
from ipyexperiments.utils.mem import *
```
## gpu_mem_allocate_mbs

`gpu_mem_allocate_mbs(n, fatal=False)`

Try to allocate `n` MBs.

Return the variable holding the allocated memory on success, `None` on failure. The exact allocated memory may be slightly different in size, within 2MB tolerance, due to the granularity of the memory pages.

Delete the variable to free up the memory.

`fatal=True` will throw an exception on failure to allocate (default is `False`).

## gpu_mem_get_free_mbs

`gpu_mem_get_free_mbs()`

Return the amount of free memory (in rounded MBs)


## gpu_mem_get_free_no_cache_mbs

`gpu_mem_get_free_no_cache_mbs()`

Return the amount of free memory after flushing caching (in rounded MBs)

## gpu_mem_get_total_mbs

`gpu_mem_get_total_mbs()`

Return the amount of total memory (in rounded MBs)


## gpu_mem_get_used_mbs

`gpu_mem_get_used_mbs()`

Return the amount of used memory (in rounded MBs)


## gpu_mem_get_used_no_cache_mbs

`gpu_mem_get_used_no_cache_mbs()`

Return the amount of used memory after flushing caching (in rounded MBs)


## gpu_mem_leave_free_mbs

`gpu_mem_leave_free_mbs(n)`

Consume whatever memory is needed so that `n` MBs are left free (approximately).

On success it returns a variable that holds the allocated memory, which needs to be kept alive as long as it's needed to hold that memory. Call `del` to release the memory when it is no longer needed.

This function is very useful in tests/debug situations where the code needs to hit OOM, so this function will leave just the requested amount of GPU RAM free, regardless of GPU utilization or size of the card.

Since typically NVIDIA reports more free memory than it can actually allocate (often as little as 80% of it is actually allocatable) give your code enough of a margin not to get hit by this mismatch of the promised free memory and the reality.

Example:

Create a reproducible test case for *CUDA Out Of Memory* that will work regardless of which GPU card it will run on:

```
from ipyexperiments.utils.mem import gpu_mem_leave_free_mbs
gpu_mem_leave_free_mbs(100)
# now run some code that uses up 100MB and causes CUDA OOM
```


## preload_pytorch

`preload_pytorch()`

Do a small operation on CUDA to get the pytorch/cuda structures in place. A must to be run first if you're going to compare any CUDA-related numbers.
