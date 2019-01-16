# CellLogger

## Demo

See [this demo notebook](https://github.com/stas00/ipyexperiments/blob/master/demo_cl.ipynb), to see how this sub-system works.

## API

1. Initiate the subsystem:
   ```python
   exp = IPyExperimentsPytorch(cl_enable=True, cl_compact=False, cl_gc_collect=True)
   # exp.cl is the subsystem object
   ```
   Parameters:
   * `cl_enable` - enable the subsystem
   * `cl_compact` - use compact one line printouts
   * `cl_gc_collect` - get correct memory usage reports. Don't use when tracking memory leaks (objects with circular reference).

   If you just want to get the per cell/line logging, pass `exp_enable=False` to disable the parent `IPyExperiments` system,

2. Start logging if it wasn't started in the constructor, or manually stopped via `.stop()`:
   ```python
   exp.cl.start()
   ```

3. Stop logging
   ```python
   exp.cl.stop()
   ```

4. Access the measured data directly (in exact bytes)
   ```python
   cpu_mem, gpu_mem, time_data = exp.cl.data
   print(cpu_mem)
   print(gpu_mem)
   print(time_data)

   ```
   gives:
   ```
   CellLoggerMemory(used_delta=128.0062427520752, peaked_delta=128.0133180618286, used_total=2282)
   CellLoggerMemory(used_delta=1024, peaked_delta=1024, used_total=3184)
   CellLoggerTime(time_delta=0.806537389755249)
   ```
   This accessor returns 3 `namedtuple`s, so that you can access the data fields by name. For example, continuing from above.

   ```python
   print(cpu_mem.used_delta)
   print(gpu_mem.used_total)
   print(time_data.time_delta)
   ```
   or to unpack it:
   ```python
   cpu_mem_used_delta, cpu_mem_peaked_delta, cpu_mem_used_total = cpu_mem
   gpu_mem_used_delta, gpu_mem_peaked_delta, gpu_mem_used_total = gpu_mem
   ```

Please refer to the [demo notebook](https://github.com/stas00/ipyexperiments/blob/master/demo_cl.ipynb) to see this API in action.

The main API is documented [here](./ipyexperiments.md#API)


## Peak Memory Usage

Often, a function may use more RAM than if we were just to measure the memory usage before and after its execution, therefore this module uses a thread to take snapshots of its actual memory usage during its run. So when the report is printed you get to see the maximum memory that was required to run this function.

For example if the report was:

```
RAM: Consumed Peaked  Used Total in 0.000s (In [4])
CPU:        0      0    170 MB
GPU:     2567   1437   5465 MB
```

That means that when the function finished it consumed `2467 MB` of GPU RAM, as compared to the memory usage before its run. However, it actually needed a total of `4000 MB` of GPU RAM to be able to run (`2467`+`1437`). So if you didn't have `4000 MB` of free unfragmented RAM available it would have failed.

## Framework Preloading

You do need to be aware that some frameworks consume a big chunk of general and GPU RAM when they are used for the first time. For example `pytorch` `cuda` [eats up](
https://docs.fast.ai/dev/gpu.html#unusable-gpu-ram-per-process) about 0.5GB of GPU RAM and 2GB of general RAM on load (not necessarily on `import`, but usually later when it's used), so if your experiment started with doing a `cuda` action for the first time in a given process, expect to lose that much RAM - this one can't be reclaimed.

But `CellLogger` does all this for you, for example, preloading `pytorch` `cuda` if the `pytorch` backend (default) is used. During the preloading it internally does:

   ```python
   import pytorch
   torch.ones((1, 1)).cuda() # preload pytorch with cuda libraries
   ```

## Cache Clearing

Before a snapshot of used GPU RAM is made, its cache is cleared, since otherwise there is no way to get any real GPU RAM usage. So this module gives very reliable data on GPU RAM (but also see [Temporary Memory Leaks](#temporary-memory-leaks).

For general RAM accounting, [tracemalloc](https://docs.python.org/3/library/tracemalloc.html) is used, which gives a correct report of how much RAM was allocated and peaked, which overcomes the interference of python internal caching. So if the code makes a large memory allocation, followed by its release and immediately a smaller allocation - the process's total memory usage won't change, yet it'll report how much RAM a specific cell has allocated.

## Temporary Memory Leaks

Modern (py-3.4+)` gc.collect()` handles circular references in objects, including those with custom `__del__` methods. So pretty much eventually, when `gc.collect()` arrives, all deleted objects get reclaimed. The problem is that in the environments like machine learning training, eventually is not good enough. Some objects that are no longer needed could easily hold huge chunks of GPU RAM and waiting for `gc.collect()` to arrive is very unreliable bad method of handling that. Moreover, allocating more GPU RAM before freeing RAM that is already not serving you leads to memory fragmentation, which is a very bad scenario - as you may have a ton of free GPU RAM, but none can be used. And at least at the moment, NVIDIA's CUDA doesn't have a *defrag* method.

In order to give you correct memory usage numbers, this module by default runs `gc.collect` before clearing GPU cache and taking a measurement of its used memory. But this could mask problems in your code, and if you turn this module off, suddenly the same code can't run on the same amount of GPU RAM.

So, make sure you compare your total GPU RAM consumption with and without `gc_collect=True` in the object `CellLogger` constructor.


## IPyExperiments's main documentation

[IPyExperiments](https://github.com/stas00/ipyexperiments/blob/master/docs/ipyexperiments.md)


## Credits

* This subsystem borrowed the peak RAM monitoring thread idea from [ipython_memwatcher](https://github.com/FrancescAlted/ipython_memwatcher) by Francesc Alted