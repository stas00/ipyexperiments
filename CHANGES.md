# Changes

## 0.1.24 (2021-02-13)

- fix preload_pytorch to load the 0th device by default
- fix the test suite with CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID to that cuda and pynvml report the same gpus
- `preload_pytorch` is no longer automatically run in ipyexperiments.utils.mem - as it may interfere with other systems. it's an explicit call now.

## 0.1.23 (2020-11-16)

- bug fix
- require pytest-ipynb for the test suite


## 0.1.22 (2020-11-15)

- add a workaround for pynvml not respecting CUDA_VISIBLE_DEVICES settings
- adjust ipyexperiments.utils.mem getter functions to accept an optional torch gpu id
- adjust Makefile to test for conda install/uninstall first, otherwise it breaks pip


## 0.1.21 (2020-11-12)

- fix conda channels

- for consistency renamed helper functions:
   * `gpu_mem_get_free_no_cache_mbs`
   * `gpu_mem_get_used_no_cache_mbs`
   (added `_mbs` like all the other functions that return rounded mbs)


## 0.1.20 (2020-11-12)

- fix requirements - needed pynvml instead of the nvidia-ml-py3 that it relied on earlier


## 0.1.19 (2020-11-08)

- skipped


## 0.1.18 (2020-11-08)

- build fix


## 0.1.18 (2020-11-08)

- require ipython>=6.0.0 (colab still uses ipython==5.5.0, which breaks events)
- py38 is supported (doc)


## 0.1.17 (2020-03-30)

- fix bug in ipython callback functions definition
- add cl_set_seed=SEED to automatically reset the seed to a fixed value in each cell


## 0.1.16 (2019-03-12)

- workaround for unreliable peak thread (avoid negative numbers)
- add locking to avoid thread race-condition

## 0.1.15 (2019-02-24)

- dynamically format column widths in reports
- add , in large numbers (1,000)


## 0.1.14 (2019-02-24)

- Delta Peaked calculation fixed in case of consumed memory being negative (released)
- switch to pynvx's pynvml wrapper


## 0.1.13 (2019-02-22)

- fix MANIFEST.in to include project sub-dirs


## 0.1.12 (2019-02-22)

- use pynvx as pynvml replacement on mac OS (thanks @phenomax)
- switched GPUMemory tuple order to be consistent with nvidia meminfo
- handle the situation where a mem peak thread gets delayed to finish
- some self.namespace.shell.user_ns vars disappear, so do a safe delete
- expose/document more of `ipyexperiments.utils.mem` helper functions

## 0.1.11 (2019-01-25)

- add `ipyexperiments.utils.ipython` module with workaround for ipython memory leak on exception problem.
- add `ipyexperiments.utils.mem` module with memory debug helper utils.


## 0.1.10 (2019-01-18)

- exp.data and exp.cl.data now both return namedtuples for the top-level entries, so it'll be easier to extend features in the future - documenting only the dictionary access, instead of raw tuples.
- cl: time delta is now printed in hh:mm:ss.msec


## 0.1.9 (2019-01-15)

- integrate IPyGPULogger as a subsystem (renamed to CellLogger), so now there is no need to install and manage two modules that share a lot of similarities.
- fix a bug where ipyexperiment obj was getting deleted when a cell was rerun


## 0.1.8 (2019-01-05)

- fix bug in memory Utilization calculation


## 0.1.6 (2019-01-04)

- on GPU backend loading report the ID, Name and Total RAM of the selected GPU
- print_state now gives an easier to read report

### Breaking changes

- made the module into proper subclasses, no more global function aliases. So now use directly the desired backend: `IPyExperimentsCPU`, `IPyExperimentsPytorch` as an experiments module. It should be trivial now to add other backends.
- and `get_stats` method has been replaced with `data` property method, which now returns one or more `IPyExperimentMemory` named tuple(s) depending on the used subclass.


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
