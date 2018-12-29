
[![pypi ipyexperiments version](https://img.shields.io/pypi/v/ipyexperiments.svg)](https://pypi.python.org/pypi/ipyexperiments)
[![Conda ipyexperiments version](https://img.shields.io/conda/v/stason/ipyexperiments.svg)](https://anaconda.org/stason/ipyexperiments)
[![Anaconda-Server Badge](https://anaconda.org/stason/ipyexperiments/badges/platforms.svg)](https://anaconda.org/stason/ipyexperiments)
[![ipyexperiments python compatibility](https://img.shields.io/pypi/pyversions/ipyexperiments.svg)](https://pypi.python.org/pypi/ipyexperiments)
[![ipyexperiments license](https://img.shields.io/pypi/l/ipyexperiments.svg)](https://pypi.python.org/pypi/ipyexperiments)

# ipyexperiments

jupyter/ipython experiment containers for GPU and general RAM re-use and memory leaks detection.

## About

This module's main purpose is to help calibrate hyper parameters in deep learning notebooks to fit the available GPU and General RAM, but, of course, it can be useful for any other use where memory limits is a constant issue. It is also useful for detecting memory leaks in your code.

Using this framework you can run multiple consequent experiments without needing to restart the kernel all the time, especially when you run out of GPU memory - the familiar to all "cuda: out of memory" error. When this happens you just go back to the notebook cell where you started the experiment, change the hyper parameters, and re-run the updated experiment until it fits the available memory. This is much more efficient and less error-prone then constantly restarting the kernel, and re-running the whole notebook.

As an extra bonus you get access to the memory consumption data, so you can use it to automate the discovery of the hyper parameters to suit your hardware's unique memory limits.

The idea behind this module is very simple - it implements a python function-like functionality, where its local variables get destroyed at the end of its run, giving us memory back, except it'll work across multiple jupyter notebook cells (or ipython). In addition it also runs `gc.collect()` to immediately release badly behaved variables with circular references, and reclaim general and GPU RAM. It also helps to discover memory leaks, and performs various other useful things behind the scenes.


## Installation

* pypi:

   ```
   pip install ipyexperiments
   ```
* conda:

   ```
   conda install -c fastai -c stason ipyexperiments
   ```

* dev:

   ```
   pip install git+https://github.com/stas00/ipyexperiments.git
   ```

## Usage

Here is an example with using code from the [`fastai`](https://github.com/fastai/fastai) library.

Please, note, that I added a visual leading space to demonstrate the idea, but, of course, it won't be a valid python code.

```
cell 1: exp1 = IPyExperiments()
cell 2:   learn1 = language_model_learner(data_lm, bptt=60, drop_mult=0.25, pretrained_model=URLs.WT103)
cell 3:   learn1.lr_find()
cell 4: del exp1
cell 5: exp2 = IPyExperiments()
cell 6:   learn2 = language_model_learner(data_lm, bptt=70, drop_mult=0.3, pretrained_model=URLs.WT103)
cell 7:   learn2.lr_find()
cell 8: del exp2
```

## Demo

The easiest way to see how this framework works is to read the [demo notebook](https://github.com/stas00/ipyexperiments/blob/master/demo.ipynb).

## Backends

Backends allow experimentation with different GPU frameworks, like `pytorch`, `tensorflow`, etc.

Currently `cpu` and `pytorch` backends are supported. `pytorch` is the default backend, it'll be used if you don't specify any other.

If you don't have a GPU or if you have it, but you don't use it for the experiment, you can initiate:

   ```python
   exp1 = IPyExperiments(backend='cpu')
   ```

Additional machine learning backends can be easily supported. Just submit a PR after adding a few lines of code in `backend_load()` to support the backend you desire. The description of what's needed is in the comments of that method - it should be very easy to do.

Please, note, that this module doesn't setup its `pip`/`conda` dependencies for the backend frameworks, since you must have already installed those before attempting to use this module.

## Multiple GPUs

This framework currently works with the GPU that is currently selected by the backend. For most users it'll be `id=0`. If you instructed your backend (e.g. `pytorch`) to use a different GPU, `IPyExperiments` will know to use that GPU instead. For example, with the `pytorch` backed, it's discovered with: `torch.cuda.current_device()`.

It can be extended to support multiple-GPUs concurrently, but I have only one GPU at the moment, so you are welcome to submit PRs supporting stats for multiple GPUs at the same time.

## API

```
from ipyexperiments import *
```

1. Create an experiment object:
   ```python
   exp1 = IPyExperiments()
   ```
   By default, the `pytorch` backend is used. You can change that to load a `cpu`-only mode, with:
   ```python
   exp1 = IPyExperiments('cpu')
   ```
   More backends will be supported in the future.

2. Get intermediary experiment usage stats:
   ```python
   consumed, reclaimed, available = exp1.get_stats()
   ```
   3 dictionaries are returned. This way is used so that in the future new entries could be added w/o breaking the API. The memory stats are in bytes.

   ```python
   print(consumed, reclaimed, available)
   {'gen_ram': 2147500032, 'gpu_ram': 0} {'gen_ram': 0, 'gpu_ram': 0} {'gen_ram': 9921957888, 'gpu_ram': 7487881216}
   ```
   This method is useful for getting stats half-way through the experiment.

3. Save specific local variables to be accessible after the experiment is finished and the rest of the local variables have been deleted.

   ```python
   exp3.keep_var_names('consumed', 'reclaimed', 'available')
   ```
   Note, that you need to pass the names of the variables and not the variables themselves.

4. Finish the experiment, delete local variables, reclaim memory. Return and print the stats:
   ```python
   final_consumed, final_reclaimed, final_available = exp1.finish() # finish experiment
   print("\nNumerical data:\n", final_consumed, final_reclaimed, final_available)
   ```

   If you don't care for saving the experiment's numbers, instead of calling `finish()`, you can just do:
   ```python
   del exp1
   ```
   If you re-run the experiment without either calling `exp1.finish()` or `del exp1`, e.g. if you decided to abort it half-way to the end, or say you hit "cuda: out of memory" error, then re-running the constructor `IPyExperiments()` assigning to the same experiment object, will trigger a destructor first. This will delete the local vars created until that point, reclaim memory and the previous experiment's stats will be printed first.

5. Context manager is supported:

   ```python
   with IPyExperiments():
       x1 = consume_cpu(2**14)
       x2 = consume_gpu(2**14)
   ```
   except, it won't be very useful if you want to use more than one notebook cell.

   If you need to access the experiment object use:

   ```python
   with IPyExperiments() as exp:
       x1 = consume_cpu(2**14)
       x2 = consume_gpu(2**14)
       exp.keep_var_names('x1')
   ```

Please refer to the [demo notebook](https://github.com/stas00/ipyexperiments/blob/master/demo.ipynb) to see this API in action.


## Framework Preloading and Memory Leak Detection

If you haven't asked for any local variables to be saved via `keep_var_names()` and if the process finished with big chunks of memory un-reclaimed - guess what - most likely you have just discovered a memory leak in your code. If all the local variables/objects were destroyed you should normally get all of the general and GPU RAM reclaimed in a well-behaved code.

You do need to be aware that some frameworks consume a big chunk of general and GPU RAM when they are used for the first time. For example `pytorch` `cuda` [eats up](
https://docs.fast.ai/dev/gpu.html#unusable-gpu-ram-per-process) about 0.5GB of GPU RAM and 2GB of general RAM on load (not necessarily on `import`, but usually later when it's used), so if your experiment started with doing a `cuda` action for the first time in a given process, expect to lose that much RAM - this one can't be reclaimed.

But `IPyExperiments` does all this for you, for example, preloading `pytorch` `cuda` if the `pytorch` backend (default) is used. During the preloading it internally does:

   ```python
   import pytorch
   torch.ones((1, 1)).cuda() # preload pytorch with cuda libraries
   ```

## Caveats

The module only deletes newly defined local variables. i.e. those that didn't exist before the experiment started and were created after it has started.

Due to the nature of access to `ipython`'s local variable list (there is just the list of names), there is no way of telling whether a certain variable has been redefined inside the scope of the experiment. The current algorithm compares the list of local variables before the experiment, and after it - and that's how it knows which ones should be deleted and their memory reclaimed.

I tried to solve this by tracking variables that changed, presumably those would be the ones reused inside the experiment, which mostly worked, but unfortunately it was incorrectly deleting some variables that were not introduced by an experiment, but happened to be modified through some function called during the experiment. Also it failed if the variable was assigned the same value as before the experiment (since there was no change in its value).

So if you do this:

```
# cell1
x1 = 1
# cell 2
with IPyExperiments():
    x1 = 10
    x2 = 20
```

only `x2` will be deleted at the end of the experiment, as there is no certain way to programmatically tell whether the local variable `x1` was introduced during the experiment or before it.

Watch the printout when the experiment completes to ensure all the desired variables get removed. For example in the example above we would get:
```
*** Deleting the following local variables:
['x2']
```

To work around this problem use unique variable names inside experiments, as compared to variables in code outside of experiments.

It's perfectly fine though to use the same variable names across different experiments:
```
# cell1
with IPyExperiments():
    x1 = 10
    x2 = 20
# cell 2
with IPyExperiments():
    x1 = 100
    x2 = 200
```
as long as you don't hop from one experiment to another without completing the first one first. It won't be a problem in this example where the experiment is contained to a single cell, but I'm referring to the more common situation, where it's spread out across many cells. Once an experiment is completed and its local vars get deleted, they can be safely defined again in another experiment.

If you have some brilliant insights on how to resolve this conundrum I'm all ears.


## Contributing

PRs with improvements and new features and Issues with suggestions are welcome.

If you work with `tensorflow`, please, consider sending a PR to support it - by mimicking the `pytorch`-backend implementation.

## History

A detailed history of changes can be found [here](https://github.com/stas00/ipyexperiments/blob/master/CHANGES.md).
