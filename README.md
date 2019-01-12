
[![pypi ipyexperiments version](https://img.shields.io/pypi/v/ipyexperiments.svg)](https://pypi.python.org/pypi/ipyexperiments)
[![Conda ipyexperiments version](https://img.shields.io/conda/v/stason/ipyexperiments.svg)](https://anaconda.org/stason/ipyexperiments)
[![Anaconda-Server Badge](https://anaconda.org/stason/ipyexperiments/badges/platforms.svg)](https://anaconda.org/stason/ipyexperiments)
[![ipyexperiments python compatibility](https://img.shields.io/pypi/pyversions/ipyexperiments.svg)](https://pypi.python.org/pypi/ipyexperiments)
[![ipyexperiments license](https://img.shields.io/pypi/l/ipyexperiments.svg)](https://pypi.python.org/pypi/ipyexperiments)

# ipyexperiments

jupyter/ipython experiment containers for profiling and reclaiming GPU and general RAM , and detecting memory leaks.

## About

This module's main purpose is to help calibrate hyper parameters in deep learning notebooks to fit the available GPU and General RAM, but, of course, it can be useful for any other use where memory limits is a constant issue. It is also useful for detecting memory leaks in your code.

Using this framework you can run multiple consequent experiments without needing to restart the kernel all the time, especially when you run out of GPU memory - the familiar to all "cuda: out of memory" error. When this happens you just go back to the notebook cell where you started the experiment, change the hyper parameters, and re-run the updated experiment until it fits the available memory. This is much more efficient and less error-prone then constantly restarting the kernel, and re-running the whole notebook.

As an extra bonus you get access to the memory consumption data, so you can use it to automate the discovery of the hyper parameters to suit your hardware's unique memory limits.

The idea behind this module is very simple - it implements a python function-like functionality, where its local variables get destroyed at the end of its run, giving us memory back, except it'll work across multiple jupyter notebook cells (or ipython). In addition it also runs `gc.collect()` to immediately release badly behaved variables with circular references, and reclaim general and GPU RAM. It also helps to discover memory leaks, and performs various other useful things behind the scenes.

If you need a more fine-grained memory profiling, the `CellLogger` sub-system reports RAM usage on a per cell-level when used with jupyter or per line of code in ipython.  You get the resource usage report automatically as soon as a command or a cell finished executing.

Currently this sub-system logs GPU RAM, general RAM and execution time. But can be expanded to track other important thing. While there are various similar loggers out there, the main focus of this implementation is to help track GPU, whose main scarce resource is GPU RAM.

![Usage demo](https://raw.githubusercontent.com/stas00/ipyexperiments/master/docs/images/usage1.png)

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
cell 1: exp1 = IPyExperimentsPytorch()
cell 2:   learn1 = language_model_learner(data_lm, bptt=60, drop_mult=0.25, pretrained_model=URLs.WT103)
cell 3:   learn1.lr_find()
cell 4: del exp1
cell 5: exp2 = IPyExperimentsPytorch()
cell 6:   learn2 = language_model_learner(data_lm, bptt=70, drop_mult=0.3, pretrained_model=URLs.WT103)
cell 7:   learn2.lr_find()
cell 8: del exp2
```

## Demo

See [this demo notebook](https://github.com/stas00/ipyexperiments/blob/master/demo.ipynb), to see how this system works.


## Documentation

1. [IPyExperiments](https://github.com/stas00/ipyexperiments/blob/master/docs/ipyexperiments.md).
2. [CellLogger sub-system](https://github.com/stas00/ipyexperiments/blob/master/docs/cell_logger.md)


## Contributing

PRs with improvements and new features and Issues with suggestions are welcome.

If you work with `tensorflow`, please, consider sending a PR to support it - by mimicking the `IPyExperimentsPytorch` implementation.

## Testing

1. Install my fork of `pytest-ipynb` (the original one is no longer being maintained)
   ```
   pip install git+https://github.com/stas00/pytest-ipynb.git
   ```
2. Run the test suite
   ```
   make test
   ```


## History

A detailed history of changes can be found [here](https://github.com/stas00/ipyexperiments/blob/master/CHANGES.md).
