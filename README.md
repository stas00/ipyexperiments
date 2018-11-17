# ipyexperiments
experiment containers for jupyter/ipython for GPU and general RAM re-use

# About
It's difficult to use functions in jupyter notebook, since we want different steps to be in different cells, so one of the main functions of this module is to emulate a function like scope of the variables - which get destroyed at the end of the experiment. Some extra magic is added to reclaim GPU and General RAM.

Using this method you can run many experiments w/o needing to restart the kernel all the time, especially when you run out of CUDA memory. You just rollback to the beginning of the experiment, change the parameters, and run the updated experiment.

## Usage

Here is an example with using code from the [`fastai`](https://github.com/fastai/fastai) library. I added a visual leading space to demonstrate the idea, but of course it won't be valid python.

```
cell 1: exp1 = IPyExperiments()
cell 2:   learn1 = language_model_learner(data_lm, bptt=70, drop_mult=0.3, pretrained_model=URLs.WT103)
cell 3:   learn1.lr_find()
cell 4: del exp1
cell 5: exp2 = IPyExperiments()
cell 6:   learn2 = language_model_learner(data_lm, bptt=70, drop_mult=0.3, pretrained_model=URLs.WT103)
cell 7:   learn2.lr_find()
cell 8: del exp2
```

## Demo
[demo notebook](https://github.com/stas00/ipyexperiments/blob/master/demo.ipynb)

## Installation
pip install git+https://github.com/stas00/ipyexperiments.git

## API

1. Create an experiment object:
   ```python
   exp1 = IPyExperiments()
   ```

2. Get intermediary experiment usage stats:
   ```python
   consumed, reclaimed, available = exp1.get_stats()
   ```
   3 dictionaries are returned. This way is used so that in the future new entries could be added w/o breaking the API.

   ```python
   print(consumed, reclaimed, available)
   {'gen_ram': 2147500032, 'gpu_ram': 0} {'gen_ram': 0, 'gpu_ram': 0} {'gen_ram': 9921957888, 'gpu_ram': 7487881216}
   ```
   This method is useful for getting stats half-way through the experiment.

3. Finish the experiment, delete local variables, reclaim memory. Return and print the stats:
   ```python
   final_consumed, final_reclaimed, final_available = exp1.finish() # finish experiment
   print("\nNumerical data:\n", final_consumed, final_reclaimed, final_available)
   ```

   If you don't care for saving the experiment numbers, instead of calling `finish()`, you can just do:
   ```python
   del exp1
   ```
   If you re-run the experiment w/o either calling `exp1.finish()` or `del exp1`, e.g. if you decided to abort it half-way to the end, then the constructor `IPyExperiments()` will trigger a destructor first and therefore previous experiment's stats will be printed first.
