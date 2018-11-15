# ipyexperiments
experiment containers for jupyter/ipython for GPU and general RAM re-use

# About
It's difficult to use functions in jupyter notebook, since we want different steps to be in different cells, so one of the main functions of this module is to emulate a function like scope of the variables - which get destroyed at the end of the experiment. Some extra magic is added to reclaim GPU and General RAM.

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
