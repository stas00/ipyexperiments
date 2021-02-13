# TODO

- try to switch to the pytorch counter instead of peak monitor thread. The 2 functions are:
torch.cuda.max_memory_allocated()
torch.cuda.reset_max_memory_allocated
Unfortunately there can be only one counter - so it'd be tricky to measure different scopes (cell-level and the whole experiment)

- still looking for a modern replacement for pytest-ipynb

tried pytest-notebook but can't figure out how to make it stop comparing the outputs - in this project the reported numbers are very inconsistent, since there are many external things that can influence those, so the default functionality of pytest-notebook is a no-go.

But if I come back to it:

pip install pytest-notebook
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID pytest --color=yes --disable-warnings --nb-test-files --log-cli-level=info tests
