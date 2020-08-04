# TODO

- try to switch to the pytorch counter instead of peak monitor thread. The 2 functions are:
torch.cuda.max_memory_allocated()
torch.cuda.reset_max_memory_allocated
Unfortunately there can be only one counter - so it'd be tricky to measure different scopes (cell-level and the whole experiment)
