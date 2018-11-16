__all__ = ['IPyExperiments']

import gc
from IPython import get_ipython
from IPython.core.magics.namespace import NamespaceMagics # Used to query namespace.

# XXX: gputil (is not on conda!)

# XXX: for now hardcoding torch dependency, but it should be optional
import torch

import psutil
import humanize
hs = humanize.naturalsize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
process = psutil.Process(os.getpid())
gpu = GPUs[0] # XXX: gputil doesn't seem to be great as it seems to cache values?

# XXX: there could be more than one GPU (or none!)
# XXX: use pytorch to get current gpu if any
def get_gpu(): return (GPU.getGPUs())[0]

def printm():
    gpu = get_gpu()

    """Print memory usage (not exact due to pytorch memory caching)"""
    print("\n*** Current state:")
    print("CPU RAM Free {0:>7s} | Proc size {1}".format(
        hs(psutil.virtual_memory().available),
        hs(process.memory_info().rss)))
    print("GPU RAM Free {0:>7s} | Used {1} | Util {2:2.1f}% | Total {3}".format(
        hs(gpu.memoryFree*1024**2), hs(gpu.memoryUsed*1024**2),
        gpu.memoryUtil*100, hs(gpu.memoryTotal*1024**2)))

class IPyExperiments():
    "Create an experiment with time/memory checkpoint"

    def __init__(self):
        print("Starting experiment...")
        self.running = True
        # base-line
        gc.collect()
        torch.cuda.empty_cache()

        # grab the notebook var names during creation
        ipython = get_ipython()
        self.namespace = NamespaceMagics()
        self.namespace.shell = ipython.kernel.shell
        self.vars = self.get_vars()
        #print(self.vars)
        self.gpu = get_gpu()
        self.gpu_used = self.gpu.memoryUsed*1024**2
        #print(f"gpu used f{self.gpu_used}" )
        self.cpu_used = process.memory_info().rss
        printm()

    def get_vars(self):
        return self.namespace.who_ls()

    def print_mem_stats(self):
        print("Stats:")

    def finish(self):
        """ Finish the experiment """

        print("Finishing experiment...")
        self.gpu = get_gpu()
        cpu_used = process.memory_info().rss
        #print(f"gpu started with {self.gpu_used}")
        gpu_used = self.gpu.memoryUsed*1024**2
        #print(f"gpu used {gpu_used}")
        cpu_consumed = cpu_used - self.cpu_used
        gpu_consumed = gpu_used - self.gpu_used
        #print(f"gpu consumed {gpu_consumed}")
        print("\n*** RAM consumed during the experiment:")
        print(f"CPU: {hs(cpu_consumed) }")
        print(f"GPU: {hs(gpu_consumed)}")

        # get the new vars since constructor
        cur_vars = self.get_vars()
        #print(cur_vars)

        # extract the vars added during the experiment and delete them
        new_vars = list(set(cur_vars) - set(self.vars))
        #print(f"del {new_vars}")
        for x in new_vars: self.namespace.xdel(x)

        # cleanup and reclamation
        collected = gc.collect()
        if collected:       print(f"cleared {collected} objects")
        if len(gc.garbage): print(f"leaked garbage of {len(gc.garbage)} objects")
        torch.cuda.empty_cache()

        # XXX: gputil caches info?
        self.gpu = get_gpu()
        cpu_reclaimed = cpu_used - process.memory_info().rss
        gpu_reclaimed = gpu_used - self.gpu.memoryUsed*1024**2
        cpu_pct = cpu_reclaimed/cpu_consumed if cpu_consumed else 1
        gpu_pct = gpu_reclaimed/gpu_consumed if gpu_consumed else 1
        print("\n*** RAM reclaimed at the end of the experiment:")
        print(f"CPU: {hs(cpu_reclaimed)} ({cpu_pct*100:.2f}%)")
        print(f"GPU: {hs(gpu_reclaimed)} ({gpu_pct*100:.2f}%)")

        printm()

        return 0

    def __del__(self):
        if self.running: self.finish()
