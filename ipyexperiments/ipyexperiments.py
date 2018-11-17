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
# gpu = GPUs[0] # XXX: gputil doesn't seem to be great as it seems to cache values?

process = psutil.Process(os.getpid())

# XXX: there could be more than one GPU (or none!)
# XXX: use pytorch to get current gpu if any
# XXX: it seems that I have to call this function every time before a change in the memory consumption of the card, otherwise it gives me old data! i.e. can't cache this object
def get_gpu(): return (GPU.getGPUs())[0]

class IPyExperiments():
    "Create an experiment with time/memory checkpoint"

    def __init__(self):
        print("Starting experiment...")
        self.running = True
        self.reclaimed = False

        # base-line
        gc.collect()
        torch.cuda.empty_cache()

        # grab the notebook var names during creation
        ipython = get_ipython()
        self.namespace = NamespaceMagics()
        self.namespace.shell = ipython.kernel.shell
        self.vars = self.get_vars()
        #print(self.vars)

        self.gen_ram_used_start = self._gen_ram_used()
        self.gpu_ram_used_start = self._gpu_ram_used()
        #print(f"gpu used f{self.gpu_ram_used_start}" )
        self.print_state()

    def get_vars(self):
        """ Return a list of local variables created since the beginning of the experiment """
        return self.namespace.who_ls()

    def _gen_ram_used(self):  return int(process.memory_info().rss)
    def _gen_ram_avail(self): return int(psutil.virtual_memory().available)
    def _gpu_ram_used(self):  return int(get_gpu().memoryUsed*1024**2)
    def _gpu_ram_avail(self): return int(get_gpu().memoryFree*1024**2)
    def _available(self):     return self._gen_ram_avail(), self._gpu_ram_avail()

    def _consumed(self):
        gen_ram_cons = self._gen_ram_used() - self.gen_ram_used_start
        gpu_ram_cons = self._gpu_ram_used() - self.gpu_ram_used_start
        #print(f"gpu started with {self.gpu_ram_used_start}")
        #print(f"gpu consumed {gpu_ram_cons}")
        return gen_ram_cons, gpu_ram_cons

    def _reclaimed(self):
        # return 0s, unless called from finish() after memory reclamation
        if self.reclaimed:
            gen_ram_recl = self.gen_ram_used_start + self.gen_ram_cons - self._gen_ram_used()
            gpu_ram_recl = self.gpu_ram_used_start + self.gpu_ram_cons - self._gpu_ram_used()
        else:
            gen_ram_recl = 0
            gpu_ram_recl = 0
        return gen_ram_recl, gpu_ram_recl

    def _format_stats(self, gen_ram_avail, gpu_ram_avail, gen_ram_cons, gpu_ram_cons, gen_ram_recl,  gpu_ram_recl ):
        cons  = {'gen_ram': gen_ram_cons,  'gpu_ram': gpu_ram_cons }
        recl  = {'gen_ram': gen_ram_recl,  'gpu_ram': gpu_ram_recl }
        avail = {'gen_ram': gen_ram_avail, 'gpu_ram': gpu_ram_avail}
        return cons, recl, avail

    def get_stats(self):
        """ Return current stats """
        gen_ram_avail, gpu_ram_avail = self._available()
        gen_ram_cons,  gpu_ram_cons  = self._consumed()
        gen_ram_recl,  gpu_ram_recl  = self._reclaimed()
        return self._format_stats(gen_ram_avail, gpu_ram_avail, gen_ram_cons, gpu_ram_cons, gen_ram_recl, gpu_ram_recl)

    def print_state(self):
        """ Print memory stats (not exact due to pytorch memory caching) """
        print("\n*** Current state:")
        print("Gen RAM Free {0:>7s} | Proc size {1}".format(
            hs(self._gen_ram_avail()),
            hs(self._gen_ram_used())))
        gpu = get_gpu()
        print("GPU RAM Free {0:>7s} | Used {1} | Util {2:2.1f}% | Total {3}".format(
            hs(gpu.memoryFree*1024**2), hs(gpu.memoryUsed*1024**2),
            gpu.memoryUtil*100, hs(gpu.memoryTotal*1024**2)))

    def finish(self):
        """ Finish the experiment, reclaim memory, return final stats """

        print("Finishing experiment...")
        self.running = False

        # first take the final snapshot of consumed resources
        gen_ram_cons,  gpu_ram_cons = self._consumed()
        self.gen_ram_cons = gen_ram_cons
        self.gpu_ram_cons = gpu_ram_cons

        # get the new vars since constructor
        cur_vars = self.get_vars()
        #print(cur_vars)

        # extract the vars added during the experiment and delete them
        new_vars = list(set(cur_vars) - set(self.vars))
        print("\n*** Deleting the following local variables:")
        print(new_vars)
        for x in new_vars: self.namespace.xdel(x)

        # cleanup and reclamation
        collected = gc.collect()
        if collected or len(gc.garbage):
            print("\n*** Potential memory leaks during the experiment:")
            if collected:       print(f"cleared {collected} objects")
            if len(gc.garbage): print(f"leaked garbage of {len(gc.garbage)} objects")
        # now we can attempt to reclaim GPU memory
        torch.cuda.empty_cache()
        self.reclaimed = True

        # now we can measure how much was reclaimed
        gen_ram_recl,  gpu_ram_recl  = self._reclaimed()
        gen_ram_pct = gen_ram_recl/gen_ram_cons if gen_ram_cons else 1
        gpu_ram_pct = gpu_ram_recl/gpu_ram_cons if gpu_ram_cons else 1

        print("\n*** RAM consumed during the experiment:")
        print(f"Gen: {hs(gen_ram_cons) }")
        print(f"GPU: {hs(gpu_ram_cons)}")
        print("\n*** RAM reclaimed at the end of the experiment:")
        print(f"Gen: {hs(gen_ram_recl)} ({gen_ram_pct*100:.2f}%)")
        print(f"GPU: {hs(gpu_ram_recl)} ({gpu_ram_pct*100:.2f}%)")

        self.print_state()

        gen_ram_avail, gpu_ram_avail = self._available()
        return self._format_stats(gen_ram_avail, gpu_ram_avail, gen_ram_cons, gpu_ram_cons, gen_ram_recl, gpu_ram_recl)

    def __del__(self):
        """ if explicit finish() wasn't called, do it on obj destruction """
        if self.running:
            self.finish()
