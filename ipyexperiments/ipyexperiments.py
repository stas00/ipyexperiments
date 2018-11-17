__all__ = ['IPyExperiments']

import gc, os, time, psutil
from IPython import get_ipython
from IPython.core.magics.namespace import NamespaceMagics # Used to query namespace.

# XXX: gputil (is not on conda!)

# XXX: for now hardcoding torch dependency, but it should be optional
import torch

import humanize
hs = humanize.naturalsize

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
        self.start_time = time.time()
        self.var_names_keep = []

        # base-line
        gc.collect()
        torch.cuda.empty_cache()

        # grab the notebook var names during creation
        ipython = get_ipython()
        self.namespace = NamespaceMagics()
        self.namespace.shell = ipython.kernel.shell
        self.var_names_start = self.get_var_names()
        #print(self.var_names_start)

        self.gen_ram_used_start = self._gen_ram_used()
        self.gpu_ram_used_start = self._gpu_ram_used()
        #print(f"gpu used f{self.gpu_ram_used_start}" )
        self.print_state()
        print("\n") # extra vertical white space, to not mix with user's outputs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.__del__()

    def keep_var_names(self, *args):
        """ Pass a list of local variable **names** to not be deleted at the end of the experiment """
        for x in args:
            if not isinstance(x, str):
                raise Exception('expecting variable names as strings')
        self.var_names_keep.extend(args)

    def get_var_names(self):
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

        # get the new var names since constructor
        var_names_cur = self.get_var_names()
        #print(var_names_cur)

        # extract the var names added during the experiment and delete
        # them, with the exception of those we were told to preserve
        var_names_new = list(set(var_names_cur) - set(self.var_names_start) - set(self.var_names_keep))
        print("\n*** Deleting the following local variables:")
        print(var_names_new)
        for x in var_names_new: self.namespace.xdel(x)
        if self.var_names_keep:
            print("\n*** Keeping the following local variables:")
            print(self.var_names_keep)

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

        elapsed_time = int(time.time() - self.start_time)
        print("\n*** Elapsed wallclock time:")
        print(f"{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

        self.print_state()

        print("\n") # extra vertical white space, to not mix with user's outputs

        gen_ram_avail, gpu_ram_avail = self._available()
        return self._format_stats(gen_ram_avail, gpu_ram_avail, gen_ram_cons, gpu_ram_cons, gen_ram_recl, gpu_ram_recl)

    def __del__(self):
        """ if explicit finish() wasn't called, do it on obj destruction """
        if self.running:
            self.finish()
