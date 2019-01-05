__all__ = ['IPyExperimentsCPU', 'IPyExperimentsPytorch']

import gc, os, sys, time, psutil
from IPython import get_ipython
from IPython.core.magics.namespace import NamespaceMagics # Used to query namespace.
from collections import namedtuple

IPyExperimentMemory = namedtuple('IPyExperimentMemory', ['consumed', 'reclaimed', 'available'])

process = psutil.Process(os.getpid())

# light weight humanize from https://stackoverflow.com/a/1094933/9201239 w/ tweaks
def hs(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0: return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Y', suffix)

class IPyExperiments():
    "Create an experiment with time/memory checkpoints"

    def __init__(self):
        """ Instantiate an object with parameters:
        """

        self.running = True
        self.reclaimed = False
        self.start_time = time.time()
        self.var_names_keep = []

        # grab the notebook var names during creation
        ipython = get_ipython()
        self.namespace = NamespaceMagics()
        self.namespace.shell = ipython.kernel.shell
        self.var_names_start = self.get_var_names()
        #print(self.var_names_start)

        # The following doesn't work:
        #
        # we have to take a snapshot of all the variables and their references,
        # so that when the experiment is over we can discover which variables were
        # used in the scope of the experiment, including ones that were defined
        # prior to the experiment (which would otherwise be missed if only
        # variables names before and after are compared).
        #self.var_start = {k:self.namespace.shell.user_ns[k] for k in self.var_names_start}

    def backend_init(self): pass

    def start(self):
        #print("Starting IPyExperiments")
        # base-line
        gc.collect()
        self.gpu_clear_cache()

        self.cpu_ram_used_start = self.cpu_ram_used()
        self.gpu_ram_used_start = self.gpu_ram_used()
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
                raise ValueError('expecting variable names as strings')
        self.var_names_keep.extend(args)

    def get_var_names(self):
        """ Return a list of local variables created since the beginning of the experiment """
        return self.namespace.who_ls()

    def _available(self): return self.cpu_ram_avail(), self.gpu_ram_avail()

    def _consumed(self):
        cpu_ram_cons = self.cpu_ram_used() - self.cpu_ram_used_start
        gpu_ram_cons = self.gpu_ram_used() - self.gpu_ram_used_start
        #print(f"gpu started with {self.gpu_ram_used_start}")
        #print(f"gpu consumed {gpu_ram_cons}")
        return cpu_ram_cons, gpu_ram_cons

    def _reclaimed(self):
        # return 0s, unless called from finish() after memory reclamation
        if self.reclaimed:
            cpu_ram_recl = self.cpu_ram_used_start + self.cpu_ram_cons - self.cpu_ram_used()
            gpu_ram_recl = self.gpu_ram_used_start + self.gpu_ram_cons - self.gpu_ram_used()
        else:
            cpu_ram_recl = 0
            gpu_ram_recl = 0
        return cpu_ram_recl, gpu_ram_recl

    def _data_format(self, cpu_ram_avail, cpu_ram_cons, cpu_ram_recl,
                           gpu_ram_avail, gpu_ram_cons, gpu_ram_recl):
        if self.backend == 'cpu':
            return (IPyExperimentMemory(cpu_ram_cons, cpu_ram_recl, cpu_ram_avail))
        else:
            return (IPyExperimentMemory(cpu_ram_cons, cpu_ram_recl, cpu_ram_avail),
                    IPyExperimentMemory(gpu_ram_cons, gpu_ram_recl, gpu_ram_avail))

    @property
    def data(self):
        """ Return current data """
        cpu_ram_avail, gpu_ram_avail = self._available()
        cpu_ram_cons,  gpu_ram_cons  = self._consumed()
        cpu_ram_recl,  gpu_ram_recl  = self._reclaimed()
        return self._data_format(cpu_ram_avail, cpu_ram_cons, cpu_ram_recl,
                                 gpu_ram_avail, gpu_ram_cons, gpu_ram_recl)

    def print_state(self):
        """ Print memory stats """
        print("\n*** Current state:")
        print(f"RAM:   Used      Free     Total    Util")

        cpu_ram_total, cpu_ram_free, cpu_ram_used = self.cpu_ram()
        cpu_ram_util = cpu_ram_used/cpu_ram_total*100 if cpu_ram_total else 100
        print(f"CPU: {hs(cpu_ram_used):>8s}  {hs(cpu_ram_free):>8s} {hs(cpu_ram_total):>8s} {cpu_ram_util:6.2f}% ")
        if self.backend == 'cpu': return

        gpu_ram_total, gpu_ram_free, gpu_ram_used = self.gpu_ram()
        gpu_ram_util = gpu_ram_used/gpu_ram_total*100 if gpu_ram_total else 100
        print(f"GPU: {hs(gpu_ram_used):>8s}  {hs(gpu_ram_free):>8s} {hs(gpu_ram_total):>8s} {gpu_ram_util:6.2f}% ")


    def finish(self):
        """ Finish the experiment, reclaim memory, return final stats """

        self.running = False

        elapsed_time = int(time.time() - self.start_time)
        print(f"\n*** Experiment finished in {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} (elapsed wallclock time)")

        # first take the final snapshot of consumed resources
        cpu_ram_cons,  gpu_ram_cons = self._consumed()
        self.cpu_ram_cons = cpu_ram_cons
        self.gpu_ram_cons = gpu_ram_cons

        # get the new var names since constructor
        var_names_cur = self.get_var_names()
        #print(var_names_cur)

        # XXX: this doesn't work, since some variables get modified during the
        # experiment, but indirectly and therefore shouldn't be deleted.
        # So the idea of comparing values before and after doesn't quite work.
        #
        # only newly introduced variables, or variables that have been re-used
        # changed_vars = [k for k in var_names_cur
        #            if not (k in self.var_start and self.namespace.shell.user_ns[k] is self.var_start[k])]

        # extract the var names added during the experiment and delete
        # them, with the exception of those we were told to preserve
        var_names_new = list(set(var_names_cur) - set(self.var_names_start) - set(self.var_names_keep))
        print("\n*** Local variables:")
        print("Deleted:", ", ".join(sorted(var_names_new)))
        for x in var_names_new: self.namespace.xdel(x)
        if self.var_names_keep:
            print("Kept:   ", ", ".join(sorted(self.var_names_keep)))

        # cleanup and reclamation
        collected = gc.collect()
        if collected:
            print("\n*** Circular ref objects gc collected during the experiment:")
            print(f"cleared {collected} objects (only temporary leakage)")
        if len(gc.garbage):
            print("\n*** Potential memory leaks during the experiment:")
            print(f"uncollected gc.garbage of {len(gc.garbage)} objects")
        # now we can attempt to reclaim GPU memory
        self.gpu_clear_cache()
        self.reclaimed = True

        # now we can measure how much was reclaimed
        cpu_ram_recl,  gpu_ram_recl  = self._reclaimed()
        cpu_ram_pct = cpu_ram_recl/cpu_ram_cons if cpu_ram_cons else 1
        gpu_ram_pct = gpu_ram_recl/gpu_ram_cons if gpu_ram_cons else 1

        print("\n*** Experiment memory:")
        print(f"RAM:  Consumed     Reclaimed")

        print(    f"CPU: {hs(cpu_ram_cons):>8s} {hs(cpu_ram_recl):>8s} ({cpu_ram_pct*100:6.2f}%)")
        if self.backend != 'cpu':
            print(f"GPU: {hs(gpu_ram_cons):>8s} {hs(gpu_ram_recl):>8s} ({gpu_ram_pct*100:6.2f}%)")

        self.print_state()

        print("\n") # extra vertical white space, to not mix with user's outputs

        cpu_ram_avail, gpu_ram_avail = self._available()
        return self._data_format(cpu_ram_avail, cpu_ram_cons, cpu_ram_recl,
                                 gpu_ram_avail, gpu_ram_cons, gpu_ram_recl)


    def __del__(self):
        # if explicit finish() wasn't called, do it on self-destruction
        if self.running: self.finish()


# currently supporting:
# - IPyExperimentsCPU: no gpu backend
# - IPyExperimentsPytorch: pytorch backend
#
# How to add support for new backends:
#
# in order to add a new backend, add a new subclass w/
# 1. import backend module
# 2. preload code that claims unreclaimable gpu memory
# 3. set the current gpu id
# 4. function that releases the cache if any
# 5. etc. - model after the IPyExperimentsPytorch subclass

class IPyExperimentsCPU(IPyExperiments):
    """ CPU backend can be used directly for non-gpu setups """

    def __init__(self):
        super().__init__()
        self.backend = 'cpu'
        self.has_gpu = False
        if self.__class__.__name__ == 'IPyExperimentsCPU':
            #print("Starting IPyExperimentsCPU")
            self.backend_init()
            self.start()

    def backend_init(self):
        super().backend_init()
        self.gpu_current_device_id = -1
        if self.__class__.__name__ == 'IPyExperimentsCPU':
            print("\n*** Experiment started with the CPU-only backend")

    #def start(self):
    #    #print("Starting IPyExperimentsCPU")
    #    super().start()

    def cpu_ram(self): return (int(psutil.virtual_memory().total),
                               int(psutil.virtual_memory().available),
                               int(process.memory_info().rss))
    def cpu_ram_total(self): return self.cpu_ram()[0]
    def cpu_ram_avail(self): return self.cpu_ram()[1]
    def cpu_ram_used(self):  return self.cpu_ram()[2]

    def gpu_ram(self): return 0, 0, 0
    def gpu_ram_used(self):  return 0
    def gpu_ram_avail(self): return 0

    def gpu_clear_cache(self): pass


class IPyExperimentsGPU(IPyExperimentsCPU):
    """ generic GPU backend must be subclassed by a specific backend before being used """

    def __init__(self):
        super().__init__()
        self.backend = 'gpu-generic'
        self.has_gpu = True
        # not a class to be used directly:
        # self.backend_init()
        # self.start()

    def backend_init(self):
        super().backend_init()

        try:
            import pynvml
        except Exception as e:
            raise Exception(f"{e}\nYou need to install the nvidia-ml-py3 module; pip install nvidia-ml-py3")

        # initialize pynvml
        self.pynvml = pynvml
        pynvml.nvmlInit()

    #def start(self):
    #    #print("Starting IPyExperimentsGPU")
    #    super().start()

    def gpu_ram(self):
        """ for the currently selected GPU device return: total, free and used RAM in bytes """
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.gpu_current_device_id)
        info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total, info.free, info.used

    def gpu_ram_used(self):  return self.gpu_ram()[2]
    def gpu_ram_avail(self): return self.gpu_ram()[1]

    def gpu_clear_cache(self): pass


class IPyExperimentsPytorch(IPyExperimentsGPU):

    def __init__(self):
        super().__init__()
        self.backend = 'pytorch'
        if self.__class__.__name__ == 'IPyExperimentsPytorch':
            #print("Starting IPyExperimentsPytorch")
            self.backend_init()
            self.start()

    def backend_init(self):
        super().backend_init()
        print("\n*** Experiment started with the Pytorch backend")

        import torch
        self.torch = torch

        # sanity check
        if not torch.cuda.is_available():
            raise Exception(f"torch.cuda.is_available() returns False; can't continue")

        self.gpu_current_device_id = self.torch.cuda.current_device()

        # force pytorch to pre-load cuDNN and its kernels to claim unreclaimable memory
        torch.ones((1, 1)).cuda()

        # check that all is ready to go, and we get the RAM info
        gpu_ram_total, gpu_ram_free, gpu_ram_used = self.gpu_ram()

        # announce which device is used for this experiment
        print(f"Device: ID {self.gpu_current_device_id}, {torch.cuda.get_device_name(self.gpu_current_device_id)} ({hs(gpu_ram_total)} RAM)")

    #def start(self):
    #    super().start()

    def gpu_clear_cache(self): self.torch.cuda.empty_cache()
