from IPython import get_ipython
from collections import namedtuple
import datetime
import gc
import logging
import os
import psutil
import random
import sys
import threading
import time

logging.basicConfig(
    format="%(filename)s:%(lineno)s - %(funcName)20s() | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
#logger.setLevel(logging.DEBUG)

def b2mb(x): return int(x/2**20)

def int2width(*n):
    "Find the max length among the int args and add a few for comma-1,000 {:,} repr"
    w = max(map(len, map(str, n)))
    c = int((w-1)/3) # accommodate commas width
    return w + c

def secs2time(secs):
    " secs to time, secs rounded to 3 decimals "
    msec = int(abs(secs-int(secs))*1000)
    return f'{datetime.timedelta(seconds=int(secs))}.{msec:03d}'

def get_nvml_gpu_id(torch_gpu_id):
    """
    Remap torch device id to nvml device id, respecting CUDA_VISIBLE_DEVICES.

    If the latter isn't set return the same id
    """
    # if CUDA_VISIBLE_DEVICES is used automagically remap the id since pynvml ignores this env var
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
        return ids[torch_gpu_id] # remap
    else:
        return torch_gpu_id

process = psutil.Process()
def cpu_ram_used():  return process.memory_info().rss

CellLoggerMemory = namedtuple('CellLoggerMemory', ['used_delta', 'peaked_delta', 'used_total'])
CellLoggerTime   = namedtuple('CellLoggerTime', ['time_delta'])
CellLoggerData   = namedtuple('CellLoggerData', ['cpu', 'gpu', 'time'])

def set_seed(seed=0):
    """
    if seed is not 0 set the passed seed val in python, numpy, pytorch, etc. RNGs (if they are loaded)
    """
    if seed == 0: return
    logger.debug("Setting seed: {seed}")

    # python RNG
    random.seed(seed)

    # numpy RNG
    if 'numpy' in sys.modules:
        import numpy as np # ensure symbol defined
        np.random.seed(seed)

    # pytorch RNGs
    if 'torch' in sys.modules:
        import torch # ensure symbol defined
        torch.manual_seed(seed)          # cpu + cuda
        torch.cuda.manual_seed_all(seed) # multi-gpu
        # slower speed! https://pytorch.org/docs/stable/notes/randomness.html#cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# all the memory measurements functions come from IPyExperiments subclasses
class CellLogger():

    def __init__(self, exp=None, compact=False, gc_collect=True, set_seed=0):

        # any subclass object of IPyExperiments that gives us access to its
        # specific memory measurement functions

        self.backend = exp.backend

        if self.backend == "pytorch":
            self.pynvml = exp.pynvml
            self.torch = exp.torch
            self.gpu_current_device_id = exp.gpu_current_device_id

        self.compact    = compact    # one line printouts
        self.gc_collect = gc_collect # don't use when tracking mem leaks
        self.set_seed   = set_seed   # set RNG seed before each cell is run to the provided value

        self.peak_monitoring = False
        self.running         = False

        self.time_start = 0
        self.time_delta = 0

        self.cpu_mem_used_peak    = -1
        self.cpu_mem_used_delta   =  0
        self.cpu_mem_used_prev    = -1
        self.cpu_mem_peaked_delta = -1

        self.gpu_mem_used_peak    = -1
        self.gpu_mem_used_delta   =  0
        self.gpu_mem_used_prev    = -1
        self.gpu_mem_peaked_delta = -1

        self.ipython = get_ipython()
        #self.input_cells = self.ipython.user_ns['In']

        # set at the end of post_run_cell to be read in the subsequent cell
        self.data = CellLoggerData(
            CellLoggerMemory(0, 0, 0),
            CellLoggerMemory(0, 0, 0),
            CellLoggerTime(0)
        )

    # XXX: all this needs to be refactored - tired of hunting lock deadlocks, so just as well drop
    # the idea of having this extendable to other backends for now and just use it for pytorch
    def gpu_clear_cache(self): self.torch.cuda.empty_cache()
    def gpu_ram(self):
        """ for the currently selected GPU device return: total, free and used RAM in bytes """
        self.gpu_clear_cache() # clear cache to report the correct data
        nvml_gpu_id = get_nvml_gpu_id(self.gpu_current_device_id)
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
        info   = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total, info.free, info.used

    def gpu_ram_used(self):  return self.gpu_ram()[2]
    def gpu_ram_avail(self): return self.gpu_ram()[1]
    # use cached handle and clear no cache
    def gpu_ram_used_fast(self, gpu_handle): return self.pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used

    def start(self):
        """Register memory profiling tools to IPython instance."""
        self.running = True
        logger.debug("CellLogger: Starting")

        # self.exp does it when needed
        #preload_pytorch()

        # initial measurements
        if self.gc_collect: gc.collect()
        self.cpu_mem_used_prev = cpu_ram_used()
        if self.backend == "pytorch":
            self.gpu_mem_used_prev = self.gpu_ram_used()
        self.ipython.events.register("pre_run_cell",  self.pre_run_cell)
        logger.debug(f"registered pre_run_cell: {self.pre_run_cell}")
        self.ipython.events.register("post_run_cell", self.post_run_cell)
        logger.debug(f"registered post_run_cell: {self.post_run_cell}")

        # run pre_run_cell() manually, since we are past that event in this cell
        self.pre_run_cell(None)

        return self


    def stop(self):
        """Unregister memory profiling tools from IPython instance."""
        if not self.running: return
        logger.debug("CellLogger: Stopping")

        self.ipython.events.unregister("pre_run_cell",  self.pre_run_cell)
        self.ipython.events.unregister("post_run_cell", self.post_run_cell)

        # try: self.ipython.events.unregister("pre_run_cell",  self.pre_run_cell)
        # except ValueError:
        #     print("Failed to unregister: pre_run_cell ")
        #     pass
        # try: self.ipython.events.unregister("post_run_cell", self.post_run_cell)
        # except ValueError:
        #     print("Failed to unregister: pre_run_cell ")
        #     pass
        #if self.peak_monitor_thread and self.peak_monitor_thread.is_alive():
        #    self.peak_monitor_thread._Thread__stop()

        self.peak_monitoring = False

        # run post_run_cell() manually, since it's no longer registered
        self.post_run_cell(None)

        self.running = False


    def pre_run_cell(self, info):
        # seed reset
        if self.set_seed != 0: set_seed(self.set_seed)

        self.cpu_mem_used_at_cell_start = cpu_ram_used()
        if self.backend == "pytorch":
            self.gpu_mem_used_at_cell_start = self.gpu_ram_used()

        # XXX: perhaps can be replaced with using torch.cuda.reset_max_cached_memory() once pytorch 1.0.1 is released, will need to check that pytorch ver >= 1.0.1
        #
        # this thread samples RAM usage as long as the current cell is running
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

        # time before we execute the current cell
        self.time_start = time.time()


    def post_run_cell(self, result):
        if not self.running: return

        # this sends a signal to peak_monitor_func to complete its loop
        self.peak_monitoring = False

        self.time_delta = time.time() - self.time_start

        if self.gc_collect: gc.collect()

        # tracemalloc was tried, but it misses all non-python memory allocations so it had to go

        self.cpu_mem_used_new = cpu_ram_used()
        self.cpu_mem_used_delta = self.cpu_mem_used_new - self.cpu_mem_used_at_cell_start
        # see the logic for gpu below for details of the following
        self.cpu_mem_peaked_delta = max(0, self.cpu_mem_used_peak - self.cpu_mem_used_at_cell_start)
        if self.cpu_mem_used_delta > 0:
            self.cpu_mem_peaked_delta = max(0, self.cpu_mem_peaked_delta - self.cpu_mem_used_delta)

        if self.backend == "pytorch":
            self.gpu_mem_used_new = self.gpu_ram_used()

            # delta_used is the difference between used mem at current vs. at cell start
            self.gpu_mem_used_delta = self.gpu_mem_used_new - self.gpu_mem_used_at_cell_start

            # peaked_delta is the temporary overhead if any.
            #
            # the idea is that in order to know how much memory the cell consumed one needs to sum
            # up used delta and peaked delta
            #
            # It is calculated as follows:
            #
            # 1. The difference between the peak memory and the used memory at the
            # start is measured:
            # 2a. If it's negative, then peaked_delta is 0
            # 2b. Otherwise, if used_delta is positive it gets subtracted from peaked_delta
            # XXX: 2a shouldn't be needed once we have a reliable peak counter
            self.gpu_mem_peaked_delta = max(0, self.gpu_mem_used_peak - self.gpu_mem_used_at_cell_start)
            if self.gpu_mem_used_delta > 0:
                self.gpu_mem_peaked_delta = max(0, self.gpu_mem_peaked_delta - self.gpu_mem_used_delta)


        if self.compact:
            if 1:
                out  = f"CPU: {b2mb(self.cpu_mem_used_delta):0.0f}/{b2mb(self.cpu_mem_peaked_delta):0.0f}/{b2mb(self.cpu_mem_used_new):0.0f} MB"
            if self.backend == "pytorch":
                out += f" | GPU: {b2mb(self.gpu_mem_used_delta):0.0f}/{b2mb(self.gpu_mem_peaked_delta):0.0f}/{b2mb(self.gpu_mem_used_new):0.0f} MB"
            out += f" | Time {secs2time(self.time_delta)} | (Consumed/Peaked/Used Total)"
            print(out)
        else:
            if 1:
                vals  = [self.cpu_mem_used_delta, self.cpu_mem_peaked_delta, self.cpu_mem_used_new]
            if self.backend == "pytorch":
                vals += [self.gpu_mem_used_delta, self.gpu_mem_peaked_delta, self.gpu_mem_used_new]
            w = int2width(*map(b2mb, vals)) + 1 # some air
            if w < 10: w = 10 # accommodate header width
            pre = '･ '
            print(f"{pre}RAM: {'△Consumed':>{w}} {'△Peaked':>{w}}    {'Used Total':>{w}} | Exec time {secs2time(self.time_delta)}")
            if 1:
                print(f"{pre}CPU: {b2mb(self.cpu_mem_used_delta):{w},.0f} {b2mb(self.cpu_mem_peaked_delta):{w},.0f} {b2mb(self.cpu_mem_used_new):{w},.0f} MB |")
            if self.backend == "pytorch":
                print(f"{pre}GPU: {b2mb(self.gpu_mem_used_delta):{w},.0f} {b2mb(self.gpu_mem_peaked_delta):{w},.0f} {b2mb(self.gpu_mem_used_new):{w},.0f} MB |")

        # for self.data accessor
        self.cpu_mem_used_prev = self.cpu_mem_used_new
        if self.backend == "pytorch":
            self.gpu_mem_used_prev = self.gpu_mem_used_new

        self.data = CellLoggerData(
            CellLoggerMemory(self.cpu_mem_used_delta, self.cpu_mem_peaked_delta, self.cpu_mem_used_prev),
            CellLoggerMemory(self.gpu_mem_used_delta, self.gpu_mem_peaked_delta, self.gpu_mem_used_prev),
            CellLoggerTime(self.time_delta)
        )


    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1
        self.gpu_mem_used_peak = -1

        if self.backend == "pytorch":
            torch_gpu_id = self.torch.cuda.current_device()
            nvml_gpu_id = get_nvml_gpu_id(torch_gpu_id)
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)

        while True:
            self.cpu_mem_used_peak = max(cpu_ram_used(), self.cpu_mem_used_peak)

            if self.backend == "pytorch":
                # no gc.collect, empty_cache here, since it has to be fast and we
                # want to measure only the peak memory usage
                gpu_mem_used = self.gpu_ram_used_fast(handle)
                self.gpu_mem_used_peak = max(gpu_mem_used, self.gpu_mem_used_peak)

            # can't sleep or will not catch the peak right
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring: break
