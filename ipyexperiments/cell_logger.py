import time, psutil, gc, tracemalloc, threading, weakref
import logging
from collections import namedtuple
from IPython import get_ipython

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
#logger.setLevel(logging.DEBUG)

have_cuda = 0
import torch
if torch.cuda.is_available():
    have_cuda = 1
    import pynvml
    pynvml.nvmlInit()

process = psutil.Process()

def preload_pytorch():
    if have_cuda: torch.ones((1, 1)).cuda()

def cpu_mem_used_get():
    "process used memory in MBs rounded down"
    return int(process.memory_info().rss/2**20)

def gpu_mem_used_get():
    "query nvidia for used memory for gpu in MBs (rounded down). If id is not passed, currently selected torch device is used. Clears pytorch cache before taking the measurements"
    torch.cuda.empty_cache() # clear cache to report the correct data
    id = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return int(info.used/2**20)

# similar to gpu_mem_used_get, but doesn't do any checks, clearing caches,
# gc.collect, etc., to be lightening fast when run in a tight loop from a peak
# memory measurement thread.
def gpu_mem_used_get_fast(gpu_handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return int(info.used/2**20)

CellLoggerMemory = namedtuple('CellLoggerMemory', ['used_delta', 'peaked_delta', 'used_total'])
CellLoggerTime   = namedtuple('CellLoggerTime', ['time_delta'])

# all the memory measurements functions come from IPyExperiments subclasses
class CellLogger():

    def __init__(self, exp=None, compact=False, gc_collect=True):

        # any subclass object of IPyExperiments that gives us access
        # to its specific memory measurement functions
        # use weakref so that the parent object can go out of scope
        # and finish
        self.exp = weakref.ref(exp)

        self.compact    = compact    # one line printouts
        self.gc_collect = gc_collect # don't use when tracking mem leaks

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
        self.input_cells = self.ipython.user_ns['In']

    @property
    def data(self):
        return (CellLoggerMemory(self.cpu_mem_used_delta, self.cpu_mem_peaked_delta, self.cpu_mem_used_prev),
                CellLoggerMemory(self.gpu_mem_used_delta, self.gpu_mem_peaked_delta, self.gpu_mem_used_prev),
                CellLoggerTime(self.time_delta)
        )

    def start(self):
        """Register memory profiling tools to IPython instance."""
        self.running = True
        logger.debug("CellLogger: Starting")

        # exp does it when needed
        #preload_pytorch()

        # initial measurements
        if self.gc_collect: gc.collect()
        self.cpu_mem_used_prev = cpu_mem_used_get()
        self.gpu_mem_used_prev = gpu_mem_used_get()

        self.ipython.events.register("pre_run_cell",  self.pre_run_cell)
        self.ipython.events.register("post_run_cell", self.post_run_cell)

        # run pre_run_cell() manually, since we are past that event in this cell
        self.pre_run_cell()

        return self


    def stop(self):
        """Unregister memory profiling tools from IPython instance."""
        if not self.running: return
        logger.debug("CellLogger: Stopping")

        try: self.ipython.events.unregister("pre_run_cell",  self.pre_run_cell)
        except ValueError: pass
        try: self.ipython.events.unregister("post_run_cell", self.post_run_cell)
        except ValueError: pass

        # run post_run_cell() manually, since it's no longer registered
        self.post_run_cell()

        self.running         = False
        self.peak_monitoring = False


    def pre_run_cell(self):
        if not self.running: return

        self.peak_monitoring = True

        # start RAM tracing
        tracemalloc.start()

        # this thread samples RAM usage as long as the current cell is running
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

        # time before we execute the current cell
        self.time_start = time.time()


    def post_run_cell(self):
        if not self.running: return

        self.time_delta = time.time() - self.time_start

        self.peak_monitoring = False

        if self.gc_collect: gc.collect()

        # instead of needing a peak memory monitoring thread, tracemalloc does
        # the job of getting newly used and peaked memory automatically, since
        # it tracks all malloc/free calls.
        cpu_mem_used_delta, cpu_mem_used_peak = list(map(lambda x: x/2**20, tracemalloc.get_traced_memory()))
        tracemalloc.stop() # reset accounting

        self.cpu_mem_used_new     = cpu_mem_used_get()
        self.cpu_mem_used_delta   = cpu_mem_used_delta
        self.cpu_mem_peaked_delta = max(0, cpu_mem_used_peak - cpu_mem_used_delta)

        self.gpu_mem_used_new     = gpu_mem_used_get()
        self.gpu_mem_used_delta   = self.gpu_mem_used_new - self.gpu_mem_used_prev
        self.gpu_mem_peaked_delta = max(0, self.gpu_mem_used_peak - self.gpu_mem_used_new)

        # not really useful, as the report is right next to the cell, the cell
        # counts aren't fixed, if re-run
        # cell_num = len(self.input_cells) - 1

        if (self.compact):
            print(f"CPU: {self.cpu_mem_used_delta:0.0f}/{self.cpu_mem_peaked_delta:0.0f}/{self.cpu_mem_used_new:0.0f} MB | GPU: {self.gpu_mem_used_delta:0.0f}/{self.gpu_mem_peaked_delta:0.0f}/{self.gpu_mem_used_new:0.0f} MB | Time {self.time_delta:0.3f}s | (Consumed/Peaked/Used Total)")
        else:
            print(f"RAM: Consumed Peaked  Used Total | Exec time {self.time_delta:0.3f}s")
            print(f"CPU:    {self.cpu_mem_used_delta:5.0f}  {self.cpu_mem_peaked_delta:5.0f}    {self.cpu_mem_used_new:5.0f} MB |")
            print(f"GPU:    {self.gpu_mem_used_delta:5.0f}  {self.gpu_mem_peaked_delta:5.0f}    {self.gpu_mem_used_new:5.0f} MB |")

        # for self.data accessor
        self.cpu_mem_used_prev = self.cpu_mem_used_new
        self.gpu_mem_used_prev = self.gpu_mem_used_new


    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1
        self.gpu_mem_used_peak = -1

        gpu_id = torch.cuda.current_device()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        while True:

            # using tracemalloc for tracing peak cpu RAM instead
            #cpu_mem_used = cpu_mem_used_get()
            #self.cpu_mem_used_peak = max(cpu_mem_used, self.cpu_mem_used_peak)

            # no gc.collect, empty_cache here, since it has to be fast and we
            # want to measure only the peak memory usage
            gpu_mem_used = gpu_mem_used_get_fast(gpu_handle)
            self.gpu_mem_used_peak = max(gpu_mem_used, self.gpu_mem_used_peak)

            time.sleep(0.001) # 1msec

            if not self.peak_monitoring: break
