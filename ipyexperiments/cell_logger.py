import time, psutil, gc, tracemalloc, threading, weakref
import logging
from collections import namedtuple
from IPython import get_ipython

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
#logger.setLevel(logging.DEBUG)

def b2mb(x): return int(x/2**20)

CellLoggerMemory = namedtuple('CellLoggerMemory', ['used_delta', 'peaked_delta', 'used_total'])
CellLoggerTime   = namedtuple('CellLoggerTime', ['time_delta'])

# all the memory measurements functions come from IPyExperiments subclasses
class CellLogger():

    def __init__(self, exp=None, compact=False, gc_collect=True):

        # any subclass object of IPyExperiments that gives us access to its
        # specific memory measurement functions
        #
        # use weakref so that the parent object can go out of scope and be freed.
        # proxy seems to be simpler to use than weakref.ref, which needs to be
        # called self.exp().foo
        self.exp = weakref.proxy(exp)

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
        #self.input_cells = self.ipython.user_ns['In']

        # set at the end of post_run_cell to be read in the subsequent cell
        self.data = (CellLoggerMemory(0, 0, 0), CellLoggerMemory(0, 0, 0), CellLoggerTime(0))

    def start(self):
        """Register memory profiling tools to IPython instance."""
        self.running = True
        logger.debug("CellLogger: Starting")

        # self.exp does it when needed
        #preload_pytorch()

        # initial measurements
        if self.gc_collect: gc.collect()
        self.cpu_mem_used_prev = self.exp.cpu_ram_used()
        if self.exp.backend != 'cpu':
            self.gpu_mem_used_prev = self.exp.gpu_ram_used()

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

        logger.debug(f"pre_run_cell: 1 f{self.exp}")

        # start RAM tracing
        tracemalloc.start()

        # XXX: perhaps can be replaced with using torch.cuda.reset_max_cached_memory() once pytorch 1.0.1 is released, will need to check that pytorch ver >= 1.0.1
        #
        # this thread samples RAM usage as long as the current cell is running
        if self.exp.backend != 'cpu':
            self.peak_monitoring = True
            peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
            peak_monitor_thread.daemon = True
            peak_monitor_thread.start()

        # time before we execute the current cell
        self.time_start = time.time()


    def post_run_cell(self):
        if not self.running: return

        # don't run this if self.exp doesn't exist anymore (weakref)
        if self.exp is None: return
        logger.debug(f"post_run_cell: 1 f{self.exp}")
        self.time_delta = time.time() - self.time_start

        if self.exp.backend != 'cpu':
            self.peak_monitoring = False

        if self.gc_collect: gc.collect()

        # instead of needing a peak memory monitoring thread, tracemalloc does
        # the job of getting newly used and peaked memory automatically, since
        # it tracks all malloc/free calls.
        cpu_mem_used_delta, cpu_mem_used_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop() # reset accounting

        self.cpu_mem_used_new     = self.exp.cpu_ram_used()
        self.cpu_mem_used_delta   = cpu_mem_used_delta
        self.cpu_mem_peaked_delta = max(0, cpu_mem_used_peak - cpu_mem_used_delta)

        if self.exp.backend != 'cpu':
            self.gpu_mem_used_new     = self.exp.gpu_ram_used()
            self.gpu_mem_used_delta   = self.gpu_mem_used_new - self.gpu_mem_used_prev
            self.gpu_mem_peaked_delta = max(0, self.gpu_mem_used_peak - self.gpu_mem_used_new)

        if self.compact:
            out = f"CPU: {b2mb(self.cpu_mem_used_delta):0.0f}/{b2mb(self.cpu_mem_peaked_delta):0.0f}/{b2mb(self.cpu_mem_used_new):0.0f} MB"
            if self.exp.backend != 'cpu':
                out += f" | GPU: {b2mb(self.gpu_mem_used_delta):0.0f}/{b2mb(self.gpu_mem_peaked_delta):0.0f}/{b2mb(self.gpu_mem_used_new):0.0f} MB"
            out += f" | Time {self.time_delta:0.3f}s | (Consumed/Peaked/Used Total)"
            print(out)
        else:
            pre = '･ '
            print(f"{pre}RAM: △Consumed △Peaked  Used Total | Exec time {self.time_delta:0.3f}s")
            print(f"{pre}CPU:     {b2mb(self.cpu_mem_used_delta):5.0f}   {b2mb(self.cpu_mem_peaked_delta):5.0f}    {b2mb(self.cpu_mem_used_new):5.0f} MB   |")
            if self.exp.backend != 'cpu':
                print(f"{pre}GPU:     {b2mb(self.gpu_mem_used_delta):5.0f}   {b2mb(self.gpu_mem_peaked_delta):5.0f}    {b2mb(self.gpu_mem_used_new):5.0f} MB   |")

        # for self.data accessor
        self.cpu_mem_used_prev = self.cpu_mem_used_new
        if self.exp.backend != 'cpu':
            self.gpu_mem_used_prev = self.gpu_mem_used_new

        self.data = (CellLoggerMemory(self.cpu_mem_used_delta, self.cpu_mem_peaked_delta, self.cpu_mem_used_prev),
                     CellLoggerMemory(self.gpu_mem_used_delta, self.gpu_mem_peaked_delta, self.gpu_mem_used_prev),
                     CellLoggerTime(self.time_delta)
                     )


    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1
        self.gpu_mem_used_peak = -1

        gpu_id     = self.exp.torch.cuda.current_device()
        gpu_handle = self.exp.pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        while True:
            if self.exp is None: break

            # using tracemalloc for tracing peak cpu RAM instead
            #cpu_mem_used = self.exp.cpu_ram_used()
            #self.cpu_mem_used_peak = max(cpu_mem_used, self.cpu_mem_used_peak)

            # no gc.collect, empty_cache here, since it has to be fast and we
            # want to measure only the peak memory usage
            gpu_mem_used = self.exp.gpu_ram_used_fast(gpu_handle)
            self.gpu_mem_used_peak = max(gpu_mem_used, self.gpu_mem_used_peak)

            time.sleep(0.001) # 1msec

            if not self.peak_monitoring: break
