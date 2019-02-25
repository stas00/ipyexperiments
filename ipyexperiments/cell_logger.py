import time, psutil, gc, tracemalloc, threading, weakref, datetime
import logging
from collections import namedtuple
from IPython import get_ipython

logging.basicConfig()
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

CellLoggerMemory = namedtuple('CellLoggerMemory', ['used_delta', 'peaked_delta', 'used_total'])
CellLoggerTime   = namedtuple('CellLoggerTime', ['time_delta'])
CellLoggerData   = namedtuple('CellLoggerData', ['cpu', 'gpu', 'time'])

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
        self.data = CellLoggerData(
            CellLoggerMemory(0, 0, 0),
            CellLoggerMemory(0, 0, 0),
            CellLoggerTime(0)
        )

    def start(self):
        """Register memory profiling tools to IPython instance."""
        self.running = True
        logger.debug("CellLogger: Starting")

        # this seem to be unreliable if the parent goes away, and the thread got
        # delayed, so make a copy (can be removed once the thread is no longer
        # needed in the code - needs pytorch to implement multiple peak mem counters)
        self.backend = self.exp.backend

        # self.exp does it when needed
        #preload_pytorch()

        # initial measurements
        if self.gc_collect: gc.collect()
        self.cpu_mem_used_prev = self.exp.cpu_ram_used()
        if self.backend != 'cpu':
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
        if self.backend != 'cpu':
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

        if self.backend != 'cpu':
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

        if self.backend != 'cpu':
            self.gpu_mem_used_new = self.exp.gpu_ram_used()

            # delta_used is the difference between current used mem and used mem at the start
            self.gpu_mem_used_delta = self.gpu_mem_used_new - self.gpu_mem_used_prev

            # peaked_delta is the overhead if any.
            # 1. The base measurement is the difference between the peak memory
            # and the used mem at the start.
            # 2. Then if used_delta is positive it gets subtracted from the base value.
            self.gpu_mem_peaked_delta = self.gpu_mem_used_peak -  self.gpu_mem_used_prev
            if self.gpu_mem_used_delta > 0: self.gpu_mem_peaked_delta -= self.gpu_mem_used_delta

        if self.compact:
            if 1:
                out  = f"CPU: {b2mb(self.cpu_mem_used_delta):0.0f}/{b2mb(self.cpu_mem_peaked_delta):0.0f}/{b2mb(self.cpu_mem_used_new):0.0f} MB"
            if self.backend != 'cpu':
                out += f" | GPU: {b2mb(self.gpu_mem_used_delta):0.0f}/{b2mb(self.gpu_mem_peaked_delta):0.0f}/{b2mb(self.gpu_mem_used_new):0.0f} MB"
            out += f" | Time {secs2time(self.time_delta)} | (Consumed/Peaked/Used Total)"
            print(out)
        else:
            if 1:
                vals  = [self.cpu_mem_used_delta, self.cpu_mem_peaked_delta, self.cpu_mem_used_new]
            if self.backend != 'cpu':
                vals += [self.gpu_mem_used_delta, self.gpu_mem_peaked_delta, self.gpu_mem_used_new]
            w = int2width(*map(b2mb, vals)) + 1 # some air
            if w < 10: w = 10 # accommodate header width
            pre = '･ '
            print(f"{pre}RAM: {'△Consumed':>{w}} {'△Peaked':>{w}}    {'Used Total':>{w}} | Exec time {secs2time(self.time_delta)}")
            if 1:
                print(f"{pre}CPU: {b2mb(self.cpu_mem_used_delta):{w},.0f} {b2mb(self.cpu_mem_peaked_delta):{w},.0f} {b2mb(self.cpu_mem_used_new):{w},.0f} MB |")
            if self.backend != 'cpu':
                print(f"{pre}GPU: {b2mb(self.gpu_mem_used_delta):{w},.0f} {b2mb(self.gpu_mem_peaked_delta):{w},.0f} {b2mb(self.gpu_mem_used_new):{w},.0f} MB |")

        # for self.data accessor
        self.cpu_mem_used_prev = self.cpu_mem_used_new
        if self.backend != 'cpu':
            self.gpu_mem_used_prev = self.gpu_mem_used_new

        self.data = CellLoggerData(
            CellLoggerMemory(self.cpu_mem_used_delta, self.cpu_mem_peaked_delta, self.cpu_mem_used_prev),
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
