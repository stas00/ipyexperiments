"Helper utility functions for memory management"

from ipyexperiments.utils.pynvml_gate import load_pynvml_env
import threading, time, gc
from collections import namedtuple

try:
    import torch # currently relying on pytorch
except Exception as e:
    raise Exception(f"{e}\nYou need to install the torch module; pip install torch")

use_gpu = torch.cuda.is_available()
if not use_gpu:
    raise RuntimeError("these functions require CUDA environment; torch.cuda.is_available() returns false")

pynvml = load_pynvml_env()

############# gpu memory helper functions ############

GPUMemory = namedtuple('GPUMemory', ['total', 'free', 'used'])

def preload_pytorch():
    """ Do a small operation on CUDA to get the pytorch/cuda structures in place.

    A must to be run first if you're going to compare any CUDA-related numbers.
    """
    torch.ones((1, 1)).cuda()

preload_pytorch()  # needed to run first to get the measurements right

def b2mb(num):
    """ convert Bs to MBs and round down """
    return int(num/2**20)

def gpu_mem_allocate_mbs(n, fatal=False):
    """ Try to allocate n MBs.

    Return the variable  holding it on success, None on failure.

    fatal=True will throw an exception on failure to allocate (default is False).
    """
    # don't try to allocate less than 6MB as it'd be imprecise, need to probably switch to bytes allocation
    try:
        return torch.ones((n*2**18)).cuda().contiguous()
    except Exception as e:
        if not fatal: return None
        raise e

# for gpu returns GPUMemory(total, free, used)
# for invalid gpu id returns GPUMemory(0, 0, 0)
def gpu_mem_get_mbs(id=None):
    """ Query nvidia for total, used and free memory for gpu in MBs. if id is not passed, currently selected torch device is used """
    if not use_gpu:
        return GPUMemory(0, 0, 0)
    if id is None:
        id = torch.cuda.current_device()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GPUMemory(*(map(b2mb, [info.total, info.free, info.used])))
    except:
        return GPUMemory(0, 0, 0)

def gpu_mem_get_total_mbs():
    """ Return the amount of total memory (in rounded MBs) """
    return gpu_mem_get_mbs().total

def gpu_mem_get_free_mbs():
    """ Return the amount of free memory (in rounded MBs) """
    return gpu_mem_get_mbs().free

def gpu_mem_get_free_no_cache():
    """ Return the amount of free memory after flushing caching (in rounded MBs) """
    gc.collect()
    torch.cuda.empty_cache()
    return gpu_mem_get_free_mbs()

def gpu_mem_get_used_mbs():
    """ Return the amount of used memory (in rounded MBs) """
    return gpu_mem_get_mbs().used

def gpu_mem_get_used_no_cache():
    """ Return the amount of used memory after flushing caching (in rounded MBs) """
    gc.collect()
    torch.cuda.empty_cache()
    return gpu_mem_get_used_mbs()

def gpu_mem_leave_free_mbs(n):
    """Consume whatever memory is needed so that n MBs are left free.

    On success it returns a variable that holds the allocated memory, which
    needs to be kept alive as long as it's needed to hold that memory. Call
    `del` to release the memory when it is no longer needed.

    This function is very useful if the test needs to hit OOM, so this function
    will leave just the requested amount of GPU RAM free, regardless of GPU
    utilization or size of the card.

    """
    avail = gpu_mem_get_free_mbs()
    assert avail > n, f"already have less available mem than desired {n}MBs"
    consume = avail - n
    #print(f"consuming {consume}MB to bring free mem to {n}MBs")
    buf = gpu_mem_allocate_mbs(consume)
    assert buf is not None, f"failed to allocate {consume}MB"
    return buf
