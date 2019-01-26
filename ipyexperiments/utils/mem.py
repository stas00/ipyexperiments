"Helper utility functions for memory management"

import pynvml, threading, time
from collections import namedtuple

have_cuda = 0
try:
    # currently relying on pytorch
    import torch
    pynvml.nvmlInit()
    have_cuda = 1
except:
    raise RuntimeError("these functions require NVIDIA environment, check the output of `nvidia-smi`") from None

############# gpu memory helper functions ############

GPUMemory = namedtuple('GPUMemory', ['total', 'used', 'free'])

def preload_pytorch():
    torch.ones((1, 1)).cuda()

preload_pytorch() # needed to run first to get the measurements right

def gpu_mem_allocate_mbs(n):
    " allocate n MBs, return the var holding it on success, None on failure "
    try:
        d = int(2**9*n**0.5)
        return torch.ones((d, d)).cuda().contiguous()
    except:
        return None

def b2mb(num):
    """ convert Bs to MBs and round down """
    return int(num/2**20)

# for gpu returns GPUMemory(total, used, free)
# for cpu returns GPUMemory(0, 0, 0)
# for invalid gpu id returns GPUMemory(0, 0, 0)
def gpu_mem_get_mbs(id=None):
    "query nvidia for total, used and free memory for gpu in MBs. if id is not passed, currently selected torch device is used"
    if not have_cuda: return GPUMemory(0, 0, 0)
    if id is None: id = torch.cuda.current_device()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GPUMemory(*(map(b2mb, [info.total, info.used, info.free])))
    except:
        return GPUMemory(0, 0, 0)

def gpu_mem_get_free_mbs():  return gpu_mem_get_mbs().free
def gpu_mem_get_used_mbs():  return gpu_mem_get_mbs().used
def gpu_mem_get_total_mbs(): return gpu_mem_get_mbs().total

def gpu_mem_leave_free_mbs(n):
    """Consume whatever memory is needed so that n MBs are left free.

    On success it returns a variable that holds the allocated memory, which
    needs to be kept alive as long as it's needed to hold that memory. Call
    `del` to release the memory when that memory is no longer needed.

    This is very useful if the test needs to hit OOM, so this function will
    leave just the requested amount of GPU RAM free, regardless of GPU
    utilization or size of the card.

    """
    avail = gpu_mem_get_free_mbs()
    assert avail > n, f"already have less available mem than desired {n}MBs"
    consume = avail - n
    #print(f"consuming {consume}MB to bring free mem to {n}MBs")
    buf = gpu_mem_allocate_mbs(consume)
    assert buf is not None, f"failed to allocate {consume}MB"
    return buf
