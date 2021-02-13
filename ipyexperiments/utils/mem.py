"Helper utility functions for memory management"

import gc
import os
from collections import namedtuple
from ipyexperiments.utils.pynvml_gate import load_pynvml_env

try:
    import torch # currently relying on pytorch
except Exception as e:
    raise Exception(f"{e}\nYou need to install the torch module; pip install torch")

use_gpu = torch.cuda.is_available()
if not use_gpu:
    raise RuntimeError("these functions require CUDA environment; torch.cuda.is_available() returns false")

pynvml = load_pynvml_env()

GPUMemory = namedtuple('GPUMemory', ['total', 'free', 'used'])

def preload_pytorch(device_id=0):
    """ Do a small operation on CUDA to get the pytorch/cuda structures in place.

    A must to be run first if you're going to compare any CUDA-related numbers.
    """
    if torch.cuda.is_initialized():
        return
    torch.ones((1, 1)).to(device_id)


### Helpers ###

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

def b2mb(num):
    """ convert Bs to MBs and round down """
    return int(num/2**20)


### Get memory stats ###

# for gpu returns GPUMemory(total, free, used)
# for invalid gpu id returns GPUMemory(0, 0, 0)
def gpu_mem_get_mbs(torch_gpu_id=None):
    """ Query nvidia for total, used and free memory for gpu in MBs. if gpu id is not passed, currently selected torch device is used """
    if not use_gpu:
        return GPUMemory(0, 0, 0)
    if torch_gpu_id is None:
        torch_gpu_id = torch.cuda.current_device()
    nvml_gpu_id = get_nvml_gpu_id(torch_gpu_id)
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return GPUMemory(*(map(b2mb, [info.total, info.free, info.used])))
    except:
        return GPUMemory(0, 0, 0)

def gpu_mem_get_total_mbs(torch_gpu_id=None):
    """ Return the amount of total memory (in rounded MBs) """
    return gpu_mem_get_mbs(torch_gpu_id).total

def gpu_mem_get_free_mbs(torch_gpu_id=None):
    """ Return the amount of free memory (in rounded MBs) """
    return gpu_mem_get_mbs(torch_gpu_id).free

def gpu_mem_get_free_no_cache_mbs(torch_gpu_id=None):
    """ Return the amount of free memory after flushing caching (in rounded MBs) """
    gc.collect()
    torch.cuda.empty_cache()
    return gpu_mem_get_free_mbs(torch_gpu_id)

def gpu_mem_get_used_mbs(torch_gpu_id=None):
    """ Return the amount of used memory (in rounded MBs) """
    return gpu_mem_get_mbs(torch_gpu_id).used

def gpu_mem_get_used_no_cache_mbs(torch_gpu_id=None):
    """ Return the amount of used memory after flushing caching (in rounded MBs) """
    gc.collect()
    torch.cuda.empty_cache()
    return gpu_mem_get_used_mbs(torch_gpu_id)


### Do things to the current GPU ###


def gpu_mem_allocate_mbs(n, fatal=False):
    """
    Try to allocate n MBs on the current device.

    Return the variable  holding it on success, None on failure.

    fatal=True will throw an exception on failure to allocate (default is False).
    """
    # don't try to allocate less than 6MB as it'd be imprecise, need to probably switch to bytes allocation
    try:
        return torch.ones((n*2**18)).cuda().contiguous()
    except Exception as e:
        if not fatal: return None
        raise e

def gpu_mem_leave_free_mbs(n):
    """
    Consume whatever memory is needed so that n MBs are left free on the current device.

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
