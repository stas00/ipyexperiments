# def to_list(buffer): return list(filter(None, map(str.strip, buffer.splitlines())))

# __all__ = to_list("""
# consume_cpu_ram_128mb
# """)

import numpy as np
import torch, pynvml
import re, gc
from math import isclose

############## ram allocation helpers #################

pynvml.nvmlInit()
id = torch.cuda.current_device()

def gpu_ram_free():
    gc.collect()
    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return int( info.free / 2**20 )

def consume_cpu_ram(n): return np.ones((n, n))
def consume_gpu_ram(n): return torch.ones((n, n)).cuda()
def consume_cpu_ram_128mb():  return consume_cpu_ram(2**12)
def consume_gpu_ram_256mb():  return consume_gpu_ram(2**13)
def consume_gpu_ram_1024mb(): return consume_gpu_ram(2**14)

def consume_gpu_ram_mbs(n, fatal=False):
    " allocate n MBs, return the var holding it on success, None on failure "
    if n < 6: return None # don't try to allocate less than 6MB
    try:
        d = int(2**9*n**0.5)
        return torch.ones((d, d)).cuda().contiguous()
    except Exception as e:
        if not fatal: return None
        raise e

def gpu_ram_leave_free_mbs(n):
    " consume whatever memory is needed so that n MBs are left free "
    avail = gpu_ram_free()
    assert avail > n, f"already have less available mem than desired {n}MBs"
    consume = avail - n
    print(f"consuming {consume}MB to bring free mem to {n}MBs")
    return consume_gpu_ram_mbs(consume, fatal=True)



############## var helpers #################

import inspect
def get_callers_locals():
    """Get the local() in the caller's caller's frame (2 frames up)."""
    frame = inspect.currentframe()
    locals = {}
    try:
        locals = (frame.f_back.f_back.f_locals)
    finally:
        del frame
    return locals

def locals_unset(var_names):
    """ unsetting the used in the cell local vars is useful
    * at the very beginning of the cell, which helps with re-running the same cell over and over again (e.g. if there are heavy GPU RAM vars)
    * at the very end of the cell, not to impact following cells if cells are run in sequences (primarily useful for tests)
    """
    l = get_callers_locals()
    for v in var_names:
        if v in l: del l[v]

def check_defined(var_names):
    " check whether the global var names are defined "
    l = get_callers_locals()
    for v in var_names: assert v in l,     f"var {v} should exist in locals()"

def check_undefined(var_names):
    " check whether the global var names are not defined "
    l = get_callers_locals()
    for v in var_names: assert v not in l, f"var {v} should not exist in locals()"



############## print helpers #################

def print_output(output):
    if output:
        print("Captured output:")
        print("| ".join([f"{'='*60}\n", *output.splitlines(True), f"\n{'='*60}\n"] ))
    else:
        print("No captured output")

############## EXP helpers #################

# --------------------------------------------------------------------- #
# the following functions work with the captured output
# output is captured by `%%capture output` from a cell before

# *** Experiment memory:
# RAM:  Consumed     Reclaimed
# CPU: 255.8 MB 255.8 MB ( 99.99%)
def get_consumed_reclaimed_size(output):
    pat = re.compile('Consumed\s+Reclaimed\nCPU:\s+([\d\.]+)\s+([\d\.]+) MB\s+\((\s*[\d\.]+)%\)', flags=re.MULTILINE)
    match = pat.findall(output)
    if match:
        (consumed_size, reclaimed_size, reclaimed_pct) = map(float, match[0])
        return consumed_size, reclaimed_size, reclaimed_pct
    else:
        raise ValueError(f"failed to match pattern {pat}")

def check_reclaimed(output):
    # basic checks
    to_match = [r'Experiment started', 'Experiment finished', r"x1, x2", r'Current state']
    for s in to_match: assert re.search(s, output), f"expecting string: {s}"

    # consumed/reclaimed checks
    consumed_size, reclaimed_size, reclaimed_pct = get_consumed_reclaimed_size(output)

    # compare: numbers are within 2% equal
    assert isclose(consumed_size, reclaimed_size, rel_tol=0.02), f"Reclaimed all memory: {consumed_size} == {reclaimed_size}"
    reclaimed_expected_pct = 99
    assert reclaimed_pct > reclaimed_expected_pct, f"expecting {reclaimed_expected_pct} reclaimed, got {reclaimed_pct}"

def check_data(output, cpu_data):
    consumed_size, reclaimed_size, reclaimed_pct = get_consumed_reclaimed_size(output)

    # compare with data
    final_consumed_size_stats  = cpu_data.consumed /2**20
    final_reclaimed_size_stats = cpu_data.reclaimed/2**20
    # numbers are within 2% equal
    assert isclose(final_consumed_size_stats,  consumed_size, rel_tol=0.02), f"Consumed {final_consumed_size_stats} vs reclaimed {cpu.consumed} memory"
    assert isclose(final_reclaimed_size_stats, reclaimed_size, rel_tol=0.02), f"Reclaimed {final_reclaimed_size_stats} vs reclaimed {cpu.reclaimed} memory"

############## CL helpers #################

# convert .data outputs to the same dimensions (MBs) as reports
def b2mb(x): print(x); return int(x/2**20)

# --------------------------------------------------------------------- #
# the following functions work with the captured output
# output is captured by `%%capture output` from a cell before

# sample:
# RAM: Consumed Peaked  Used Total | Exec time 0.046s
def check_report_strings(output):
    # basic checks
    to_match = [r'Consumed', 'Peaked']
    for s in to_match: assert re.search(s, output), f"expecting string: {s}"

# sample:
# CPU:      123    321     2159 MB
# GPU:      356    789     2160 MB
def get_sizes(output, type):
    match = re.findall(type + r': +([\d\.]+) +([\d\.]+) +([\d\.]+) MB', output)
    (consumed, peaked, total) = map(float, match[0])
    return consumed, peaked, total

def get_sizes_cpu(output): return get_sizes(output, "CPU")
def get_sizes_gpu(output): return get_sizes(output, "GPU")

# compare reported numbers against expected
def check_match(consumed_reported, peaked_reported,
                consumed_expected, peaked_expected, abs_tol=0):
    assert isclose(consumed_reported, consumed_expected, abs_tol=abs_tol), f"Consumed RAM reported: {consumed_reported} == real: {consumed_expected}"
    assert isclose(peaked_reported,   peaked_expected,   abs_tol=abs_tol), f"Peaked RAM reported: {peaked_reported} == real: {peaked_expected}"

# these functions extract the reported data from the output
def check_report_cpu(output, consumed_expected, peaked_expected, abs_tol=0):
    consumed_reported, peaked_reported, total_reported = get_sizes_cpu(output)
    check_match(consumed_reported, peaked_reported,
                consumed_expected, peaked_expected, abs_tol)

def check_report_gpu(output, consumed_expected, peaked_expected, abs_tol=0):
    consumed_reported, peaked_reported, total_reported = get_sizes_gpu(output)
    check_match(consumed_reported, peaked_reported,
                consumed_expected, peaked_expected, abs_tol)
