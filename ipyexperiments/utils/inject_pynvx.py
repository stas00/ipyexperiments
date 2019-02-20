"Monkey patch to use pynvml api on mac os through pynvx"

def get_pynvml():
    import platform
    from collections import namedtuple

    pynvml_i = None

    # monkey patch from https://github.com/fastai/fastai/commit/e6c7cc2001624f9c6e551426c89de9a12fbf4272
    if platform.system() == "Darwin":
        try:
            import pynvx

            GPUMemory = namedtuple('GPUMemory', ['total', 'free', 'used'])
            # missing function
            def cudaDeviceGetHandleByIndex(id): return pynvx.cudaDeviceGetHandles()[id]
            setattr(pynvx, 'cudaDeviceGetHandleByIndex', cudaDeviceGetHandleByIndex)

            # different named and return value needs be a named tuple
            def cudaDeviceGetMemoryInfo(handle):
                info = pynvx.cudaGetMemInfo(handle)
                return GPUMemory(*info)
            setattr(pynvx, 'cudaDeviceGetMemoryInfo', cudaDeviceGetMemoryInfo)

            # remap the other functions
            for m in ['Init', 'DeviceGetCount', 'DeviceGetHandleByIndex', 'DeviceGetMemoryInfo']:
                setattr(pynvx, f'nvml{m}', getattr(pynvx, f'cuda{m}'))

            pynvml_i = pynvx

        except Exception as e:
            raise Exception(f"{e}\nYou need to install the pynvx module; pip install pynvx")
    else:
        try:
            import pynvml
            pynvml_i = pynvml
        except Exception as e:
                raise Exception(f"{e}\nYou need to install the nvidia-ml-py3 module; pip install nvidia-ml-py3")

    return pynvml_i
