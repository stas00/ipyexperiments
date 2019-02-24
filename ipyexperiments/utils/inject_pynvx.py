""" Transparently load pynvml for Linux/Windows and pynvx for OSX, resulting in the pynvml API subset available on all supported platforms """

import platform

def load_pynvml_env():
    "Imports pynvml bits according to the given platform and init it"
    try:
        import pynvml
    except Exception as e:
        raise Exception(f"{e}\npynvml is required: pip install nvidia-ml-py3")

    # on OSX we use pynvx with pynvml wrapper (still requires pynvml)
    if  platform.system() == "Darwin":
        try:
            from pynvx import pynvml
        except Exception as e:
            raise Exception(f"{e}\npynvx is required; pip install pynvx")

    pynvml.nvmlInit()
    return pynvml
