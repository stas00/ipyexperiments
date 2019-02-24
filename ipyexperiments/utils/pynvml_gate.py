""" Transparently load pynvml for Linux/Windows and pynvx for OSX, resulting in the pynvml API subset available on all supported platforms """

import platform

def load_pynvml_env():
    "Imports pynvml bits according to the given platform and init it"
    import pynvml

    # on OSX we use pynvx with pynvml wrapper (still requires pynvml)
    # currently there is no conda pynvx package, hence the runtime check
    if platform.system() == "Darwin":
        try:
            from pynvx import pynvml
        except Exception as e:
            raise Exception(f"{e}\npynvx is required; pip install pynvx")

    pynvml.nvmlInit()
    return pynvml
