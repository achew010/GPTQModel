from enum import Enum


class Backend(Enum):
    AUTO = 0  # choose the fastest one based on quant model compatibility
    TRITON = 3

def get_backend(backend: str):
    try:
        return Backend[backend]
    except KeyError:
        raise ValueError(f"Invalid Backend str: {backend}")
