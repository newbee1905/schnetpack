from .compatibility import *
import importlib
import torch
from typing import Type, Union, List

from schnetpack import properties as spk_properties

def as_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string to torch.dtype"""
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[6:]

    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "float":
        return torch.float
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "half":
        return torch.half
    elif dtype_str == "uint8":
        return torch.uint8
    elif dtype_str == "int8":
        return torch.int8
    elif dtype_str == "int16":
        return torch.int16
    elif dtype_str == "short":
        return torch.short
    elif dtype_str == "int32":
        return torch.int32
    elif dtype_str == "int":
        return torch.int
    elif dtype_str == "int64":
        return torch.int64
    elif dtype_str == "long":
        return torch.long
    elif dtype_str == "complex64":
        return torch.complex64
    elif dtype_str == "cfloat":
        return torch.cfloat
    elif dtype_str == "complex128":
        return torch.complex128
    elif dtype_str == "cdouble":
        return torch.cdouble
    elif dtype_str == "quint8":
        return torch.quint8
    elif dtype_str == "qint8":
        return torch.qint8
    elif dtype_str == "qint32":
        return torch.qint32
    elif dtype_str == "bool":
        return torch.bool
    else:
        raise AttributeError(f"Unsupported dtype string: {dtype_str}")


def int2precision(precision: Union[int, torch.dtype]):
    """
    Get torch floating point precision from integer.
    If an instance of torch.dtype is passed, it is returned automatically.

    Args:
        precision (int, torch.dtype): Target precision.

    Returns:
        torch.dtype: Floating point precision.
    """
    if isinstance(precision, torch.dtype):
        return precision
    else:
        try:
            return getattr(torch, f"float{precision}")
        except AttributeError:
            raise AttributeError(f"Unknown float precision {precision}")


def str2class(class_path: str) -> Type:
    """
    Obtain a class type from a string

    Args:
        class_path: module path to class, e.g. ``module.submodule.classname``

    Returns:
        class type
    """
    class_path = class_path.split(".")
    class_name = class_path[-1]
    module_name = ".".join(class_path[:-1])
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls


def required_fields_from_properties(properties: List[str]) -> List[str]:
    """
    Determine required external fields based on the response properties to be computed.

    Args:
        properties (list(str)): List of response properties for which external fields should be determined.

    Returns:
        list(str): List of required external fields.
    """
    required_fields = set()

    for p in properties:
        if p in spk_properties.required_external_fields:
            required_fields.update(spk_properties.required_external_fields[p])

    required_fields = list(required_fields)

    return required_fields
