# -*- coding: utf-8 -*-
# File dot/__init__.py
"""
## Python package dot
"""

__version__ = '0.0.0'

def dot(a, b):
    """Compute the dot product of a and b.

    Args:
        a: array of numbers
        b: array of numbers, of same length as a
    Returns:
        a number
    Raises:
        ValueError if len(a) != len(b)
    """
    n = len(a)
    if len(b) != n:
        raise ValueError("Unequal array length.")
    d = 0
    for i in range(n):
        d += a[i] * b[i]
    return d

import numba
import warnings
# CAUTION!
warnings.filterwarnings("ignore", category=numba.NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)

jit_dot = numba.jit(dot)

from dot.cpp_impl import dot as cpp_dot
from dot.f90_impl import dot as f90_dot

import numpy
npy_dot = numpy.dot



