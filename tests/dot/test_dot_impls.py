# -*- coding: utf-8 -*-
# File tests/dot/test_dot.py
"""Tests for dot package."""
import random
import sys


import pytest
import numpy as np

sys.path.insert(0,'.')

from dot import *

IMPLEMENTATIONS = [
    dot     # the initial Python implementation
  , jit_dot # numba.jit(dot)
  , cpp_dot # the C++ implementation
  , f90_dot # the Modern Fortran implementation
  , np.dot  # the Numpy implementation
]

IMPLEMENTATION_DESCRIPTIONS = {
    dot    : 'python dot implementation',
    jit_dot: 'numba.jit() dot implementation',
    cpp_dot: 'C++ dot implementation',
    f90_dot: 'f90 dot implementation',
    npy_dot: 'numpy.dot implementation',
}
# Print the descriptions of the implementations for ease of reference.
# In case of failure, the value of dot_impl will be reported, but, although unique,
# it is not extremely informative (it is basically a function pointer).
print("\nImplementation descriptions:")
for dot_impl, desc in IMPLEMENTATION_DESCRIPTIONS.items():
    print(f'{str(dot_impl) : <44} {desc:>30}')
print()
ARRAY_TYPES = (np.ndarray, list)
IMPLEMENTATION_ARRAY_TYPES = {
    dot     : ARRAY_TYPES
  , jit_dot : ARRAY_TYPES
  , cpp_dot : (np.ndarray,)
  , f90_dot : ARRAY_TYPES
  , np.dot  : ARRAY_TYPES
}

VERBOSE = True
N_TEST_REPETITIONS = 1000
N_TIME_REPETITIONS = 10

def test_dot_commutative():
    """The test driver for commutativity of dot product implementations"""

    # a locally defined method
    def assert_dot_commutative(dot_impl, a, b):
        """The test itself"""
        ab = dot_impl(a, b)
        ba = dot_impl(b, a)
        assert ab == ba

    # do the test for all implementations in and array pairs generated by _random_array_pair_generator
    _test(assert_dot_commutative
         , implementations=IMPLEMENTATIONS
         , array_pair_generator=_random_array_pair_generator(n=2)
         )

def test_time_dot():
    # `_time` seems like a superfluous method, but it allows to set up different
    # timings for different `_time_array_pair_generator`s.
    _time( implementations=IMPLEMENTATIONS
         , array_pair_generator=_time_array_pair_generator(10,40,2)
         , n_iter=-400
         )

################################################################################
# Helper functions
################################################################################
from time import perf_counter
def _time(implementations: list, array_pair_generator, n_iter=10):
    """Time the functions in list `implementations`.

    Args:
        implementations: list of dot product implementations
        array_pair_generator: generates pairs of array for which the dot product will be computed
        n_iter: if positive, the number the dot product is computed to provide a meaningful timing.
            if negative, -n_iter is supposed a multiple of the maximum array length and  n x array size
            is supposed to be constant.
    """
    for a, b in array_pair_generator:
        for type_a, type_b in _array_pair_type_generator(ARRAY_TYPES):
            for dot_impl in implementations:
                array_types = IMPLEMENTATION_ARRAY_TYPES[dot_impl]
                if  type_a in array_types \
                and type_b in array_types:
                    aa = a if type_a is np.ndarray else type_a(a)
                    bb = b if type_b is np.ndarray else type_b(b)
                    t0 = perf_counter()
                    if n_iter >= 0:
                        n = n_iter
                    else:
                        n = -n_iter // len(a)
                        if n < 2:
                            raise ValueError
                    for _ in range(n):
                        dot_impl(a,b)
                    t = (perf_counter() - t0) / n
                    print(f'{dot_impl=} ({IMPLEMENTATION_DESCRIPTIONS[dot_impl]})\n  ( a={type_a}\n  , b={type_b}\n  , size={len(a)}\n  , repeat={n}) took {t:>10.3}s')

def _test(test_function, implementations, array_pair_generator):
    if VERBOSE:
        print(f'{test_function=}')
    for a, b in array_pair_generator:
        for type_a, type_b in _array_pair_type_generator(ARRAY_TYPES):
            for dot_impl in IMPLEMENTATIONS:
                array_types = IMPLEMENTATION_ARRAY_TYPES[dot_impl]
                if  type_a in array_types \
                and type_b in array_types:
                    aa = a if type_a is np.ndarray else type_a(a)
                    bb = b if type_b is np.ndarray else type_b(b)
                    if VERBOSE:
                        print(f'{dot_impl=} ({IMPLEMENTATION_DESCRIPTIONS[dot_impl]}) (a={type_a}, b={type_b}')
                    dot_impl(aa, bb)

def _array_pair_type_generator(array_types):
    """generator function that yields every possible type pairs from a list of types `array_types`"""
    for type_a in array_types:
        for type_b in array_types:
            yield (type_a, type_b)


def _random_array_pair_generator(
      n: int = 1
    , same_length : bool = True
    , min_length : int = 1
    , max_length : int = 20
):
    """Generator function for generating `n` pairs of arrays with random values, and random lengths

    Args:
        n: number of pairs to be generated
        same_length: if True, the arrays in a pair have the same length
        min_length: minimum length of the arrays
        max_length: maximum length of the arrays
    """
    j = -1
    for i in range(n):
        # generate two random numpy arrays
        a_length = random.randint(min_length, max_length)
        a = np.random.random(a_length)
        if same_length:
            b = np.random.random(a_length)
        else:
            b_length = random.randint(min_length, max_length)
            # if b_length happens to be equal to a_length generate new b_length:
            while b_length == a_length:
                b_length = random.randint(min_length, max_length)
            b = np.random.random(b_length)

        yield (a, b)


def _time_array_pair_generator(
      min_length : int = 100
    , max_length : int = 1_000_000
    , length_factor: int = 100
):
    """Generator function for generating pairs of arrays with random values for timings. Arrays will
    grow in length from `min_length` to `max_length` by a factor `length_factor`.

    Args:
        min_length: minimum length of the arrays
        length_factor: increase the array length by this factor each time
        max_length: maximum length of the arrays
    """
    m = min_length
    while m <= max_length:
        a = np.random.random(m)
        b = np.random.random(m)
        yield (a, b)

        m *= length_factor


# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_time_dot
    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print('-*# finished #*-')
