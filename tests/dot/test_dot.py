# -*- coding: utf-8 -*-
# File tests/dot/test_dot.py
"""Tests for dot package."""

import random
import sys

import pytest

sys.path.insert(0,'.')

from dot import *

def test_dot_aa():
    a = [1, 2, 3]
    expected = 1 + 2*2 + 3*3
    result = dot(a,a)
    assert result == expected


def test_dot_commutative():
    # Fix the seed for the random number generator of module random.
    random.seed(0)
    # repeat the test 1000 times:
    for _ in range(1000):
        # choose a random array size
        n = random.randint(0,20)
        # generate two random arrays
        a = [random.random() for i in range(n)]
        b = [random.random() for i in range(n)]
        # test commutativity:
        ab = dot(a,b)
        ba = dot(b,a)
        assert ab == ba


def test_a_zero():
    # Fix the seed for the random number generator of module random.
    random.seed(0)
    # repeat the test 1000 times:
    for _ in range(1000):
        # choose a random array size
        n = random.randint(0,20)
        # generate two random arrays
        a = [random.random() for i in range(n)]
        zero = [0 for i in range(n)]
        # test commutativity:
        a_dot_zero = dot(a,zero)
        assert a_dot_zero == 0
        zero_dot_a = dot(zero,a)
        assert zero_dot_a == 0


def test_a_one():
    # Fix the seed for the random number generator of module random.
    random.seed(0)
    # repeat the test 1000 times:
    for _ in range(1000):
        # choose a random array size
        n = random.randint(0,20)
        # generate two random arrays
        one = [1 for i in range(n)]
        a = [random.random() for i in range(n)]
        sum_a = sum(a)
        # test
        a_dot_one = dot(a,one)
        assert a_dot_one == sum_a
        one_dot_a = dot(one,a)
        assert one_dot_a == sum_a


def test_different_length():
    a = [1, 2]
    b = [1, 1, 1]
    with pytest.raises(ValueError):
        ab = dot(a, b)
    with pytest.raises(ValueError):
        ba = dot(b, a)

from time import perf_counter
def test_time():
    print(f'\ntest_time()')
    n_repetitions = 10
    print(f'{n_repetitions=}')
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        a = [random.random() for i in range(n)]
        b = [random.random() for i in range(n)]

        t0 = perf_counter()
        for _ in range(n_repetitions):
            a_dot_b = dot(a, b)
        seconds = (perf_counter() - t0) / n_repetitions
        print(f'{n=} dot(a, b) took {seconds}s')

import array
import numpy as np
def test_time_2():
    print(f'\ntest_time_types()')
    n_repetitions = 10
    print(f'{n_repetitions=}')
    array_types = [list, array.array, np.array]
    for array_type in array_types:
        print(f'{array_type=}')
        for n in [1_000, 10_000, 100_000, 1_000_000]:
            a = [random.random() for i in range(n)]
            b = [random.random() for i in range(n)]
            if array_type is list:
                pass
            elif array_type is array.array:
                a = array_type('d', a)
                b = array_type('d', b)
            elif array_type is np.array:
                a = array_type(a)
                b = array_type(b)

            t0 = perf_counter()
            for _ in range(n_repetitions):
                a_dot_b = dot(a, b)
            seconds = (perf_counter() - t0) / n_repetitions
            print(f'{n=} dot(a, b) took {seconds}s')

def test_time_numba():
    print(f'\ntest_time_numba()')
    n_repetitions = 10
    print(f'{n_repetitions=}')
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        # a = [random.random() for i in range(n)]
        # b = [random.random() for i in range(n)]
        a = np.random.random(n)
        b = np.random.random(n)
        t0 = perf_counter()
        for _ in range(n_repetitions):
            a_dot_b = jit_dot(a, b)
        seconds = (perf_counter() - t0) / n_repetitions
        print(f'{n=} dot(a, b) took {seconds}s')


def test_time_cpp():
    print(f'\ntest_time_cpp()')
    n_repetitions = 10
    print(f'{n_repetitions=}')
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        # a = [random.random() for i in range(n)]
        # b = [random.random() for i in range(n)]
        a = np.random.random(n)
        b = np.random.random(n)
        t0 = perf_counter()
        for _ in range(n_repetitions):
            a_dot_b = cpp_dot(a, b)
        seconds = (perf_counter() - t0) / n_repetitions
        print(f'{n=} dot(a, b) took {seconds}s')


def test_time_f90():
    print(f'\ntest_time_f90()')
    n_repetitions = 10
    print(f'{n_repetitions=}')
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        # a = [random.random() for i in range(n)]
        # b = [random.random() for i in range(n)]
        a = np.random.random(n)
        b = np.random.random(n)
        t0 = perf_counter()
        for _ in range(n_repetitions):
            a_dot_b = f90_dot(a, b)
        seconds = (perf_counter() - t0) / n_repetitions
        print(f'{n=} dot(a, b) took {seconds}s')


def test_time_numpy():
    print(f'\ntest_time_numpy()')
    n_repetitions = 10
    print(f'{n_repetitions=}')
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        # a = [random.random() for i in range(n)]
        # b = [random.random() for i in range(n)]
        a = np.random.random(n)
        b = np.random.random(n)
        t0 = perf_counter()
        for _ in range(n_repetitions):
            a_dot_b = np.dot(a, b)
        seconds = (perf_counter() - t0) / n_repetitions
        print(f'{n=} dot(a, b) took {seconds}s')


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_dot_aa

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print('-*# finished #*-')

# eof
