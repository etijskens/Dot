#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for f90 module `dot.f90_impl`."""

import sys
sys.path.insert(0,'.')

import numpy as np
import dot


def test_f90_dot_one():
    a = np.array([0,1,2,3,4],dtype=float)
    b = np.ones(a.shape,dtype=float)
    expected = 10.
    d = dot.f90_dot(a, b)
    assert d == expected

#===============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
#===============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_f90_dot_one

    print(f"__main__ running {the_test_you_want_to_debug} ...")
    the_test_you_want_to_debug()
    print('-*# finished #*-')
#===============================================================================
