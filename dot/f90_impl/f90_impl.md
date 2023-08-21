This file documents a python module built from Fortran code with f2py.
You should document the Python interfaces, *NOT* the Fortran interfaces.

Module dot.f90_impl
*************************

Module :py:mod:`f90_impl` built from fortran code in dot.f90_impl

.. function:: add(x,y,z)
   :module: dot.f90_impl

   Compute the sum of *x* and *y* and store the result in *z* (overwrite).

   :param x: 1D Numpy array with ``dtype=numpy.float64`` (input)
   :param y: 1D Numpy array with ``dtype=numpy.float64`` (input)
   :param z: 1D Numpy array with ``dtype=numpy.float64`` (output)
