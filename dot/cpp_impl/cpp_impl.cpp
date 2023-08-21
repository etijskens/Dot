/*
 *  C++ source file for module dot.cpp_impl
 *  File dot/cpp_impl/cpp_impl.cpp
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h> // add support for multi-dimensional arrays
#include <string>

namespace nb = nanobind;

double
dot
  ( nb::ndarray<double> a // in
  , nb::ndarray<double> b // in
  )
{// detect argument errors.
    if( a.ndim() != 1 )
        throw std::domain_error(std::string("Argument 1 is not a 1D-array."));
    if( b.ndim() != 1 )
        throw std::domain_error(std::string("Argument 2 is not a 1D-array."));
    size_t n = a.shape(0);
    if ( n != b.shape(0) )
        throw std::domain_error(std::string("The arguments do not have the same length."));
 // we do not intend to modify a, nor b, hence declare const
    double const * a_ = a.data();
    double const * b_ = b.data();
 // do the actual work
    double d = 0;
    for(size_t i = 0; i < n; ++i) {
        d += a_[i] * b_[i];
    }
    return d;
}


NB_MODULE(cpp_impl, m) {
    m.doc() = "A simple example python extension";

    m.def("dot", &dot, "Dot product of two float 1D arrays.");

    m.def("inspect"
         , [](nb::ndarray<> a)
           {
                printf("Array data pointer : %p\n", a.data());
                printf("Array dimension : %zu\n", a.ndim());
                for (size_t i = 0; i < a.ndim(); ++i) {
                    printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
                    printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
                }
                printf("Device ID = %u (cpu=%i, cuda=%i)\n"
                , a.device_id()
                , int(a.device_type() == nb::device::cpu::value)
                , int(a.device_type() == nb::device::cuda::value)
                );
                printf("Array dtype: int16=%i, uint32=%i, float32=%i, float64=%i\n"
                , a.dtype() == nb::dtype<int16_t>()
                , a.dtype() == nb::dtype<uint32_t>()
                , a.dtype() == nb::dtype<float>()
                , a.dtype() == nb::dtype<double>()
                );
            }
          , "inspect an array"
    );
}