

#ifndef __IMGPROC__PYTHON__CONVERTER_CV_MAT_HPP__
#define __IMGPROC__PYTHON__CONVERTER_CV_MAT_HPP__

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "./converter_ndarray.hpp"

namespace pybind11::detail
{
    template <>
    struct type_caster<cv::Mat>
    {
        using value_type = cv::Mat;

        PYBIND11_TYPE_CASTER(value_type, _("cvMat"));

        bool load(handle src, bool) { return NDArrayConverter::toMat(src.ptr(), value); }

        static handle cast(const value_type& src, return_value_policy /* policy */, handle /* parent */)
        {
            return handle(NDArrayConverter::toNDArray(src));
        }
    };
}   // namespace pybind11::detail

#endif
