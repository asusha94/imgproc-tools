

#ifndef __IMGPROC__PYTHON__CONVERTER_CV_SIZE_HPP__
#define __IMGPROC__PYTHON__CONVERTER_CV_SIZE_HPP__

#include <opencv2/core.hpp>

#include <pybind11/pybind11.h>

namespace pybind11::detail
{
    template <>
    struct type_caster<cv::Size>
    {
        using value_type = cv::Size;

        PYBIND11_TYPE_CASTER(value_type, _("cvSize"));

        bool load(handle src, bool)
        {
            namespace py = pybind11;

            if (!src) return false;

            auto o = src.ptr();

            if (PyTuple_Check(o))
            {
                auto sz = (int) PyTuple_Size(o);

                if (sz == 2)
                {
                    PyObject* oi;

                    oi = PyTuple_GET_ITEM(o, 0);
                    if (PyLong_Check(oi))
                    {
                        value.width = PyLong_AsLong(oi);

                        oi = PyTuple_GET_ITEM(o, 1);
                        if (PyLong_Check(oi))
                        {
                            value.height = PyLong_AsLong(oi);
                            return true;
                        }
                    }
                }
            }
            else if (PyList_Check(o))
            {
                auto sz = (int) PyList_Size(o);

                if (sz == 2)
                {
                    PyObject* oi;

                    oi = PyList_GET_ITEM(o, 0);
                    if (PyLong_Check(oi))
                    {
                        value.width = PyLong_AsLong(oi);

                        oi = PyList_GET_ITEM(o, 1);
                        if (PyLong_Check(oi))
                        {
                            value.height = PyLong_AsLong(oi);
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        static handle cast(value_type src, return_value_policy /* policy */, handle /* parent */)
        {
            namespace py = pybind11;

            return py::make_tuple(src.width, src.height);
        }
    };
}   // namespace pybind11::detail

#endif
