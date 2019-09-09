
#include <pybind11/pybind11.h>

#include <imgproc/tools.hpp>

#include "./converter_cv_mat.hpp"
#include "./converter_cv_size.hpp"

namespace py = pybind11;

namespace
{
    static cv::Mat warp_perspective_4d(const cv::Mat& src,
                                       const cv::Mat& warp_mat,
                                       cv::Size dsize,
                                       const cv::Mat& src_cam_mat = {})
    {
        cv::Mat dst_mat;
        {
            py::gil_scoped_release release;
            imgproc::warpPerspective4D(src, dst_mat, warp_mat, dsize, src_cam_mat);
        }

        return dst_mat;
    }
}   // namespace

PYBIND11_MODULE(MODULE_NAME, m)
{
    NDArrayConverter::init_numpy();

    m.def("warp_perspective_4d",
          &warp_perspective_4d,
          py::arg("src"),
          py::arg("warp_mat"),
          py::arg("dsize"),
          py::arg("src_cam_mat") = nullptr,
          R"(
              Applies a perspective transformation to an image.

              Parameters
              ----------
              src : np.ndarray
                Input image.
              warp_mat : np.ndarray
                Transformation 4x4 matrix.
              dsize : tuple or list
                The size of the output image.
              src_cam_mat : np.ndarray, optional
                Camera 4x4 matrix of the input image.

              Returns
              -------
              dst: np.ndarray
                Output image.
          )");
}
