#include <imgproc/tools.hpp>

#include "./cv_utils.hpp"
#include "./type_utils.hpp"

namespace imgproc
{
    namespace _impl
    {
        template <class Element>
        static void warpPerspective4D(const cv::Mat& src,
                                      cv::Mat& dst,
                                      const cv::Matx44d& warp_mat,
                                      const cv::Matx44d& src_cam_mat)
        {
            for (auto y = 0; y < dst.rows; y++)
            {
                auto dst_row = dst.ptr<Element>(y);

                for (auto x = 0; x < dst.cols; x++)
                {
                    cv::Vec4d p(x, y, 1, 1);

                    auto p_ = warp_mat * p;
                    if (p_[2] < 1)
                        dst_row[x] = Element();
                    else
                    {
                        p_[0] /= p_[2];
                        p_[1] /= p_[2];

                        p_ = src_cam_mat * p_;

                        int li = p_[0];
                        int ri = li + 1;
                        int ti = p_[1];
                        int bi = ti + 1;

                        Element lt, lb, rt, rb;
                        if (ti >= 0 && ti < src.rows)
                        {
                            auto src_row_ti = src.ptr<Element>(ti);
                            if (li >= 0 && li < src.cols) lt = src_row_ti[li];
                            if (ri >= 0 && ri < src.cols) rt = src_row_ti[ri];
                        }

                        if (bi >= 0 && bi < src.rows)
                        {
                            auto src_row_bi = src.ptr<Element>(bi);
                            if (li >= 0 && li < src.cols) lb = src_row_bi[li];
                            if (ri >= 0 && ri < src.cols) rb = src_row_bi[ri];
                        }

                        auto alpha = p_[0] - li;
                        auto beta  = p_[1] - ti;

                        dst_row[x] =
                            (lt * (1 - alpha) + rt * alpha) * (1 - beta) + (lb * (1 - alpha) + rb * alpha) * beta;
                    }
                }
            }
        };
    }   // namespace _impl

    void warpPerspective4D(cv::InputArray src,
                           cv::OutputArray dst,
                           cv::InputArray M,
                           cv::Size dsize,
                           cv::InputArray srcCameraMatrix /* = cv::noArray() */)
    {
        auto src_mat = src.getMat();

        dst.create(dsize.empty() ? src.size() : dsize, src.type());
        auto dst_mat = dst.getMat();

        auto warp_mat = cv::Matx44d::eye();
        cv_utils::convertMatToMatx(M, warp_mat);

        auto src_cam_mat = cv::Matx44d::eye();
        if (!srcCameraMatrix.empty())
            cv_utils::convertMatToMatx(srcCameraMatrix, src_cam_mat);
        else
        {
            src_cam_mat(0, 0) = src_cam_mat(0, 3) = src.cols() / 2;
            src_cam_mat(1, 1) = src_cam_mat(1, 3) = src.rows() / 2;
        }

        typedef void (*worker_func_ptr_t)(const cv::Mat&, cv::Mat&, const cv::Matx44d&, const cv::Matx44d&);

        worker_func_ptr_t worker_func;
        {
            int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

            switch (depth)
            {
            case CV_8S:
                worker_func = type_utils::index_visit<4>(
                    [](auto i) {
                        using Element = cv::Vec<cv::int8_t, decltype(i)::value>;
                        return &_impl::warpPerspective4D<Element>;
                    },
                    cn);
                break;
            case CV_8U:
                worker_func = type_utils::index_visit<4>(
                    [](auto i) {
                        using Element = cv::Vec<cv::uint8_t, decltype(i)::value>;
                        return &_impl::warpPerspective4D<Element>;
                    },
                    cn);
                break;
            case CV_16S:
                worker_func = type_utils::index_visit<4>(
                    [](auto i) {
                        using Element = cv::Vec<cv::int16_t, decltype(i)::value>;
                        return &_impl::warpPerspective4D<Element>;
                    },
                    cn);
                break;
            case CV_16U:
                worker_func = type_utils::index_visit<4>(
                    [](auto i) {
                        using Element = cv::Vec<cv::uint16_t, decltype(i)::value>;
                        return &_impl::warpPerspective4D<Element>;
                    },
                    cn);
                break;
            case CV_32S:
                worker_func = type_utils::index_visit<4>(
                    [](auto i) {
                        using Element = cv::Vec<cv::int32_t, decltype(i)::value>;
                        return &_impl::warpPerspective4D<Element>;
                    },
                    cn);
                break;
            case CV_32F:
                worker_func = type_utils::index_visit<4>(
                    [](auto i) {
                        using Element = cv::Vec<float, decltype(i)::value>;
                        return &_impl::warpPerspective4D<Element>;
                    },
                    cn);
                break;
            case CV_64F:
                worker_func = type_utils::index_visit<4>(
                    [](auto i) {
                        using Element = cv::Vec<double, decltype(i)::value>;
                        return &_impl::warpPerspective4D<Element>;
                    },
                    cn);
                break;
            default:
                throw std::runtime_error("unsupported data type");
            }
        }

        worker_func(src_mat, dst_mat, warp_mat, src_cam_mat);
    }
}   // namespace imgproc
