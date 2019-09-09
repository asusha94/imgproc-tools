
#ifndef __IMGPROC__CPP__CV_UTILS_HPP__
#define __IMGPROC__CPP__CV_UTILS_HPP__

#include <opencv2/core.hpp>

namespace cv_utils
{
    namespace _impl
    {
        template <class MatxType>
        struct convertMatToMatx;

        template <class DType, int M, int N>
        struct convertMatToMatx<cv::Matx<DType, M, N>>
        {
            using matx_type = cv::Matx<DType, M, N>;

            static inline void call(cv::InputArray src, matx_type& dst)
            {
                CV_Assert(src.rows() <= M && src.cols() <= N && src.channels() == 1);

                auto src_mat = src.getMat();

                int depth = CV_MAT_DEPTH(src.type());

                switch (depth)
                {
                case CV_8S:
                    call<cv::int8_t>(src_mat, dst);
                    break;
                case CV_8U:
                    call<cv::uint8_t>(src_mat, dst);
                    break;
                case CV_16S:
                    call<cv::int16_t>(src_mat, dst);
                    break;
                case CV_16U:
                    call<cv::uint16_t>(src_mat, dst);
                    break;
                case CV_32S:
                    call<cv::int32_t>(src_mat, dst);
                    break;
                case CV_32F:
                    call<float>(src_mat, dst);
                    break;
                case CV_64F:
                    call<double>(src_mat, dst);
                    break;
                default:
                    throw std::runtime_error("unsupported data type");
                }
            }

            template <typename Element>
            static inline void call(const cv::Mat& src, matx_type& dst)
            {
                for (auto i = 0; i < N; i++)
                {
                    auto row = src.ptr<Element>(i);
                    for (auto j = 0; j < M; j++)
                    {
                        dst(i, j) = row[j];
                    }
                }
            }
        };
    }   // namespace _impl

    template <class MatxType>
    inline void convertMatToMatx(cv::InputArray src, MatxType& dst)
    {
        _impl::convertMatToMatx<MatxType>::call(src, dst);
    }
}   // namespace cv_utils

#endif
