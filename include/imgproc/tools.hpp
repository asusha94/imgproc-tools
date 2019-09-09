
#ifndef __IMGPROC__TOOLS_HPP__
#define __IMGPROC__TOOLS_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace imgproc
{
    /**
     * @brief Applies a perspective transformation to an image.
     *
     * @param src input image.
     * @param dst output image that has the size dsize and the same type as src .
     * @param M \f$4\times 4\f$ transformation matrix.
     * @param dsize size of the output image.
     * @param srcCameraMatrix
     */

    void warpPerspective4D(cv::InputArray src,
                           cv::OutputArray dst,
                           cv::InputArray M,
                           cv::Size dsize,
                           cv::InputArray srcCameraMatrix = cv::noArray());
}   // namespace imgproc

#endif
