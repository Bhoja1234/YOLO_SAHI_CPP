#pragma once
#include <vector>
#include "global.h"
#include <opencv2/opencv.hpp>

class SAHI {
public:
    SAHI(const int slice_width, const int slice_height, const float overlap_width_ratio, const float overlap_height_ratio);
    ~SAHI();

    std::vector<cv::Rect> calculateSliceRegions();

    std::vector<cv::Rect> sliceImage( const cv::Mat& image );

    void mapToOriginal(std::vector<DL_RESULT>& boundingBox, const cv::Rect& cordinate);

    std::vector<DL_RESULT> NMSResults(std::vector<DL_RESULT>& boundingBox);

private:
    int slice_height_, slice_width_;
    int image_height_, image_width_;
    float overlap_height_ratio_;
    float overlap_width_ratio_;

};
