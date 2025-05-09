#include "SAHI.h"

SAHI::SAHI(const int slice_width, const int slice_height, 
            const float overlap_width_ratio, const float overlap_height_ratio)
    :slice_width_(slice_width), slice_height_(slice_height), 
    overlap_width_ratio_(overlap_width_ratio), overlap_height_ratio_(overlap_height_ratio) {};

SAHI::~SAHI() {};

std::vector<cv::Rect> SAHI::calculateSliceRegions() {
    int step_height = slice_height_ - static_cast<int>(slice_height_ * overlap_height_ratio_);
    int step_width = slice_width_ - static_cast<int>(slice_width_ * overlap_width_ratio_);

    int y_max = image_height_ - slice_height_;
    int x_max = image_width_ - slice_width_;

    // ‘§…Ëvector≥ﬂ¥Á£¨
    int num_rows = (y_max / step_height) + 1;
    int num_cols = (x_max / step_width) + 1;
    std::vector<cv::Rect> regions;
    regions.reserve(num_rows * num_cols);

    //int index = 0;
    for (int y = 0; y < image_height_; y += step_height) {
        for (int x = 0; x < image_width_; x += step_width) {
            int width = slice_width_;
            int height = slice_height_;

            int temp_x = x;
            int temp_y = y;

            if (x + width > image_width_) temp_x -= (x + width) - image_width_;
            if (y + height > image_height_) temp_y -= (y + height) - image_height_;

            regions.emplace_back(cv::Rect(temp_x, temp_y, width, height));
        }
    }
    return regions;
}

std::vector<cv::Rect> SAHI::sliceImage(const cv::Mat& image) {
    image_height_ = image.rows;
    image_width_ = image.cols;

    std::vector<cv::Rect> sliceRegions = calculateSliceRegions();

    return sliceRegions;
}

void SAHI::mapToOriginal(
    std::vector<DL_RESULT>& boundingBox, const cv::Rect& cordinate
) {
    for (auto& box : boundingBox) {
        box.box.x += cordinate.x;
        box.box.y += cordinate.y;
    }
}

std::vector<DL_RESULT> SAHI::NMSResults(std::vector<DL_RESULT>& boundingBox) {
    std::vector<DL_RESULT> results;
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (const auto& box : boundingBox) {
        boxes.push_back(box.box);
        scores.push_back(box.confidence);
    }

    cv::dnn::NMSBoxes(boxes, scores, 0.4, 0.4, indices);
    for (const auto& index : indices) {
        results.push_back(boundingBox[index]);
    }
    return results;
}


