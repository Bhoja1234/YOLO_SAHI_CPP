#pragma once
#include <opencv2/opencv.hpp>

struct PoseKeyPoint {
    float x;
    float y;
    float confidence;
};

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    cv::Mat boxMask; //矩形框内mask，节省内存空间和加快速度
    std::vector<PoseKeyPoint> keyPoints; // 一个box内包含多个关键点
} DL_RESULT;

//// SAHI 滑动切片推理
//typedef struct _SAHIImgData
//{
//    cv::Mat img;
//    int x, y, w, h;
//} SAHIImgData;

extern bool isGPU;