#pragma once
#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <numeric>

#include "global.h"

// WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>


enum MODEL_TYPE
{
    //FLOAT32 MODEL
    YOLO_DETECT = 1,
    YOLO_SEGMENT = 2,
    YOLO_CLS = 3,
    YOLO_POSE = 4,
    ANOMALIB = 5,
};

typedef struct _DL_INIT_PARAM {
    std::string ModelPath;
    MODEL_TYPE modelType = YOLO_DETECT;
    std::vector<int> imgSize = { 640, 640 };
    std::array<float, 3> mean = { 0.0, 0.0, 0.0 }; // 均值
    std::array<float, 3> std = { 1.0, 1.0, 1.0 }; // 标准差

    float rectConfidenceThreshold = 0.6;  // 置信度得分
    float iouThreshold = 0.5;  // IOU阈值
    float maskThreshold = 0.5;  // mask阈值
    bool cudaEnable = false;
    int LogSeverityLevel = 3;
    int IntraOpNumThreads = 1;
    int keyPointsNum = 17; // 关键点个数
    // 无监督的内容
    float pixel_threshold = 0.5;
    float min_val = 0.0;
    float max_val = 1.0;
} DL_INIT_PARAM;

struct MaskParams {
    //int segChannels = 32;
    //int segWidth = 160;
    //int segHeight = 160;
    int netWidth = 640;
    int netHeight = 640;
    float maskThreshold = 0.5;
    cv::Size srcImgShape;
    cv::Vec4d params = { 1,1,0,0 };
};


class DCSP_CORE {
public:
    DCSP_CORE();

    ~DCSP_CORE();

public:
    void Initialize(const DL_INIT_PARAM& iParams);
    int CreateSession(DL_INIT_PARAM& iParams);
    int RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);
    int WarmUpSession();
    int TensorProcess(clock_t& starttime_1, cv::Mat& iImg, float* blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);
    int PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

    std::vector<std::string> classes{};
    

private:
    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float maskThreshold;
    int keyPointsNum;
    // 无监督的内容
    float pixel_threshold = 0.5;
    float min_val = 0.0f;
    float max_val = 1.0f;

    float resizeScales;//缩放系数
    int dx=0; // x轴偏移
    int dy=0; // y轴偏移

    //std::array<float, 3> mean;
    //std::array<float, 3> std;

    template <typename T>
    T VectorProduct(const std::vector<T>& v)
    {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    };
};

