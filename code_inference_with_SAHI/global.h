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
    cv::Mat boxMask; //���ο���mask����ʡ�ڴ�ռ�ͼӿ��ٶ�
    std::vector<PoseKeyPoint> keyPoints; // һ��box�ڰ�������ؼ���
} DL_RESULT;

//// SAHI ������Ƭ����
//typedef struct _SAHIImgData
//{
//    cv::Mat img;
//    int x, y, w, h;
//} SAHIImgData;

extern bool isGPU;