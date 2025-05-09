#include "utils.h"

void Check_GPU() {
    // 获取所有的支持设备列表CPU/GPU
    Ort::Env env;
    Ort::Session session(nullptr); // 创建一个空会话
    Ort::SessionOptions sessionOptions{ nullptr };//创建会话配置
    //获取所有的支持设备列表CPU/GPU
    auto providers = Ort::GetAvailableProviders();
    //for (auto& provider : providers) {
    //    std::cout << provider << std::endl;
    //}
    auto cudaAvailable = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
    if (cudaAvailable == providers.end())//没有找到cuda列表
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (cudaAvailable != providers.end())//找到cuda列表
    {
        std::cout << "Inference device: GPU" << std::endl;
        isGPU = true;
    }
    else // 什么也没找到，默认使用CPU
    {
        std::cout << "Inference device: CPU" << std::endl;
    }
}

void Detector(DCSP_CORE*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images_dec";

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );

            }

            cv::namedWindow("Result of Detection", cv::WINDOW_NORMAL);
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

void DetectorWithSAHI(DCSP_CORE*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images_dec";
    SAHI sahi(512, 512, 0.1, 0.1);

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> all_res; // 记录图片的检测结果
            std::vector<cv::Rect> all_regions; // 记录图片的分割区域
            // 小图切割
            all_regions = sahi.sliceImage(img);
            // 小图检测 + 位置还原
            for (auto& region : all_regions) {
                cv::Mat img_slice = img(region);
                std::vector<DL_RESULT> res;
                p->RunSession(img_slice, res);
                sahi.mapToOriginal(res, region);
                all_res.insert(all_res.end(), res.begin(), res.end());
            }
            // 非极大值抑制
            all_res = sahi.NMSResults(all_res);

            for (auto& re : all_res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );

            }

            cv::namedWindow("Result of Detection", cv::WINDOW_NORMAL);
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

void DetectTest(bool use_sahi)
{
    DCSP_CORE* yoloDetector = new DCSP_CORE;
    yoloDetector->classes = { "person" };
    DL_INIT_PARAM params;
    params.ModelPath = "./models/yolov8n.onnx";
    params.rectConfidenceThreshold = 0.5;
    params.iouThreshold = 0.5;
    params.imgSize = { 640, 640 };
    params.modelType = YOLO_DETECT;

    if (isGPU) {
        // GPU FP32 inference
        params.cudaEnable = true;
    }
    else {
        // CPU inference
        params.cudaEnable = false;
    }

    yoloDetector->CreateSession(params);

    if (use_sahi) {
        DetectorWithSAHI(yoloDetector);
    }
    else {
        Detector(yoloDetector);
    }
}

// 可能无法执行分割模型，因为NMS无法对分割模型进行操作
void Segment(DCSP_CORE*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images_dec";

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            cv::Mat mask = img.clone();
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );

                mask(re.box).setTo(color, re.boxMask);

            }
            cv::addWeighted(img, 0.5, mask, 0.5, 0, img);
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

void SegmentTest()
{
    DCSP_CORE* yoloDetector = new DCSP_CORE;
    //ReadCocoYaml(yoloDetector);
    yoloDetector->classes = { "person" };
    DL_INIT_PARAM params;
    params.ModelPath = "./models/yolov8n-seg.onnx";
    params.rectConfidenceThreshold = 0.5;
    params.iouThreshold = 0.5;
    params.imgSize = { 640, 640 };
    params.modelType = YOLO_SEGMENT;

    if (isGPU) {
        // GPU FP32 inference
        params.cudaEnable = true;
    }
    else {
        // CPU inference
        params.cudaEnable = false;
    }
    yoloDetector->CreateSession(params);
    Segment(yoloDetector);
}

