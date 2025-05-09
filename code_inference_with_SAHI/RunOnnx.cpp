#include "RunOnnx.h"
#include <regex>

#define benchmark
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#define max(a,b)            (((a) > (b)) ? (a) : (b))

// 向一维向量中注入归一化后的图像数据
void BlobFromImage(cv::Mat& iImg, float* iBlob) {
	int channels = iImg.channels();
	int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;
	//cv::Scalar _mean_ = {iParams.mean[0], iParams.mean[1], iParams.mean[2]};
    //cv::Scalar _std_ = {iParams.std[0], iParams.std[1], iParams.std[2]};

	for (int h = 0; h < imgHeight; h++) {
		for (int w = 0; w < imgWidth; w++) {
			for (int c = 0; c < channels; c++) {
				// 归一化图像数据
				float pix = iImg.at<cv::Vec3b>(h, w)[c];
				pix = pix / 255.0f;
				//pix = (pix - mean[c]) / std[c];
                iBlob[c * imgHeight * imgWidth + h * imgWidth + w] = pix;
			}
		}
	}
}

void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<DL_RESULT>& output, const MaskParams& maskParams,
	int& dx, int& dy) {

	float mask_threshold = maskParams.maskThreshold;  // 0.5
	cv::Size orig_img_shape = maskParams.srcImgShape;  // width: 810, height: 1080

	cv::Mat protos = maskProtos.reshape(0, { maskProtos.size[1],  maskProtos.size[2] * maskProtos.size[3] }); // {1, 32, 160*160}
	cv::Mat matmul_res = (maskProposals * protos).t();  // {1, 32, n} * {n, 32, 160*160} = {1, n, 160*160}
	cv::Mat masks = matmul_res.reshape(output.size(), { maskProtos.size[2], maskProtos.size[3] });  // {n, 160, 160}

	std::vector<cv::Mat> maskChannels;
	split(masks, maskChannels);
	for (int i = 0; i < output.size(); ++i) {
		cv::Mat dest, mask;  // dest {160, 160}
		//sigmoid
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);  

		// dest {160, 160} -> {640, 640}
		resize(dest, dest, cv::Size(maskParams.netWidth, maskParams.netHeight), cv::INTER_NEAREST);
		// 修正desk的填充区域
        dest = dest(cv::Rect(dx, dy, dest.cols - 2* dx, dest.rows - 2* dy));

		cv::resize(dest, mask, orig_img_shape, cv::INTER_NEAREST);
		
		mask = mask(output[i].box) > mask_threshold;
		output[i].boxMask = mask;
	}
}


DCSP_CORE::DCSP_CORE() {}
DCSP_CORE::~DCSP_CORE() {
	delete session;
	for (auto it = inputNodeNames.begin(); it != inputNodeNames.end(); it++) {
		delete[] *it;
	}
	inputNodeNames.clear();

	for (auto it = outputNodeNames.begin(); it != outputNodeNames.end(); it++) {
		delete[] *it;
	}
	outputNodeNames.clear();
}

int DCSP_CORE::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg) {
	if (iImg.channels() == 1) {
		cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
	}
	else
	{
		cv::cvtColor(iImg, oImg, cv::COLOR_BGR2RGB);
	}

	// 等比缩放
	if (iImg.cols >= iImg.rows) {
		resizeScales = iImg.cols / (float)iImgSize.at(0);
		cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
	}
	else
	{
		resizeScales = iImg.rows / (float)iImgSize.at(1);
		cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
	}

	//// 左上角填充
	//cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
	//oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
	//oImg = tempImg;

	// 居中填充
	cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
	dx = (iImgSize.at(0) - oImg.cols) / 2;
    dy = (iImgSize.at(1) - oImg.rows) / 2;
    oImg.copyTo(tempImg(cv::Rect(dx, dy, oImg.cols, oImg.rows)));
    oImg = tempImg;

	return 1;
}

void DCSP_CORE::Initialize(const DL_INIT_PARAM& iParams) {
	// 初始化私有变量
	rectConfidenceThreshold = iParams.rectConfidenceThreshold;
	iouThreshold = iParams.iouThreshold;
	maskThreshold = iParams.maskThreshold;
	imgSize = iParams.imgSize;
	modelType = iParams.modelType;
	//mean = iParams.mean;
    //std = iParams.std;
	cudaEnable = iParams.cudaEnable;
	keyPointsNum = iParams.keyPointsNum;

	// 无监督的内容
	pixel_threshold = iParams.pixel_threshold;
	min_val = iParams.min_val;
	max_val = iParams.max_val;
}

// 创建一个会话，成功则返回1，否则返回0
int DCSP_CORE::CreateSession(DL_INIT_PARAM& iParams){
	// 检测是否包含汉字
	std::regex pattern("[\u4e00-\u9fa5]"); 
	bool result = std::regex_search(iParams.ModelPath, pattern);
	if (result) {
		perror("[DCSP_ONNX]:Model path error.Change your model path without chinese characters.\n");
		return 0;
	}
	
	try {
		Initialize(iParams);
		// 创建环境与会话
		env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
		Ort::SessionOptions sessionOption;
		if (iParams.cudaEnable)
		{
			cudaEnable = iParams.cudaEnable;
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = 0;
			sessionOption.AppendExecutionProvider_CUDA(cudaOption);
		}
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		sessionOption.SetIntraOpNumThreads(iParams.IntraOpNumThreads);
		sessionOption.SetLogSeverityLevel(iParams.LogSeverityLevel);

		// _WIN32
		int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.ModelPath.c_str(), static_cast<int>(iParams.ModelPath.length()), nullptr, 0);
		wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.ModelPath.c_str(), static_cast<int>(iParams.ModelPath.length()), wide_cstr, ModelPathSize);
		wide_cstr[ModelPathSize] = L'\0';
		const wchar_t* modelPath = wide_cstr;

		// 启动会话
		session = new Ort::Session(env, modelPath, sessionOption);
		Ort::AllocatorWithDefaultOptions allocator;

		// 自动获取输入输出节点名
		size_t inputNodesNum = session->GetInputCount();
		for (size_t i = 0; i < inputNodesNum; i++) {
			Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
			char* temp_buf = new char[50];
			strcpy(temp_buf, input_node_name.get());
			inputNodeNames.push_back(temp_buf);
		}
        size_t outputNodesNum = session->GetOutputCount();
		for (size_t i = 0; i < outputNodesNum; i++) {
			Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
			char* temp_buf = new char[10];
			strcpy(temp_buf, output_node_name.get());
			outputNodeNames.push_back(temp_buf);
		}
		options = Ort::RunOptions{ nullptr };
		WarmUpSession();
		return 1;
	}
	catch (const std::exception& e) {
		// 展示错误信息
		const char* str1 = "[DCSP_ONNX]:Create session error.\n";
		const char* str2 = e.what();
		std::string result = std::string(str1) + std::string(str2);
		std::cout << result << std::endl;
		perror("[DCSP_ONNX]:Create session error.\n");
		return 0;
	}
}

// 运行推理，输入图片，将结果存放在给定的地址中
int DCSP_CORE::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult)
{
#ifdef benchmark
	clock_t starttime_1 = clock();
#endif // benchmark

	cv::Mat processedImg;
	// 转换大小并重设色彩空间
	PreProcess(iImg, imgSize, processedImg);
	float* blob = new float[processedImg.total() * 3];
	// 数据注入至blob中
	BlobFromImage(processedImg, blob);
	// 设置输入数据维度
	std::vector<int64_t> inputNodeDims = {1, 3, imgSize.at(0), imgSize.at(1) };
	// 转换为张量并完成推理
	TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
	return 1;
}

int DCSP_CORE::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, float* blob, std::vector<int64_t>& inputNodeDims,
	std::vector<DL_RESULT>& oResult)
{
	// 创建输入张量
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
		blob,
		3 * imgSize.at(0) * imgSize.at(1),
		inputNodeDims.data(),
		inputNodeDims.size());
#ifdef benchmark
	clock_t starttime_2 = clock();
#endif // benchmark

	// 执行推导
	auto outputTensor = session->Run(options, inputNodeNames.data(), 
		&inputTensor, 1, 
		outputNodeNames.data(), outputNodeNames.size());
#ifdef benchmark
	clock_t starttime_3 = clock();
#endif // benchmark

	Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
	auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
	// outputNodeDims存储了输出结果的维度
	// output存储了网络输出结果
	std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
	std::cout << "output format: xW = " << outputNodeDims[1] << "x" << outputNodeDims[2] << std::endl;
	const float* output = outputTensor[0].GetTensorData<float>();
	delete[] blob;
	switch (modelType)
	{
	case ANOMALIB:
	{
		int net_width = outputNodeDims[2];
        int net_height = outputNodeDims[3];
		cv::Mat rawData(net_width, net_height, CV_32F, (void*)output); // 存储输出结果
		
		cv::Mat norm = ((rawData - pixel_threshold) / (max_val - min_val) + 0.5) * 255;
		//norm.convertTo(normUint8, CV_8UC1);

		//转换为伪彩色图
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(rawData, &minVal, &maxVal, &minLoc, &maxLoc);
		cv::Mat normUint8 = ((rawData - minVal) / (maxVal - minVal)) * 255;; //转换为uint8灰度图
		//cv::Mat normUint8 = ((rawData - min_val) / (max_val - min_val)) * 255;; //转换为uint8灰度图
		normUint8.convertTo(normUint8, CV_8UC1);
		cv::Mat colorMap, resize_img;
		cv::applyColorMap(normUint8, colorMap, cv::COLORMAP_JET);
		cv::resize(iImg, resize_img, cv::Point(imgSize[0], imgSize[1]));
		//与原图叠加生成热图
		cv::Mat heatMap;
		cv::addWeighted(colorMap, 0.4, resize_img, 0.6, 0, heatMap);
		//// 生成缺陷区域
		cv::Mat binaryMap;
		cv::threshold(rawData, binaryMap, pixel_threshold, 255, cv::THRESH_BINARY);
		cv::resize(binaryMap, binaryMap, cv::Size(640, 640));
		cv::imshow("Contour_raw", binaryMap);

		//cv::Mat binaryMap1;
		//cv::threshold(norm, binaryMap1, 128, 255, cv::THRESH_BINARY);
		//int kernel_size = 9;
		//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
		//cv::Mat openedMap;
  //      cv::morphologyEx(binaryMap1, openedMap, cv::MORPH_OPEN, kernel);
		//cv::resize(binaryMap1, binaryMap1, cv::Size(640, 640));
		//cv::resize(openedMap, openedMap, cv::Size(640, 640));
		//cv::imshow("Contour_norm", binaryMap1);
  //      cv::imshow("Contour_opened", openedMap);

		//if (binaryMap.type() != CV_8UC1) {
		//	binaryMap.convertTo(binaryMap, CV_8UC1);
		//}
		//std::vector<std::vector<cv::Point>> contours;
		//std::vector<cv::Vec4i> hierarchy;
		//cv::findContours(binaryMap, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		//cv::drawContours(iImg, contours, -1, cv::Scalar(0, 255, 0), 2);

		DL_RESULT result;
		result.classId = 1;
		result.confidence = 1.0f;
		result.boxMask = heatMap;
		oResult.push_back(result);

		break;
	}
	case YOLO_DETECT:
	{
		int strideNum = outputNodeDims[2];//8400  总共可以检出8400个目标，大部分是无效目标
		int signalResultNum = outputNodeDims[1];//84  4个坐标属性，80个类别置信度
		std::vector<int> class_ids;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		cv::Mat rawData(signalResultNum, strideNum, CV_32F, (void*)output); // 存储输出结果
		rawData = rawData.t();

		float* data = (float*)rawData.data;

		for (int i = 0; i < strideNum; ++i) {
			float* classesScores = data + 4;
			cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
			cv::Point class_id;
			double maxClassScore;
			cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
			if (maxClassScore > rectConfidenceThreshold)
			{
				confidences.push_back(maxClassScore);
				class_ids.push_back(class_id.x);
				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];

				int left = MAX(int((x - 0.5 * w + 0.5 - dx) * resizeScales), 0);
				int top = MAX(int((y - 0.5 * h + 0.5 - dy) * resizeScales), 0);
				int width = int(w * resizeScales);
				int height = int(h * resizeScales);

				boxes.push_back(cv::Rect(left, top, width, height));
			}
			data += signalResultNum;
		}

		std::vector<int> nmsResult;
		cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
		for (int i = 0; i < nmsResult.size(); ++i)
		{
			int idx = nmsResult[i];
			DL_RESULT result;
			result.classId = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx];
			oResult.push_back(result);
		}

		break;
	}
	case YOLO_CLS:
	{
		cv::Mat rawData(1, this->classes.size(), CV_32F, (void*)output); // 存储输出结果
		float* data = (float*)rawData.data;

		DL_RESULT result;
		for (int i = 0; i < this->classes.size(); i++) {
			result.classId = i;
			result.confidence = data[i];
			oResult.push_back(result);
		}
        break;
	}
    case YOLO_SEGMENT:
	{
		//输入张量shape
		std::vector<int64_t> boxShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();  // 目标检测分支 1*116*8400
		std::vector<int64_t> maskShape = outputTensor[1].GetTensorTypeAndShapeInfo().GetShape(); // 实例分割分支 1*32*160*160

		//int64_t one_output_length = boxShape[1] * boxShape[2] * boxShape[3];
		int signalResultNum = (int)boxShape[1]; // 116 4+80+32
		int cls_num = signalResultNum - 4 - maskShape[1]; // class num 80 

		cv::Mat rawData(cv::Size((int)boxShape[2], (int)boxShape[1]), CV_32F, (void*)output);  //[bs,116,8400]=>[bs,8400,116]
		rawData = rawData.t();

		float* data = (float*)rawData.data;
		int rows = rawData.rows;
		std::vector<int> class_ids;//结果  id数组
		std::vector<float> confidences;//结果  每个id对应置信度数组
		std::vector<cv::Rect> boxes;//每个  id矩形框
		std::vector<std::vector<float>> picked_proposals;  //rawData[:,:, 5 + _className.size():net_width]===> for mask

		for (int r = 0; r < rows; r++) {
			float* classesScores = data + 4;
			//cv::Mat scores(1, cls_num, CV_32F, classesScores);
			cv::Mat scores(1, this->classes.size(), CV_32F, classesScores);
			cv::Point class_id;
			double maxClassScore;
			cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
			if (maxClassScore >= rectConfidenceThreshold) {
				std::vector<float> temp_proto(data + 4 + cls_num, data + signalResultNum);
				picked_proposals.push_back(temp_proto);
				//rect [x,y,w,h] 在(640,640)的尺度上
				float x = data[0];  //x
				float y = data[1];  //y
				float w = data[2];  //w
				float h = data[3];  //h

				int left = MAX(int((x - 0.5 * w + 0.5 - dx) * resizeScales), 0);
				int top = MAX(int((y - 0.5 * h + 0.5 - dy) * resizeScales), 0);
				int width = int(w * resizeScales);
				int height = int(h * resizeScales);
				//int left = int(x - 0.5 * w);
				//int top = int(y - 0.5 * h);
				//int width = int(w);
				//int height = int(h);

				class_ids.push_back(class_id.x);
				confidences.push_back(maxClassScore);
				boxes.push_back(cv::Rect(left, top, width, height));
			}
			data += signalResultNum;//下一行
		}

		std::vector<int> nms_result;
		cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nms_result);
		std::vector<std::vector<float>> temp_mask_proposals;
		cv::Rect holeImgRect(0, 0, iImg.cols, iImg.rows);
		for (int i = 0; i < nms_result.size(); ++i) {
			int idx = nms_result[i];
			DL_RESULT result;
			result.classId = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[idx]);
			oResult.push_back(result);
		}

		MaskParams mask_params;
		mask_params.srcImgShape = iImg.size();
		mask_params.netHeight = imgSize[1];
		mask_params.netWidth = imgSize[0];
		mask_params.maskThreshold = maskThreshold;

		float* mask_output = outputTensor[1].GetTensorMutableData<float>(); // GetTensorData
		std::vector<int> mask_protos_shape = { 1, (int)maskShape[1], (int)maskShape[2], (int)maskShape[3] }; // {1,32,160,160}
		cv::Mat maskData = cv::Mat(mask_protos_shape, CV_32F, mask_output);

		cv::Mat mask_proposals;
		for (int i = 0; i < temp_mask_proposals.size(); i++) {
			mask_proposals.push_back(cv::Mat(temp_mask_proposals[i]).t());
		}
		GetMask(mask_proposals, maskData, oResult, mask_params, dx, dy);

        break;
	}
	case YOLO_POSE:
	{
		int net_width = outputNodeDims[1];
		int key_point_length = net_width - 5;  // 17 * 5
		if (keyPointsNum * 3 != key_point_length) {
			std::cout << "Pose should be shape [x, y, confidence] with 17-points" << std::endl;
            return -1;
		}
		//int64_t one_output_length = VectorProduct(outputNodeDims) / outputNodeDims[0];
		cv::Mat rawData(net_width, outputNodeDims[2], CV_32F, (void*)output);
		rawData = rawData.t();// 存储输出结果.t();  //[bs,56,8400]=>[bs,8400,56]
		float* pdata = (float*)rawData.data;
		int rows = rawData.rows;
		std::vector<int> class_ids;//结果id数组
		std::vector<float> confidences;//结果每个id对应置信度数组
		std::vector<cv::Rect> boxes;//每个id矩形框
		std::vector<std::vector<PoseKeyPoint>> pose_key_points; //保存kpt

		for (int r = 0; r < rows; ++r) {
			float* max_class_socre = pdata + 4;
			cv::Mat score(1, this->classes.size(), CV_32FC1, max_class_socre);
			cv::Point class_id;
			double maxClassScore;
			cv::minMaxLoc(score, 0, &maxClassScore, 0, &class_id);
			if (maxClassScore >= rectConfidenceThreshold) {

				//rect [x,y,w,h]
				float x = pdata[0];  //x
				float y = pdata[1];  //y
				float w = pdata[2];  //w
				float h = pdata[3];  //h
				int left = MAX(int((x - 0.5 * w + 0.5 - dx) * resizeScales), 0);
				int top = MAX(int((y - 0.5 * h + 0.5 - dy) * resizeScales), 0);
				int width = int(w * resizeScales);
                int height = int(h * resizeScales);
				
				class_ids.push_back(class_id.x);
				confidences.push_back(maxClassScore);
				boxes.push_back(cv::Rect(left, top, width, height));

				std::vector<PoseKeyPoint> temp_kpts;
				for (int kpt = 0; kpt < key_point_length; kpt += 3) {
					PoseKeyPoint temp_kp;
					temp_kp.x = (pdata[5 + kpt] - dx) * resizeScales;
					temp_kp.y = (pdata[6 + kpt] - dy) * resizeScales;
					temp_kp.confidence = pdata[7 + kpt];
					temp_kpts.push_back(temp_kp);
				}
				pose_key_points.push_back(temp_kpts);
			}
			pdata += net_width;//下一行
		}

		std::vector<int> nms_result;
		cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nms_result);
		cv::Rect holeImgRect(0, 0, iImg.cols, iImg.rows);
		for (int i = 0; i < nms_result.size(); ++i) {
			int idx = nms_result[i];
			DL_RESULT result;
			result.classId = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx] & holeImgRect;
			result.keyPoints = pose_key_points[idx];
			oResult.push_back(result);
		}
		break;
	}
	
	default:
		std::cout << "[DCSP_CORE]: " << "Not support model type." << std::endl;
	}

#ifdef benchmark
	clock_t starttime_4 = clock();
	double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
	double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
	double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
	if (cudaEnable)
	{
		std::cout << "[DCSP_CORE(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
	}
	else
	{
		std::cout << "[DCSP_CORE(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
	}
#endif
	return 1;
}

int DCSP_CORE::WarmUpSession()
{
	clock_t starttime_1 = clock();
	cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
	cv::Mat processedImg;
	PreProcess(iImg, imgSize, processedImg);

	float* blob = new float[iImg.total() * 3];
	BlobFromImage(processedImg, blob);
	std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), 
		blob, 
		3 * imgSize.at(0) * imgSize.at(1),
		YOLO_input_node_dims.data(), 
		YOLO_input_node_dims.size());
	auto output_tensors = session->Run(
		options, 
		inputNodeNames.data(), 
		&input_tensor, 
		1, 
		outputNodeNames.data(),
		outputNodeNames.size());
	delete[] blob;
	clock_t starttime_4 = clock();
	double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
	if (cudaEnable)
	{
		std::cout << "[DCSP_CORE(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
	}
	return 1;
}
