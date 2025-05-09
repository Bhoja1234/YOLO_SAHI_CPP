# Reference_With_SAHI
onnx model inference using SAHI
基于Onnxruntime的YOLO模型部署，添加了SAHI推理
只支持检测模型

## 环境
- Opencv 4.x
- Onnxruntime 1.9
- CUDA 12.5
- OS: windows11 x64（x86不支持CUDA）

## 使用说明
```cpp
int main()
{
    // 检查GPU
    Check_GPU();

    bool use_sahi = true;
    DetectTest(use_sahi);

    return 0;
}
```  
 `use_sahi`用来控制是否开启SAHI，SAHI的具体参数设置要在utils.cpp文件中`DetectorWithSAHI()`函数设置。  
 模型设置请参考：  
 [yolo-onnxruntime-cpp](https://github.com/Bhoja1234/yolo-onnxruntime-cpp "title")

## 代码参考
[https://github.com/MarkusKinn/SAHI](https://github.com/MarkusKinn/SAHI)  
[https://github.com/obss/sahi](https://github.com/obss/sahi) 
