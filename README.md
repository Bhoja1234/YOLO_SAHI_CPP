# Reference_With_SAHI
onnx model inference using SAHI
基于Onnxruntime的YOLO模型部署，添加了SAHI推理
只支持检测模型

## 环境
- Opencv 4.x
- Onnxruntime 1.9
- CUDA 12.5
- OS: windows11 x64（x86不支持CUDA）

## 心得
如果图片中有大目标的时候（比如人物特写的照片），使用SAHI会导致检测效果割裂。对于中等大小的物体，会因为分割图片而导致该物体被若干个不相交的矩形框给框住。  
只有当使用场景全部是小目标的时候，使用SAHI会获得非常好的效果。  
SAHI好像只支持检测模型，对于分割模型的Mask图不太好操作

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
