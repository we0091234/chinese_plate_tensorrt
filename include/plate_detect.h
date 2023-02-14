#ifndef __PLATE_DETECT__
#define __PLATE_DETECT__
#include "logging.h"
#include "utils.hpp"
#include "NvInfer.h"
#include <iostream>
#include "utils.hpp"
#include <fstream>
#include "preprocess.h"
#include "postprocess.h"

#define MAX_IMAGE_INPUT_SIZE_THRESH 5000 * 5000
#define MAX_OBJECTS 1024
#define NUM_BOX_ELEMENT 15
#define NUM_CLASSES 2

using namespace nvinfer1;

struct affine_matrix
{
    float i2d[6];
    float d2i[6];
};

struct bbox 
{
     float x1,y1,x2,y2;
     float landmarks[8];  //关键点4个
     float score;
     int label;
};

class PlateDetect
{
    public:
    PlateDetect();
    void loadTrtModel(const char *trtmodel,Logger &gLogger);
    void detect(cv::Mat &img,std::vector<bbox> &bboxes,float prob_threshold = 0.25f, float nms_threshold = 0.45f);
    ~PlateDetect();

    private:
     IRuntime* runtime=nullptr;
     ICudaEngine* engine=nullptr;
     IExecutionContext* context=nullptr;
     float * prob=nullptr;  //trt输出 
     void *buffers[2]; 
     int output_size ;   //trt输出大小 
     int output_candidates; //多少行 640输入是 25200

     const char* input_blob_name = "input"; //onnx 输入  名字
     const char* output_blob_name = "output"; //onnx 输出 名字

     int input_w = 640;
     int input_h = 640;

    cudaStream_t stream;

    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    float *affine_matrix_host=nullptr;
    float *affine_matrix_device=nullptr;
    float *decode_ptr_host=nullptr;
    float *decode_ptr_device=nullptr;
     
     
  


};
#endif