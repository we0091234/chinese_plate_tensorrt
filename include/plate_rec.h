#ifndef __PLATE_REC__
#define __PLATE_REC__
#include "NvInfer.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
using namespace nvinfer1;

static std::vector<std::string> plate_string={"#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖", \
"闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","学","警","港","澳","挂","使","领","民","航","深", \
"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};

static char *color_list[]={"黑色","蓝色","绿色","白色","黄色"};

class PlateRec
{
    public:
    PlateRec();
    void loadTrtModel(const char *trtmodel,Logger &gLogger);
    void plate_rec_color(cv::Mat &img,cv::Size size,std::string &plate_no,std::string &plate_color);
    ~PlateRec();
    
    private:
     IRuntime* runtime=nullptr;
     ICudaEngine* engine=nullptr;
     IExecutionContext* context=nullptr;
     float * prob_1=nullptr;  //字符识别输出
     float * prob_2=nullptr;  //颜色识别输出
     float *blob=nullptr;    //tensorrt 输入

    int output_size_color=5; //颜色5种
    int time_step = 21;   //时间步长，这里值得是最长的字符长度
    int num_char=78; //字符类别数  车牌78种类别字符
    float mean_value=0.588;  //均值
    float std_value =0.193;   //方差
    int output_size = time_step*num_char;  //字符识别输出大小
    const char* plate_rec_input_name = "images"; //onnx 输入  名字
    const char* plate_rec_out_name_1= "output_1"; //onnx 字符识别分支
    const char* plate_rec_out_name_2= "output_2"; //onnx 颜色识别分支
};


#endif