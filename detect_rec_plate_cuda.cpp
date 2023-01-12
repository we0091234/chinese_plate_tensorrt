#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "include/utils.hpp"
#include "preprocess.h"
#include "postprocess.h"
#define MAX_IMAGE_INPUT_SIZE_THRESH 5000 * 5000
#define MAX_OBJECTS 1024
#define NUM_BOX_ELEMENT 15
struct affine_matrix
{
    float i2d[6];
    float d2i[6];
};

struct bbox 
{
     float x1,y1,x2,y2;
     float landmarks[8];
     float score;
     int label;
};

void  get_d2i_matrix(affine_matrix &afmt,cv::Size to,cv::Size from)
{
    float scale = std::min(to.width/float(from.width),to.height/float(from.height));
    afmt.i2d[0]=scale;
    afmt.i2d[1]=0;
    afmt.i2d[2]=-scale*from.width*0.5+to.width*0.5;
    afmt.i2d[3]=0;
    afmt.i2d[4]=scale;
    afmt.i2d[5]=-scale*from.height*0.5+to.height*0.5;

    cv::Mat mat_i2d(2,3,CV_32F,afmt.i2d);
    cv::Mat mat_d2i(2,3,CV_32F,afmt.d2i);
    cv::invertAffineTransform(mat_i2d,mat_d2i);
    memcpy(afmt.d2i,mat_d2i.ptr<float>(0),sizeof(afmt.d2i));
}


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
const std::vector<std::string> plate_string={"#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖", \
"闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","学","警","港","澳","挂","使","领","民","航","深", \
"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};

const std::vector<std::string> plate_string_yinwen={"#","<beijing>","<hu>","<tianjin>","<chongqing>","<hebei>","<jing>","<meng>","<liao>","<jilin>","<hei>","<su>","<zhe>","<wan>", \
"<fujian>","<gan>","<lun>","<henan>","<hubei>","<hunan>","<yue>","<guangxi>","<qiong>","<chuan>","<guizhou>","<yun>","<zang>","<shanxi>","<gan>","<qinghai>",\
"<ning>","<xin>","<xue>","<police>","<hongkang>","<Macao>","<gua>","<shi>","<ling>","<min>","<hang>","<shen>", \
"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 2;  //单层车牌，双层车牌两类


const char* INPUT_BLOB_NAME = "input"; //onnx 输入  名字
const char* OUTPUT_BLOB_NAME = "output"; //onnx 输出 名字
static Logger gLogger;









void blobFromImage_plate(cv::Mat& img,float mean_value,float std_value,float *blob)
{
    // float* blob = new float[img.total()*3];
    // int channels = NUM_CLASSES;
    int img_h = img.rows;
    int img_w = img.cols;
    int k = 0;
    for (size_t c = 0; c <3; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                    blob[k++] =
                    ((float)img.at<cv::Vec3b>(h, w)[c]/255.0-mean_value)/std_value;
            }
        }
    }
    // return blob;
}





float getNorm2(float x,float y)
{
    return sqrt(x*x+y*y);
}

cv::Mat getTransForm(cv::Mat &src_img, cv::Point2f  order_rect[4]) //透视变换
{
      cv::Point2f w1=order_rect[0]-order_rect[1];
            cv::Point2f w2=order_rect[2]-order_rect[3];
            auto width1 = getNorm2(w1.x,w1.y);
            auto width2 = getNorm2(w2.x,w2.y);
            auto maxWidth = std::max(width1,width2);

            cv::Point2f h1=order_rect[0]-order_rect[3];
            cv::Point2f h2=order_rect[1]-order_rect[2];
            auto height1 = getNorm2(h1.x,h1.y);
            auto height2 = getNorm2(h2.x,h2.y);
            auto maxHeight = std::max(height1,height2);
            //  透视变换
            std::vector<cv::Point2f> pts_ori(4);
            std::vector<cv::Point2f> pts_std(4);

            pts_ori[0]=order_rect[0];
            pts_ori[1]=order_rect[1];
            pts_ori[2]=order_rect[2];
            pts_ori[3]=order_rect[3];

            pts_std[0]=cv::Point2f(0,0);
            pts_std[1]=cv::Point2f(maxWidth,0);
            pts_std[2]=cv::Point2f(maxWidth,maxHeight);
            pts_std[3]=cv::Point2f(0,maxHeight);

            cv::Mat M = cv::getPerspectiveTransform(pts_ori,pts_std);
            cv:: Mat dstimg;
            cv::warpPerspective(src_img,dstimg,M,cv::Size(maxWidth,maxHeight));
            return dstimg;
}
 
cv::Mat get_split_merge(cv::Mat &img)   //双层车牌 分割 拼接
{
    cv::Rect  upper_rect_area = cv::Rect(0,0,img.cols,int(5.0/12*img.rows));
    cv::Rect  lower_rect_area = cv::Rect(0,int(1.0/3*img.rows),img.cols,img.rows-int(1.0/3*img.rows));
    cv::Mat img_upper = img(upper_rect_area);
    cv::Mat img_lower =img(lower_rect_area);
    cv::resize(img_upper,img_upper,img_lower.size());
    cv::Mat out(img_lower.rows,img_lower.cols+img_upper.cols, CV_8UC3, cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0,0,img_upper.cols,img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols,0,img_lower.cols,img_lower.rows)));

    return out;
}


std::string decode_outputs(float *prob,int output_size)
{
    std::string plate ="";
    std::string pre_str ="#";
    for (int i = 0; i<output_size; i++)
    {
       int  index = int(prob[i]);
        if (plate_string[index]!="#" && plate_string[index]!=pre_str)
            plate+=plate_string[index];
        pre_str = plate_string[index];
        
    }
    return plate;
}

std::string decode_outputs_pingyin(float *prob,int output_size) //拼音
{
    std::string plate ="";
    std::string pre_str ="#";
    for (int i = 0; i<output_size; i++)
    {
       int  index = int(prob[i]);
        if (plate_string_yinwen[index]!="#" && plate_string_yinwen[index]!=pre_str)
            plate+=plate_string_yinwen[index];
        pre_str = plate_string_yinwen[index];
        
    }
    return plate;
}


void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape,const char *INPUT_BLOB_NAME,const char *OUTPUT_BLOB_NAME) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    // context.enqueue(1, buffers, stream, nullptr);
    context.enqueueV2( buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}



int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    char *trtModelStreamDet{nullptr};
    char *trtModelStreamRec{nullptr};
    size_t size{0};
    size_t size_rec{0};
    // argv[1]="/mnt/Gu/xiaolei/cplusplus/trt_project/chinese_plate_recoginition/build/plate_detect.trt"; 
    // argv[2]="/mnt/Gu/xiaolei/cplusplus/trt_project/chinese_plate_recoginition/build/plate_rec.trt";
    // argv[3]="/mnt/Gu/xiaolei/cplusplus/trt_project/chinese_plate_recoginition/test_imgs/single_blue.jpg";
    // argv[4]="output.jpg";

    const std::string engine_file_path {argv[1]};  
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStreamDet = new char[size];
        assert(trtModelStreamDet);
        file.read(trtModelStreamDet, size);
        file.close();
    }

    const std::string engine_file_path_rec {argv[2]};
    std::ifstream file_rec(engine_file_path_rec, std::ios::binary);
    if (file_rec.good()) {
        file_rec.seekg(0, file_rec.end);
        size_rec = file_rec.tellg();
        file_rec.seekg(0, file_rec.beg);
        trtModelStreamRec = new char[size_rec];
        assert(trtModelStreamRec);
        file_rec.read(trtModelStreamRec, size_rec);
        file_rec.close();
    }

    //det模型trt初始化
    IRuntime* runtime_det = createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    ICudaEngine* engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
    assert(engine_det != nullptr); 
    IExecutionContext* context_det = engine_det->createExecutionContext();
    assert(context_det != nullptr);
    delete[] trtModelStreamDet;

    //rec模型trt初始化
    IRuntime* runtime_rec = createInferRuntime(gLogger);
    assert(runtime_rec!= nullptr);
    ICudaEngine* engine_rec = runtime_rec->deserializeCudaEngine(trtModelStreamRec, size_rec);
    assert(engine_rec != nullptr); 
    IExecutionContext* context_rec = engine_rec->createExecutionContext();
    assert(context_rec != nullptr);
    delete[] trtModelStreamRec;

    float *buffers[2];
    const int inputIndex = engine_det->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine_det->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
   

    auto out_dims = engine_det->getBindingDimensions(1);
    auto output_size = 1;
    int OUTPUT_CANDIDATES = out_dims.d[1];

       for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }


    CHECK(cudaMalloc((void**)&buffers[inputIndex],  3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc((void**)&buffers[outputIndex], output_size * sizeof(float)));


     // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    float *affine_matrix_host=nullptr;
    float *affine_matrix_device=nullptr;
    float *decode_ptr_host=nullptr;
    float *decode_ptr_device=nullptr;
    decode_ptr_host = new float[1+MAX_OBJECTS*NUM_BOX_ELEMENT];
    // prepare input data cache in pinned memory 
    CHECK(cudaMallocHost((void**)&affine_matrix_host,sizeof(float)*6));
    CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CHECK(cudaMalloc((void**)&affine_matrix_device,sizeof(float)*6));
    CHECK(cudaMalloc((void**)&decode_ptr_device,sizeof(float)*(1+MAX_OBJECTS*NUM_BOX_ELEMENT)));

    auto out_dims_rec = engine_rec->getBindingDimensions(1);
    auto output_size_rec = 1;
    int OUTPUT_CANDIDATES_REC = out_dims_rec.d[1];

    for(int j=0;j<out_dims_rec.nbDims;j++) {
        output_size_rec *= out_dims_rec.d[j];
    }

    // static float* prob = new float[output_size];

    static float* prob_rec = new float[output_size_rec];

      
 
 // 识别模型 参数
     int plate_rec_input_w = 168;  
    int plate_rec_input_h = 48;
    float* blob_rec=new float[plate_rec_input_w*plate_rec_input_h*3];

    float mean_value=0.588;
    float std_value =0.193;

    const char* plate_rec_input_name = "images"; //onnx 输入  名字
    const char* plate_rec_out_name= "output"; //onnx 输出 名字

//  识别模型 参数
    
    cv::Point2f rect[4];
    cv::Point2f order_rect[4];
    cv::Point  point[1][4];

    // std::string imgPath ="/mnt/Gpan/Mydata/pytorchPorject/Chinese_license_plate_detection_recognition/imgs";
    std::string input_image_path=argv[3];
     std::string imgPath=argv[3];
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imgPath.c_str()),imagList,fileType);
    double sumTime = 0;
    int index = 0;
    for (auto &input_image_path:imagList) 
    {
        affine_matrix afmt;
        cv::Mat img = cv::imread(input_image_path);
        get_d2i_matrix(afmt,cv::Size(INPUT_W,INPUT_H),cv::Size(img.cols,img.rows));
          double begin_time = cv::getTickCount();
         float *buffer_idx = (float*)buffers[inputIndex];
        size_t size_image = img.cols * img.rows * 3;
        size_t size_image_dst = INPUT_H * INPUT_W * 3;
        memcpy(img_host, img.data, size_image);
        memcpy(affine_matrix_host,afmt.d2i,sizeof(afmt.d2i));
        CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, affine_matrix_device,stream);   //cuda 前处理
        double time_pre = cv::getTickCount();
        double time_pre_=(time_pre-begin_time)/cv::getTickFrequency()*1000;
        // std::cout<<"preprocessing time is "<<time_pre_<<" ms"<<std::endl;
        // doInference_cu(*context_det,stream, (void**)buffers,prob,1,output_size);
        context_det->enqueueV2((void **)buffers,stream,nullptr);
        CHECK(cudaMemsetAsync(decode_ptr_device,0,sizeof(int),stream));
        float *predict =(float *) buffers[outputIndex];
        decode_kernel_invoker(predict,OUTPUT_CANDIDATES,NUM_CLASSES,4,BBOX_CONF_THRESH,affine_matrix_device,decode_ptr_device,MAX_OBJECTS,stream);  //cuda 后处理

        nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, stream);//cuda nms
        
        CHECK(cudaMemcpyAsync(decode_ptr_host,decode_ptr_device,sizeof(float)*(1+MAX_OBJECTS*NUM_BOX_ELEMENT),cudaMemcpyDeviceToHost,stream));
        cudaStreamSynchronize(stream);

        int boxes_count=0;
        int count = std::min((int)*decode_ptr_host,MAX_OBJECTS);
        std::vector<bbox> boxes;

         for (int i = 0; i<count;i++)
        {
           int basic_pos = 1+i*NUM_BOX_ELEMENT;
           int keep_flag= decode_ptr_host[basic_pos+6];
           if (keep_flag==1)
           {
             boxes_count+=1;
             bbox  box;
             box.x1 =  decode_ptr_host[basic_pos+0];
             box.y1 =  decode_ptr_host[basic_pos+1];
             box.x2 =  decode_ptr_host[basic_pos+2];
             box.y2 =  decode_ptr_host[basic_pos+3];
             box.score=decode_ptr_host[basic_pos+4];
             int landmark_pos = basic_pos+7;
             for (int id = 0; id<4; id+=1)
             {
                box.landmarks[2*id]=decode_ptr_host[landmark_pos+2*id];
                box.landmarks[2*id+1]=decode_ptr_host[landmark_pos+2*id+1];
             }
             boxes.push_back(box);
           }
        }
         
        std::cout<<input_image_path<<" ";
        
        for (int i = 0; i<boxes.size(); i++)
        {
           
            for (int j= 0; j<4; j++)
            order_rect[j]=cv::Point(boxes[i].landmarks[2*j],boxes[i].landmarks[2*j+1]);
            
           cv::Mat roiImg = getTransForm(img,order_rect);  //根据关键点进行透视变换
           int label = boxes[i].label;
           if (label)             //判断是否双层车牌，是的话进行分割拼接
                roiImg=get_split_merge(roiImg);
            //    cv::imwrite("roi.jpg",roiImg);
            cv::resize(roiImg,roiImg,cv::Size(plate_rec_input_w,plate_rec_input_h));
            cv::Mat pr_img =roiImg;
        
            auto rec_b = cv::getTickCount();
            blobFromImage_plate(pr_img,mean_value,std_value,blob_rec);
            auto rec_e = cv::getTickCount();
            auto rec_gap = (rec_e-rec_b)/cv::getTickFrequency()*1000;
            
            doInference(*context_rec, blob_rec, prob_rec, output_size_rec, pr_img.size(),plate_rec_input_name,plate_rec_out_name);
            auto plate_number = decode_outputs(prob_rec,output_size_rec);
            auto plate_number_pinyin= decode_outputs_pingyin(prob_rec,output_size_rec); 
            cv::Point origin; 
            origin.x = boxes[i].x1;
            origin.y = boxes[i].y1;
            cv::putText(img, plate_number_pinyin, origin, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2, 8, 0);
            std::cout<<" "<<plate_number;
        
        }
          double end_time = cv::getTickCount();
          auto time_gap = (end_time-begin_time)/cv::getTickFrequency()*1000;
        std::cout<<"  time_gap: "<<time_gap<<"ms ";
         if (index)
            {
                // double use_time =(cv::getTickCount()-begin_time)/cv::getTickFrequency()*1000;
                sumTime+=time_gap;
            }
        std::cout<<std::endl;
        // delete [] blob_detect;
        index+=1;
    }

//    cv::imwrite("out.jpg",img);
 
    // destroy the engine
    std::cout<<"averageTime:"<<(sumTime/(imagList.size()-1))<<"ms"<<std::endl;
    context_det->destroy();
    engine_det->destroy();
    runtime_det->destroy();
 
    context_rec->destroy();
    engine_rec->destroy();
    runtime_rec->destroy();
   delete [] blob_rec;
   delete [] prob_rec;
   delete [] decode_ptr_host;
    cudaStreamDestroy(stream);
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(affine_matrix_device));
    CHECK(cudaFreeHost(affine_matrix_host));
    CHECK(cudaFree(decode_ptr_device));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    return 0;
}
