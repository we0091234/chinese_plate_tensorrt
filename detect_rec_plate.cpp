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
// static const int NUM_CLASSES = 2;  //单层车牌，双层车牌两类


// const char* INPUT_BLOB_NAME = "input"; //onnx 输入  名字
// const char* OUTPUT_BLOB_NAME = "output"; //onnx 输出 名字
static Logger gLogger;

cv::Mat static_resize(cv::Mat& img,int &top,int &left)  //对应yolov5中的letter_box
{
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    left = (INPUT_W-unpad_w)/2;
    top = (INPUT_H-unpad_h)/2;
    int right = INPUT_W-unpad_w-left;
    int bottom = INPUT_H-unpad_h-top;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
      
    cv::Mat out;
  
    cv::copyMakeBorder(re,out,top,bottom,left,right,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
 
    return out;
}

struct Object
{
    cv::Rect_<float> rect; //
    float landmarks[8]; //4个关键点
    int label;
    float prob;
};


static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int find_max(float *prob,int num) //找到类别
{
    int max= 0;
    for(int i=1; i<num; i++)
    {
        if (prob[max]<prob[i])
         max = i;
    }

    return max;

}


static void generate_yolox_proposals(float *feat_blob, float prob_threshold,
                                     std::vector<Object> &objects,int OUTPUT_CANDIDATES) {
  const int num_class = 2;

  const int num_anchors = OUTPUT_CANDIDATES;

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    // const int basic_pos = anchor_idx * (num_class + 5 + 1);
    // float box_objectness = feat_blob[basic_pos + 4];

    // int cls_id = feat_blob[basic_pos + 5];
    // float score = feat_blob[basic_pos + 5 + 1 + cls_id];
    // score *= box_objectness;


    const int basic_pos = anchor_idx * (num_class + 5 + 8); //5代表 x,y,w,h,object_score  8代表4个关键点
    float box_objectness = feat_blob[basic_pos + 4];

    int cls_id = find_max(&feat_blob[basic_pos +5+8],num_class);   //找到类别
    float score = feat_blob[basic_pos + 5 +8 + cls_id];
    score *= box_objectness;


    if (score > prob_threshold) {
      // yolox/models/yolo_head.py decode logic
      float x_center = feat_blob[basic_pos + 0];
      float y_center = feat_blob[basic_pos + 1];
      float w = feat_blob[basic_pos + 2];
      float h = feat_blob[basic_pos + 3];
      float x0 = x_center - w * 0.5f;
      float y0 = y_center - h * 0.5f;
      
      float *landmarks=&feat_blob[basic_pos +5];
    

      Object obj;
      obj.rect.x = x0;
      obj.rect.y = y0;
      obj.rect.width = w;
      obj.rect.height = h;
      obj.label = cls_id;
      obj.prob = score;
      for (int i = 0; i<8; i++)
      {
         obj.landmarks[i]=landmarks[i];
      }
      

      objects.push_back(obj);
    }
  }
}

float* blobFromImage(cv::Mat& img,float *blob){
    // float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    int k = 0;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                // blob[c * img_w * img_h + h * img_w + w] =
                //     (float)img.at<cv::Vec3b>(h, w)[c];
                    blob[k++] =
                    (float)img.at<cv::Vec3b>(h, w)[2-c]/255.0;   //2-c 是因为opencv读取的是BGR 检测模型训练时候用的RGB
            }
        }
    }
    return blob;
}

float* blobFromImage_plate(cv::Mat& img,float mean_value,float std_value)
{
    float* blob = new float[img.total()*3];
    int channels = 3;
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
    return blob;
}

static void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h,int OUTPUT_CANDIDATES,int top,int left) {
        std::vector<Object> proposals;
        generate_yolox_proposals(prob,  BBOX_CONF_THRESH, proposals,OUTPUT_CANDIDATES);
        // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        // std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x-left) / scale;
            float y0 = (objects[i].rect.y-top) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width-left) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height-top) / scale;
            
            float *landmarks = objects[i].landmarks;
            for(int i= 0; i<8; i++)
            {
                if(i%2==0)
                landmarks[i]=(landmarks[i]-left)/scale;
                else
                landmarks[i]=(landmarks[i]-top)/scale;
            }
            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
}

const float color_list[4][3] =
{
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {0, 255, 255},
};




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
    context.enqueue(1, buffers, stream, nullptr);
    // context.enqueueV2( buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
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

void readTrtModel(const std::string trtFile,char *&trtModelStream,size_t &size)
{
    // size_t size{0};
    std::ifstream file(trtFile, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
}

class trtModel
{
    public:
    char *trtModelStream{nullptr};
    IRuntime *runtime{nullptr};
    ICudaEngine *engine{nullptr};
    IExecutionContext *context{nullptr};
    size_t size{0};
    int output_size=1;
    float *prob=nullptr;
    int OUTPUT_CANDIDATES;
    std::string input_blob_name;
    std::string out_blob_name;

    trtModel(const std::string trtModelPath,std::string input_blob_name,std::string out_blob_name)
    {
        readTrtModel(trtModelPath,this->trtModelStream,this->size);
        runtime =createInferRuntime(gLogger);
        assert(runtime != nullptr);
        engine = runtime->deserializeCudaEngine(this->trtModelStream, this->size);
        assert(engine != nullptr); 
        context = engine->createExecutionContext();
        assert(context != nullptr);
        
        get_output_size();
        prob = new float[output_size];
        this->input_blob_name=input_blob_name;
        this->out_blob_name=out_blob_name;
    }

    ~trtModel()
    {
        if (!trtModelStream)
        delete [] trtModelStream;
        if(!runtime)
        runtime->destroy();
        if(!engine)
        engine->destroy();
        if(!context)
        context->destroy();
        if(!prob)
        delete [] prob;
        // std::cout<<"对象销毁成功"<<std::endl;
    }
    void get_output_size()
    {
        // int out_size=1;
        auto out_dims = engine->getBindingDimensions(1);
        OUTPUT_CANDIDATES = out_dims.d[1];

        for(int j=0;j<out_dims.nbDims;j++)
        {
            output_size *= out_dims.d[j];
        }
    }

    void doInference( float* input, cv::Size input_shape) 
    {
        const char *input_name=input_blob_name.c_str();
        const char *out_name =out_blob_name.c_str();
        const ICudaEngine& engine = context->getEngine();
       
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine.getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine.getBindingIndex(input_name);

        assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
        const int outputIndex = engine.getBindingIndex(out_name);
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
        context->enqueue(1, buffers, stream, nullptr);
        // context.enqueueV2( buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(prob, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
}

};

void pre_pressing(cv::Mat &img,int &top,int &left,float &scale,float *blob_detect)
{
      cv::Mat pr_img = static_resize(img,top,left);
      blob_detect = blobFromImage(pr_img,blob_detect);
      scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
}



int main(int argc, char** argv) 
{
    cudaSetDevice(DEVICE);

   //检测模型参数
    std::string detect_input_name = "input";// 检测 模型 onnx 输入  名字
    std::string detect_output_name= "output";//检测模型 onnx 输出  名字

  //识别模型参数//
    std::string rec_input_name = "images"; //识别模型 onnx 输入  名字
    std::string  rec_out_name= "output"; //识别模型 onnx 输出 名字
    int plate_rec_input_w = 168,plate_rec_input_h =48;  
    float mean_value=0.588,std_value=0.193;
    cv::Point2f order_rect[4];
  //识别模型参数//

    trtModel detectModel(argv[1],detect_input_name,detect_output_name);  //初始化检测模型
    trtModel recModel(argv[2],rec_input_name,rec_out_name);             //初始化识别模型

    float* blob_detect=new float[INPUT_H*INPUT_W*3];
    float* blob_rec;

    std::string input_image_path=argv[3];
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(argv[3]),imagList,fileType);
    double pre_pressin_time=0;
    double forword_sumTime = 0;
    int index= 0;
    for (auto &input_image_path:imagList) 
    {
        std::cout<<input_image_path<<" ";
        cv::Mat img = cv::imread(input_image_path);
        
        int top=0,left=0;
        float scale=0;
        auto pre_time_b=cv::getTickCount();
        pre_pressing(img,top,left,scale,blob_detect);  //检测前处理
        auto pre_time_e=cv::getTickCount();
        auto time_gap_pre = (pre_time_e-pre_time_b)/cv::getTickFrequency()*1000;
       if (index)
       pre_pressin_time+=time_gap_pre;
        
        auto time_b = cv::getTickCount();
        detectModel.doInference(blob_detect,cv::Size(INPUT_W,INPUT_H));
        std::vector<Object> objects;
        decode_outputs(detectModel.prob, objects, scale, img.cols, img.rows,detectModel.OUTPUT_CANDIDATES,top,left);
        for (int i = 0; i<objects.size(); i++)
        {
            cv::rectangle(img, objects[i].rect, cv::Scalar(0,255,0), 2);
            for (int j= 0; j<4; j++)
            {
            cv::Scalar color = cv::Scalar(color_list[j][0], color_list[j][1], color_list[j][2]);
            cv::circle(img,cv::Point(objects[i].landmarks[2*j], objects[i].landmarks[2*j+1]),5,color,-1);
            order_rect[j]=cv::Point(objects[i].landmarks[2*j],objects[i].landmarks[2*j+1]);
            }
            
           cv::Mat roiImg = getTransForm(img,order_rect);  //根据关键点进行透视变换
           int label = objects[i].label;
           if (label)             //判断是否双层车牌，是的话进行分割拼接
                roiImg=get_split_merge(roiImg);
            //    cv::imwrite("roi.jpg",roiImg);
            cv::resize(roiImg,roiImg,cv::Size(plate_rec_input_w,plate_rec_input_h));
            cv::Mat pr_img =roiImg;
            // std::cout << "blob image" << std::endl;
            
            blob_rec = blobFromImage_plate(pr_img,mean_value,std_value);
            recModel.doInference(blob_rec,pr_img.size());
            // doInference(*recModel.context, blob_rec, prob_rec, output_size_rec, pr_img.size(),plate_rec_input_name,plate_rec_out_name);
            auto plate_number = decode_outputs(recModel.prob,recModel.output_size);
            auto plate_number_pinyin= decode_outputs_pingyin(recModel.prob,recModel.output_size);
            
            cv::Point origin; 
            origin.x = objects[i].rect.x;
            origin.y = objects[i].rect.y;
            cv::putText(img, plate_number_pinyin, origin, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2, 8, 0);
            std::cout<<plate_number<<" ";
       
        }
        auto time_e = cv::getTickCount(); 
        std::cout<<std::endl;
       
       auto time_gap = (time_e-time_b)/cv::getTickFrequency()*1000;
       if (index)
       forword_sumTime+=time_gap;
       index+=1;
    }

std::cout<<"forward平均时间: "<<forword_sumTime/(imagList.size()-1)<<" ms"<<std::endl;
std::cout<<"前处理平均时间: "<<pre_pressin_time/(imagList.size()-1)<<" ms"<<std::endl;
//    cv::imwrite("out.jpg",img);
  delete [] blob_rec;
   delete [] blob_detect;
    return 0;
}
