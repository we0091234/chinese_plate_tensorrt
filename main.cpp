#include "plate_rec.h"
#include "plate_detect.h"
#include "utils.hpp"
#include "cuda_runtime.h"
// #include <opencv2/freetype.hpp>
static Logger gLogger;

// void drawBboxes(cv::Mat &img ,std::vector<bbox> &bboxes,std::string &text,int i)
// {
//     std::string ttf_pathname = "/mnt/Gu/xiaolei/cplusplus/trt_project/trt_plate/font/NotoSansCJK-Regular.otf";
//     int top = bboxes[i].y1;
//     int left = bboxes[i].x1;
//     int baseLine1;
//     cv::Size labelSize1 = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine1);
//     top = cv::max(top, labelSize1.height);
//     //画框
//     cv::rectangle(img, cv::Point(left, top - round(1.6*labelSize1.height)), cv::Point(left + round(1.2*labelSize1.width), top + baseLine1), cv::Scalar(255, 255, 255), cv::FILLED);
//     cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();

//     ft2->loadFontData(ttf_pathname,0);
//     //标签
//     ft2->putText(img, text, cv::Point(left, top), 21, cv::Scalar(255, 0, 0), -1,4,true);

// }

int main(int argc,char **argv)
{
    
    
    std::string imgPath =std::string(argv[3]);
    
    std::vector<std::string> imagList;
    std::vector<std::string>fileType{"jpg","png"};
    readFileList(const_cast<char *>(imgPath.c_str()),imagList,fileType);


    const char *trt_model_detect = argv[1];  //检测模型
    const char *trt_model_rec = argv[2];   //识别模型
    std::vector<bbox> bboxes;
    // std::string image_path = "/mnt/Gu/xiaolei/cplusplus/trt_project/trt_plate/color_test/tmpD60D.png";
 
    PlateRec plate_rec;
    plate_rec.loadTrtModel(trt_model_rec,gLogger);
    PlateDetect plate_detect;
    plate_detect.loadTrtModel(trt_model_detect,gLogger);


    cv::Point2f order_rect[4];

    int index = 0;
    double sumTime = 0;
    for(auto &file_path:imagList)
    {
         std::cout<<file_path<<" ";
    std::string plate_no;
    std::string plate_color;
     cv::Mat img = cv::imread(file_path);
    double begin_time = cv::getTickCount();
    plate_detect.detect(img,bboxes);   //检测
     for (int i = 0; i<bboxes.size(); i++)
     {

            cv::rectangle(img,cv::Point(bboxes[i].x1,bboxes[i].y1),cv::Point(bboxes[i].x2,bboxes[i].y2),cv::Scalar(0,255,0));
            
            std::string plate_no;
            std::string plate_color;
            for (int j= 0; j<4; j++)
            order_rect[j]=cv::Point(bboxes[i].landmarks[2*j],bboxes[i].landmarks[2*j+1]);
            cv::Mat roiImg = getTransForm(img,order_rect);  //根据关键点进行透视变换
            if (bboxes[i].label==1)             //判断是否双层车牌，是的话进行分割拼接
                roiImg=get_split_merge(roiImg);
            cv::resize(roiImg,roiImg,cv::Size(168,48));
            plate_rec.plate_rec_color(roiImg,cv::Size(168,48),plate_no,plate_color);  //识别
            
            std::string label1=plate_no+std::string(" ")+plate_color;
            // drawBboxes(img ,bboxes,label1,i);
            std::cout<<label1<<" ";
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
        index+=1;
    bboxes.clear();
    
    // std::string save_img_name = std::to_string(index)+".jpg";
    // cv::imwrite(save_img_name,img);
    }

   

    
   std::cout<<"averageTime:"<<(sumTime/(imagList.size()-1))<<"ms"<<std::endl;
    


    return 0;
}