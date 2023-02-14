#include "plate_rec.h"
#include "plate_detect.h"
#include "utils.hpp"
static Logger gLogger;

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
           
            std::string plate_no;
            std::string plate_color;
            for (int j= 0; j<4; j++)
            order_rect[j]=cv::Point(bboxes[i].landmarks[2*j],bboxes[i].landmarks[2*j+1]);

            cv::Mat roiImg = getTransForm(img,order_rect);  //根据关键点进行透视变换
            if (bboxes[i].label==1)             //判断是否双层车牌，是的话进行分割拼接
                roiImg=get_split_merge(roiImg);
            cv::resize(roiImg,roiImg,cv::Size(168,48));
            plate_rec.plate_rec_color(roiImg,cv::Size(168,48),plate_no,plate_color);  //识别
            std::cout<<plate_no<<" "<<plate_color<<" ";
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
    }

   

    
   std::cout<<"averageTime:"<<(sumTime/(imagList.size()-1))<<"ms"<<std::endl;
    

    // cv::imwrite("result.jpg",img);

    return 0;
}