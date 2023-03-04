#ifndef __MYSUBSCRIBE_H__
#define __MYSUBSCRIBE_H__

// #include "yolox.h"

#include "ros/ros.h"
#include "object.pb.h"
#include "std_msgs/String.h"
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/image_encodings.h>
#include<image_transport/image_transport.h>
#include<std_msgs/UInt8MultiArray.h>
#include <time.h>
#include <mutex>

struct MsgImageHead
{
    int headBytes;
    int channel;
    int64 frameld;
    int64 timestamp;
    int step;
    int encoding;
    int width;
    int height;
    int flag =0xfefefe;
    int dataByte;
};

// enum TaskId
// {
//     PERSON = 0,
//     LIGHT = 1
// };

class MySubscribe  
{  
    public:  
        MySubscribe(std::string topic);
        ~MySubscribe();
        void CallBack(const std_msgs::UInt8MultiArray& msg);
        void CallBack2(const std_msgs::UInt8MultiArray::ConstPtr& msg);
        void CallBack3(const sensor_msgs::ImageConstPtr& msg); 
        void ReceiveImg();
        cv::Mat &GetImg();


    private: 
        int ii = 0;
        clock_t start0, end0; 

        std::mutex mReceiveLock;
        cv::Mat mReceivedImg;
        ros::NodeHandle n;  
        ros::Subscriber sub;
        std::string subName;      
};

#endif
