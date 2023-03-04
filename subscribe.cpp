#include "subscribe.h"
#include <iostream>

MySubscribe::MySubscribe(std::string topic)
{ 
    subName = topic;

}  
MySubscribe::~MySubscribe()
{

}

void MySubscribe::ReceiveImg()
{
    // ROS_INFO("MySubscribe Received \n");
    // ros::init(argc, argv, "image_listener");
    // subName = "/cd206/driver/image/chX";
    // subName = "camera/image";

    //Topic you want to subscribe
    sub = n.subscribe(subName, 1, &MySubscribe::CallBack3, this);
//    ros::MultiThreadedSpinner s(3);
//    ros::spin(s);
    ros::spin();
}

void MySubscribe::CallBack3(const sensor_msgs::ImageConstPtr& msg)  
{
    // static int cnt = 0;
    // ROS_INFO("Received \n");
    cv::Mat bgrImage = cv_bridge::toCvShare(msg, "bgr8")->image;
    mReceiveLock.lock();
    mReceivedImg = bgrImage.clone();
    mReceiveLock.unlock();
        // ii += 1;
        // if(ii == 100)
        // {
        //     start0 = clock();
        // }
        // if(ii == 500)
        // {
        //     end0 = clock();
        //     float t = (double)(end0 - start0)/CLOCKS_PER_SEC;
        //     std::cout << "light fps = " << 400 /t << std::endl;
        // }
//    usleep(10*1000);
} 

void MySubscribe::CallBack2(const std_msgs::UInt8MultiArray::ConstPtr& msg)  
{
    // static int cnt = 0;
    // ROS_INFO("Received \n");
    cv::Mat bgrImage(720, 1280, CV_8UC3);
    std::copy(msg->data.begin(), msg->data.end(), bgrImage.data);
    mReceiveLock.lock();
    mReceivedImg = bgrImage.clone();
    mReceiveLock.unlock();
        // ii += 1;
        // if(ii == 100)
        // {
        //     start0 = clock();
        // }
        // if(ii == 500)
        // {
        //     end0 = clock();
        //     float t = (double)(end0 - start0)/CLOCKS_PER_SEC;
        //     std::cout << "light fps = " << 400 /t << std::endl;
        // }
//    usleep(10*1000);
}  

void MySubscribe::CallBack(const std_msgs::UInt8MultiArray& msg)
{  
    // static int cnt = 0;
		// ROS_INFO("Received \n");
        //****************************************8
        int headSize = sizeof(MsgImageHead);
        unsigned char headArr[headSize];
        for(int i = 0; i < headSize; i++)
        {
            headArr[i] = msg.data[i];
        }
        MsgImageHead head;
        memcpy(&head, headArr, headSize);
        cv::Mat bgrImage(head.height, head.width, CV_8UC3);
        std::copy(msg.data.begin() + headSize, msg.data.end(), bgrImage.data);
        cv::resize(bgrImage, bgrImage, cv::Size(head.width, head.height));
        
        mReceiveLock.lock();
        mReceivedImg = bgrImage;
        mReceiveLock.unlock();		
}   

cv::Mat &MySubscribe::GetImg()
{
    const std::lock_guard<std::mutex> lock(mReceiveLock);
    return mReceivedImg;
}

