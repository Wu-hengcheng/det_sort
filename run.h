#ifndef __PUBLISH_H__
#define __PUBLISH_H__

#include "yolox.h"

#include "ros/ros.h"
#include "object.pb.h"
#include "std_msgs/String.h"
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/image_encodings.h>
#include<image_transport/image_transport.h>
#include<std_msgs/UInt8MultiArray.h>
#include <time.h>
#include "subscribe.h"
#include <yaml-cpp/yaml.h>


class Run
{
    public:
        Run(int i);
        ~Run();
        void Start();
        void PersonThreadFunc();
        void ReceivedThreadFunc();
        void testFunc();
        void detect_callback(const std_msgs::String::ConstPtr& msg);
        void testrec();
    private:
        std::thread mReceiveThread;
        std::thread mPersonThread;
        MySubscribe *mSubscriber;

        ros::Publisher person_pub; 
        std::string pub_topic , sub_img_topic;
        std::unique_ptr<Yolox> persondetector;
        ros::NodeHandle n;

        int person_det_mode;


        int pi = 0;
        clock_t p_start0, p_end0;
};
#endif
