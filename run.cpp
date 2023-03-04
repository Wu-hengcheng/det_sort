#include <iostream>
#include "run.h"

Run::Run(int i):person_det_mode(i)
{
    // Init();
    YAML::Node config = YAML::LoadFile("../config/config.yaml");
    pub_topic = config["person_publish_topic"].as<std::string>();
    sub_img_topic = config["sub_img_topic"].as<std::string>();
}

Run::~Run()
{
    ROS_INFO("CONSTRUCT!!!");
    delete mSubscriber;
}

void Run::Start()
{
    ROS_INFO("StartRun");
    mReceiveThread = std::thread(&Run::ReceivedThreadFunc, this);
    sleep(1);
     mPersonThread = std::thread(&Run::PersonThreadFunc, this);
#if 0
    std::thread testThread = std::thread(&Run::testFunc, this);
#endif
    if(mReceiveThread.joinable())
    {
        mReceiveThread.join();
    }
     if(mPersonThread.joinable())
     {
         mPersonThread.join();
     }
}
void Run::ReceivedThreadFunc()
{
    mSubscriber = new MySubscribe(sub_img_topic);
    mSubscriber->ReceiveImg();
}

void Run::PersonThreadFunc()
{
    ROS_INFO("PersonThreadFunc ");
    
    persondetector = std::unique_ptr<Yolox>(new Yolox(person_det_mode));
    person_pub = n.advertise<std_msgs::String>(pub_topic , 1000);
    // cv::Mat tmpimg = cv::imread("../0001_166.png")
    while(true)
    {
        
        cv::Mat tmpimg = (mSubscriber->GetImg()).clone();
        if(tmpimg.data == nullptr)
        {
            ROS_INFO("EMPTY!!!!!!!!!!!!!!!!!!!11");
	    sleep(1);
            continue;
        }
         pi += 1;
         if(pi % 500 == 0&&pi%1000 != 0)
         {
//             std::cout << "start pi = " << pi << std::endl;
             p_start0 = clock();
         }
         persondetector->run(tmpimg, person_pub/*, personbox, img_cnt*/);
//         persondetector->run(tmpimg, PERSON, person_pub/*, personbox, img_cnt*/);
          if(pi%1000 == 0)
          {
//              std::cout << "end pi = " << pi << std::endl;
              p_end0 = clock();
              float t = (double)(p_end0 - p_start0)/CLOCKS_PER_SEC;
              std::cout << "person fps = " << 500 /t << std::endl;
          }

         usleep(16*1000);
    }

}
void Run::testFunc()
{
    std::thread testrecthread(&Run::testrec, this);
    if(testrecthread.joinable())
    {
       testrecthread.join(); 
    }
}
void Run::testrec()
{
	ros::NodeHandle n;
    ros::Subscriber detect_sub = n.subscribe("/cd206/perception/cameraDetect/chX/light", 10, &Run::detect_callback, this);
    ros::MultiThreadedSpinner s(2);
    ros::spin(s);            
}

void Run::detect_callback(const std_msgs::String::ConstPtr& msg)
{    
    CameraObject::CameraObjects objects_msg;
    objects_msg.ParseFromString(msg->data);
    // int id = objects_msg.id();
    std::cout << "id = " << objects_msg.type() << std::endl;
    std::cout << "timestamp = " << objects_msg.timestamp() << std::endl;
    for (int i = 0; i < objects_msg.boxes_size(); i++)
    {
        const CameraObject::CameraBox box = objects_msg.boxes(i);
        std::cout << "x_" << i << " = " << box.xpixel() << std::endl;
        std::cout << "y_" << i << " = " << box.ypixel() << std::endl;
        std::cout << "w_" << i <<  " = " << box.wpixel() << std::endl;
        std::cout << "h_" << i <<  " = " << box.hpixel() << std::endl;
        std::cout << "realx_" << i <<  " = " << box.xworld() << std::endl;
        std::cout << "realy_" << i <<  " = " << box.yworld() << std::endl;
        std::cout << "realz_" << i <<  " = " << box.zworld() << std::endl;
        
        std::cout << "lenReal_" << i <<  " = " << box.lenreal() << std::endl;
        std::cout << "wReal_" << i <<  " = " << box.wreal() << std::endl;
        std::cout << "hReal_" << i <<  " = " << box.hreal() << std::endl;

        std::cout << "prob_" << i << " = "  << box.conf() << std::endl;
        std::cout << "label_" << i << " = "  << box.label() << std::endl;
    }
    
}
