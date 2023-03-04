
// #include "opencv2/opencv.hpp"
// #include<opencv2/opencv.hpp>
// #include<opencv2/highgui/highgui.hpp>
// #include<opencv2/imgproc/imgproc.hpp>

#include<stdio.h>
#include <sys/time.h>
#include<math.h>
#include<vector>
#include <chrono>
#include <thread>
#include <iostream>
// #include "light2stage.h"
#include "run.h"


using namespace cv;
using namespace std;

#define TIME_START time_start=std::chrono::steady_clock::now();
#define TIME_END(NAME) time_end=std::chrono::steady_clock::now(); \
        duration=std::chrono::duration<double,std::micro>(time_end - time_start).count(); \
        cout<<(NAME)<<": takes time ::: "<<duration<<" us"<<endl;
auto time_start = std::chrono::steady_clock::now();
auto time_end   = std::chrono::steady_clock::now();
auto duration = 0.0F;

long int getCurrentTime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}


// int main(int argc, char** argv)
// {	
// // {'green': 0, 'left_green': 1, 'left_red': 2, 'left_yollow': 3, 'off': 4, 'other': 5, 'red': 6, 'right_green': 7, 
// // 'right_red': 8, 'right_yellow': 9, 'straight_green': 10, 'straight_red': 11, 'straight_yellow': 12, 'yellow': 13}

//     // ros::init(argc, argv, "image_listener");
//     cv::Mat img = cv::imread("/home/nvidia/Project/light2stage/0001_166.png");

//     Light2Stage yolox(1);
//     std::cout << img.rows << "+" << img.cols << std::endl;
//     yolox.run(img);
// 	return 0;
// }
int main(int argc, char** argv)
{	
    ros::init(argc, argv, "image_listener1");
    if(argc > 1 && std::string(argv[1]) == "-c")
    {
        Run runner(0);
        runner.Start();
    }
    else
    {
        Run runner(1);
        runner.Start();
    }
    
	return 0;
}


