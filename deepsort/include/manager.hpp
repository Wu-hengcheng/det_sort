#ifndef _MANAGER_H
#define _MANAGER_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "deepsort.h"
#include "../../logging.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "time.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include "yolov5_lib.h"
#include "deepsort.h"
#include "../../object.pb.h"

using std::vector;
using namespace cv;
//static Logger gLogger;

class Trtyolosort{
public:
	// init 
	Trtyolosort(std::string sort_engine_path , int gpuID , std::vector<std::string> names  , bool show_img);
	// detect and show
	int TrtDetect(cv::Mat &frame,float &conf_thresh,std::vector<DetectBox> &det);

	//wuhc: add pure sort code that compatible with detection code
	int TrtDetect(cv::Mat &frame , CameraObject::CameraObjects &det );

	void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes);

	// wuhc: add show function 
	void showDetection(cv::Mat& img, CameraObject::CameraObjects &boxes );

	const float color_list[80][3] =
    {
        {0.000, 0.447, 0.741},
        {0.850, 0.325, 0.098},
        {0.929, 0.694, 0.125},
        {0.494, 0.184, 0.556},
        {0.466, 0.674, 0.188},
        {0.301, 0.745, 0.933},
        {0.635, 0.078, 0.184},
        {0.300, 0.300, 0.300},
        {0.600, 0.600, 0.600},
        {1.000, 0.000, 0.000},
        {1.000, 0.500, 0.000},
        {0.749, 0.749, 0.000},
        {0.000, 1.000, 0.000}
    };

private:
	std::string sort_engine_path_ ;
    //class names
    std::vector<std::string> track_names;

    //show 
    bool show ;

    // deepsort parms
    DeepSort* DS;
    std::vector<DetectBox> t;
};
#endif  // _MANAGER_H

