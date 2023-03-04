#include "manager.hpp"
using std::vector;
using namespace cv;
static Logger gLogger;

Trtyolosort::Trtyolosort(std::string sort_engine_path , int gpuID , std::vector<std::string> names , bool show_img){
	sort_engine_path_ = sort_engine_path;
    track_names = names;
	DS = new DeepSort(sort_engine_path_, 128, 256, gpuID, &gLogger);
	printf("create DeepSort  , instance = %p\n", DS);
    show = show_img;
}
void Trtyolosort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1); 
        std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		//std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
		// std::string lbl = cv::format("ID:%d_x:%f_y:%f",(int)box.trackID,(box.x1+box.x2)/2,(box.y1+box.y2)/2);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
    }
    cv::imshow("img", temp);
    cv::waitKey(1);
}
int Trtyolosort::TrtDetect(cv::Mat &frame,float &conf_thresh,std::vector<DetectBox> &det){
	// yolo detect
	// auto ret = yolov5_trt_detect(trt_engine, frame, conf_thresh,det);
	DS->sort(frame,det);
	showDetection(frame,det);
	return 1 ;
}

//wuhc : implement of the function definition
int Trtyolosort::TrtDetect(cv::Mat &frame , CameraObject::CameraObjects &det){

	DS->sort(frame,det);
    if (show)
	showDetection(frame, det);
	return 1 ;
}

void Trtyolosort::showDetection(cv::Mat& img, CameraObject::CameraObjects& boxes ) {
    //todo: add save video function
    cv::Mat temp = img.clone();
	//wuhc: show real class name and different colors boxes
    cv::Scalar color , txt_color , txt_bk_color;
    cv::String lbl;

    for (auto box : boxes.boxes()) {
        cv::Point lt(box.xpixel(), box.ypixel());
        cv::Point br(box.xpixel()+box.wpixel(), box.ypixel()+box.hpixel());
        color = cv::Scalar(color_list[(int)box.label()][0], color_list[(int)box.label()][1], color_list[(int)box.label()][2]);
        float c_mean = cv::mean(color)[0];
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }
        cv::rectangle(temp, lt, br, color*255, 1);

        lbl = cv::format("C:%s_id:%d_conf:%.2f", track_names[(int)box.label()].c_str(), (int)box.trackid(), box.conf());

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        txt_bk_color = color * 0.7 * 255;
        int y = lt.y +1;
        if (y > temp.rows) y = temp.rows;
        cv::rectangle(temp, cv::Rect(cv::Point(lt.x, y-label_size.height-baseLine), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);
        // cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
        cv::putText(temp, lbl, cv::Point(lt.x, y - label_size.height+5), cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
    cv::imshow("yolosort_img", temp);
    cv::waitKey(1);
}
