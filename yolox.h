#ifndef _YOLOX_H_
#define _YOLOX_H_

#include "logging.h"
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
#include "pthread.h"
#include <thread>
#include <string.h>

#include "ros/ros.h"
#include "object.pb.h"
#include "std_msgs/String.h"

#include <NvOnnxParser.h>
// #include "calibrator.h"
#include <fstream>
#include <yaml-cpp/yaml.h>

#include "deepsort/include/manager.hpp"

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
// #define NMS_THRESH 0.45
// #define BBOX_CONF_THRESH 0.3

#define BATCH_SIZE 1
// #define USE_INT8
#define USE_FP16
using namespace nvinfer1;

//enum TaskId
//{
//    PERSON = 0,
//    LIGHT = 1
//};

class Yolox
{
private:
    cv::Mat mIput;
    cv::Mat mPersonOutput;
    cv::Mat mOutput;

    //cv::FileStorage depth("../depth.yaml", cv::FileStorage::READ);
    //cv::FileStorage width("../width.yaml", cv::FileStorage::READ);
    cv::FileStorage depth;
    cv::FileStorage width;
    cv::FileStorage undistort;
    cv::Mat depth_map;
    cv::Mat width_map;

    //wuhc: add sort objective 
    Trtyolosort* deepsort;
    std::string sort_engine_path;

    //wuhc: add yolo onnx and engine path
    std::string yolo_onnx_path ,  yolo_engine_path;



    int INPUT_W;
    int INPUT_H;
    int NUM_CLASSES;
    const char* INPUT_BLOB_NAME;
    const char* OUTPUT_BLOB_NAME;
    bool show_img ;
    float nms_thresh , bbox_conf_thresh;
    Logger gLogger;
    void* buffers[2];
    const std::string ModelName = {"person"};
    const std::string ROOT_PATH = "../";
    std::vector<std::string> class_names;
public:
    Yolox(int mode);
    // ~Yolox();
    IExecutionContext* mcontext;
    cv::Mat xmap;
    cv::Mat ymap;

    float* mprob;
    int moutput_size;
    cudaStream_t mstream;
    int minputIndex;
    int moutputIndex;
    int mBatchSize;
    IRuntime* mruntime;
    ICudaEngine* mengine;

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
    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };
    void release();

    cv::Mat static_resize(cv::Mat& img);
    void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
    static inline float intersection_area(const Object& a, const Object& b)
    {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
    }
    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
    void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects);
    float* blobFromImage(cv::Mat& img);
    void decode_outputs(CameraObject::CameraObjects &objects_msg, float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h/*, std::ofstream &file, int ii*/);
    cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects/*, std::string f*/);
    void doInference(float* input, cv::Size input_shape);
    void run(cv::Mat& img, ros::Publisher object_pub/*, std::ofstream &file, int ii*/);

    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, nvinfer1::DataType dt);
    void api_to_model(unsigned int maxBatchSize, IHostMemory** modelStream);
    bool make_engine();
};

#endif

