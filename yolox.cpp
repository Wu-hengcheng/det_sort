#include "yolox.h"

//const char* Yolox::class_names[20] = {"person" , "vehicle"};
Yolox::Yolox(int mode)
{
    // depth.open("../depth.yaml", cv::FileStorage::READ);
    // width.open("../width.yaml", cv::FileStorage::READ);
    // undistort.open("../undistort.yaml", cv::FileStorage::READ);
    // depth_map.create(cv::Size(1280*720, 1), CV_64FC1);
    // width_map.create(cv::Size(1280*720, 1), CV_64FC1);
    // xmap.create(cv::Size(1280*720, 1), CV_64FC1);
    // ymap.create(cv::Size(1280*720, 1), CV_64FC1);

    // depth["depth"] >> depth_map;
    // width["width"] >> width_map;

    // undistort["mapx"] >> xmap;
    // undistort["mapy"] >> ymap;
    YAML::Node config = YAML::LoadFile("../config/config.yaml");
    sort_engine_path = config["deepsort"]["deepsort_engine_path"].as<std::string>();
    INPUT_W = config["yolox"]["input_w"].as<int>();
    INPUT_H = config["yolox"]["input_h"].as<int>();
    NUM_CLASSES = config["yolox"]["num_classes"].as<int>();
    INPUT_BLOB_NAME = config["yolox"]["input_blob_name"].as<std::string>().c_str();
    OUTPUT_BLOB_NAME = config["yolox"]["output_blob_name"].as<std::string>().c_str();
    
    //std::cout << config["yolox"]["class_names"].as<std::string>() << std::endl;
    std::string class_all = config["yolox"]["class_names"].as<std::string>();
    std::stringstream ss(class_all);
    char c = ' ';
    std::vector<std::string> result;
    std::string str;
    //int i = 0 ; 
    while(getline(ss , str , c))
    {
        class_names.push_back(str);
        //std::cout << class_names[i++] << std::endl;
    }

    show_img = config["show_img"].as<bool>();
    nms_thresh = config["yolox"]["nms_thresh"].as<float>();
    bbox_conf_thresh = config["yolox"]["bbox_conf_thresh"].as<float>();
    yolo_onnx_path = config["yolox"]["onnx_path"].as<std::string>();
    yolo_engine_path = config["yolox"]["engine_path"].as<std::string>();
    

    //wuhc: create sort objective
    deepsort = new Trtyolosort(sort_engine_path , DEVICE , class_names , show_img);

    if(mode == 0)
    {
        std::cout << "enter mode == 0" << std::endl;
        make_engine();

    }

//    path_list.push_back(ROOT_PATH + "config/person/person_fp16.engine");
//    path_list.push_back(ROOT_PATH + "config/light/light_fp16.engine");
    // std::string path_list = ROOT_PATH + "config/person/person_fp16.engine";
        cudaSetDevice(DEVICE);
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{nullptr};
        size_t size{0};

        std::ifstream file(yolo_engine_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        mruntime = createInferRuntime(gLogger);
        assert(mruntime != nullptr);
        mengine = mruntime->deserializeCudaEngine(trtModelStream, size);
        assert(mengine != nullptr);
        mcontext = mengine->createExecutionContext();
        assert(mcontext != nullptr);
        delete[] trtModelStream;
        auto out_dims = mengine->getBindingDimensions(1);
        moutput_size = 1;
        for(int j = 0; j < out_dims.nbDims; j++)
        {
            moutput_size *= out_dims.d[j];
        }
        mprob = new float[moutput_size];

        const ICudaEngine& engine2 = mcontext->getEngine();
        mBatchSize = engine2.getMaxBatchSize();
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine2.getNbBindings() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        minputIndex = engine2.getBindingIndex(INPUT_BLOB_NAME);

        assert(engine2.getBindingDataType(minputIndex) == nvinfer1::DataType::kFLOAT);
        moutputIndex = engine2.getBindingIndex(OUTPUT_BLOB_NAME);

        // Create GPU buffers on deviceg
        CHECK(cudaMalloc(&buffers[minputIndex], mBatchSize * 3 * INPUT_W * INPUT_H * sizeof(float)));
        CHECK(cudaMalloc(&buffers[moutputIndex], mBatchSize * moutput_size*sizeof(float)));

        assert(engine2.getBindingDataType(moutputIndex) == nvinfer1::DataType::kFLOAT);
        //cv::FileStorage depth("../depth.yaml", cv::FileStorage::READ);
    //cv::FileStorage width("../width.yaml", cv::FileStorage::READ);
        // Create stream
        CHECK(cudaStreamCreate(&mstream));

}



ICudaEngine* Yolox::createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, nvinfer1::DataType dt)
 {
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    // std::string tmp = ROOT_PATH + "config/" + ModelName + "/" + ModelName + ".onnx";
    const char* onnx_filename = yolo_onnx_path.c_str();

    parser->parseFromFile(onnx_filename, static_cast<int>(Logger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    std::cout << "successfully load the onnx model" << std::endl;

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB  // 16MB, 2^20 bytes = 1MB

#ifdef USE_INT8
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    std::string calib_name_s = ROOT_PATH + "config/" + ModelName + "/" +ModelName + "_calibration/";
    const char * calib_name = calib_name_s.c_str();
    std::string table_name_s = ROOT_PATH + "config/" + ModelName + "/" +ModelName + "_int8calib.table";
    const char * table_name = table_name_s.c_str();
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, calib_name, table_name , INPUT_BLOB_NAME);

    config->setInt8Calibrator(calibrator);
#elif defined(USE_FP16)
    std::cout << "Your platform support FP16: " << (builder->platformHasFastFp16() ? "true" : "false") << std::endl;
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();  // 完成处理后释放

    // Release host memory
    return engine;
}

void Yolox::api_to_model(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig(); // int8

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, nvinfer1::DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

bool Yolox::make_engine()
{
    IHostMemory* modelStream{ nullptr };
    api_to_model(BATCH_SIZE, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream p(yolo_engine_path, std::ios::binary);

    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return true;
}

cv::Mat Yolox::static_resize(cv::Mat& img)
{
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

 void Yolox::generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_y = INPUT_H / stride;
        int num_grid_x = INPUT_W / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}


void Yolox::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void Yolox::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void Yolox::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


void Yolox::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_anchors = grid_strides.size();

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

float* Yolox::blobFromImage(cv::Mat& img)
{
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}


void Yolox::decode_outputs(CameraObject::CameraObjects &objects_msg, float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h/*, std::ofstream &file, int ii*/)
{
    std::vector<Object> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(strides, grid_strides);
    // generate_yolox_proposals(grid_strides, prob,  BBOX_CONF_THRESH, proposals);
    generate_yolox_proposals(grid_strides, prob,  bbox_conf_thresh, proposals);
    // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;
    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    // nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    nms_sorted_bboxes(proposals, picked, nms_thresh);

    int count = picked.size();
//    std::vector<Object> selectobjects;
//     std::cout << "num of boxes: " << count << std::endl;

    objects.resize(count);

    CameraObject::CameraBox *box;

    if(count == 0)
    {
        objects_msg.set_type(3);
    }
    objects_msg.set_type(0);
//    int mycnt = 0;
    unsigned int time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    objects_msg.set_timestamp(time_ms);
    // for (int i = 0; i < count; i++)
    // {
    //     float tmp = std::max(std::min((proposals[picked[i]].rect.x) / scale, (float)(img_w - 1)), 0.f);
    //     // if(tmp > 450 && tmp <650 && proposals[picked[i]].label == 1) //
    //     if(tmp > 0 && tmp <1280 && proposals[picked[i]].label == 1) //
    //     {
    //         mycnt += 1;
    //         selectobjects.push_back(proposals[picked[i]]);
    // }
    // objects.resize(mycnt);
    for (int i = 0; i < count; i++)
    {

        objects[i] = proposals[picked[i]];
        // objects[i] = selectobjects[i];
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        // file << x0 << " " << y0 << " " << x1 - x0 << " " << y1 - y0 << " " << ii << "\n";
        //单个框赋值
        // // ***********************************************************************
        // int d_x = x0 + (x1 - x0) / 2;
        // int d_y = y0 + y1 - y0;

//        int depth_index = (x0 + (x1 - x0) / 2) + (y0 + 1 + y1 - y0)*img_h;
        // double  depth_value = depth_map.at<double>(d_y, d_x) /1000;
        // double  width_value = width_map.at<double>(d_y, d_x) /1000;

        box = objects_msg.add_boxes();
        box->set_xpixel(x0);
        box->set_ypixel(y0);
        box->set_wpixel(x1 - x0);
        box->set_hpixel(y1 - y0);
        box->set_xworld(0);
        box->set_yworld(0);
        box->set_zworld(0);
        box->set_lenreal(0);
        box->set_wreal(0);
        box->set_hreal(0);
        box->set_conf(objects[i].prob);
        box->set_label(objects[i].label);
    }
    // file << "==\n";

}
cv::Mat Yolox::draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects/*, std::string f*/)
{
    cv::Mat image = bgr.clone();

    // CameraObject::CameraBox *box;
    // if(objects.size() == 0)
    // {
    //     objects_msg.set_id(3);
    // }
    // objects_msg.set_id((int)id);
    // unsigned int time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // objects_msg.set_timestamp(time_ms);

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                // obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);
// ***********************************************************************
        // int d_x = obj.rect.x + obj.rect.width / 2;
        // int d_y = obj.rect.y + obj.rect.height;

        // if (d_y > 720)
        // {
        //     d_y = 720;
        // }


        // if (d_x > 1280)
        // {
        //     d_x = 1280;
        // }


        // int depth_index = (obj.rect.x + obj.rect.width / 2) + (obj.rect.y + 1 + obj.rect.height)*image.rows;
        // double  depth_value = depth_map.at<double>(d_y, d_x) /1000;
        // double  width_value = width_map.at<double>(d_y, d_x) /1000;

        //单个框赋值
        // box = objects_msg.add_boxes();
        // box->set_x(obj.rect.x);
        // box->set_y(obj.rect.y);
        // box->set_w(obj.rect.width);
        // box->set_h(obj.rect.height);
        // box->set_prob(objects[i].prob);
        // box->set_depth(depth_value);
        // box->set_real_x(width_value);

        // realwidth = width_value;
        // std::cout << "depth_value = " << depth_value << std::endl;
        // std::cout << "x, y = " << obj.rect.x + obj.rect.width / 2 << ", " << obj.rect.y + 1 + obj.rect.height << std::endl;
        // std::cout << "height =  " << image.rows << std::endl;
        // std::cout << "width =  " << image.cols << std::endl;
// ***********************************************************************
        // char text[256];

        // sprintf(text, "%s %.1f%% %s %.2f%s %s %.2f%s", class_names[obj.label], obj.prob * 100, "depth:", depth_value, "m", "width:", width_value, "m");
        // sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

        cv::String text = cv::format("c:%s , conf: %.2f", class_names[obj.label].c_str() , obj.prob * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y-label_size.height-baseLine), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y - label_size.height+5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);

    }

    return image;
}

void Yolox::doInference(float* input, cv::Size input_shape)
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[minputIndex], input, mBatchSize * 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, mstream));
    (*(mcontext)).enqueue(mBatchSize, buffers, mstream, nullptr);
    CHECK(cudaMemcpyAsync(mprob, buffers[moutputIndex], moutput_size * mBatchSize * sizeof(float), cudaMemcpyDeviceToHost, mstream));
    cudaStreamSynchronize(mstream);
}

void Yolox::release()
{
    // Release stream and buffers
    cudaStreamDestroy(mstream);
    CHECK(cudaFree(buffers[minputIndex]));
    CHECK(cudaFree(buffers[moutputIndex]));

    // destroy the engine
    mcontext->destroy();
    mengine->destroy();
    mruntime->destroy();
}

void Yolox::run(cv::Mat &img, ros::Publisher object_pub/*, std::ofstream & file,int ii*/)
{
//    ROS_INFO("person!!!!!!!!!!!!!!!!!!!1");
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img);
    float* blob;
    blob = blobFromImage(pr_img);
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // auto start = std::chrono::system_clock::now();

    doInference(blob, pr_img.size());
    // auto end0 = std::chrono::system_clock::now();
    std::vector<Object> objects;
    CameraObject::CameraObjects objects_msg;
    decode_outputs(objects_msg, mprob, objects, scale, img_w, img_h/*, file, ii*/);

    //wuhc: add sort code
    deepsort->TrtDetect(img , objects_msg);

//wuhc: show function move to Trtyolosort::showDetection() 

//     if (show_img) //show
//    {
//     cv::Mat wimg = draw_objects(img, objects); // tlwh
//     cv::imshow("res", wimg);
//     cv::waitKey(1);
//     }

    std_msgs::String msg;
    objects_msg.SerializeToString(&msg.data);
    if(msg.data.size() == 0)
    {
        std::cout << "Error in SerializeAsString" << std::endl;
    }
    object_pub.publish(msg);//发布封装完毕的消息msg。Master会查找订阅该话题的节点，并完成两个节点的连接，传输消息;
    // ros::spinOnce();//处理订阅话题的所有回调函数callback()，
//    loop_rate.sleep(); //休眠，休眠时间由loop_rate()设定}cd

//    cv::imwrite("person.png", mPersonOutput);
    // delete the pointer to the float
    delete blob;
//    std::cout << "runrunrun" << std::endl;
// return wimg;
}

