#include "deepsort.h"

DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger) {
    YAML::Node config = YAML::LoadFile("../config/config.yaml");
    this->gpuID = gpuID;
    this->enginePath = modelPath;
    this->batchSize = batchSize;
    this->featureDim = featureDim;
    this->imgShape = cv::Size(64, 128);
    this->maxBudget = config["deepsort"]["maxBudget"].as<int>();
    this->maxCosineDist = config["deepsort"]["maxCosineDist"].as<float>();
    this->maxAge = config["deepsort"]["maxAge"].as<int>();
    this->nInit = config["deepsort"]["nInit"].as<int>();
    this->maxIouDist = config["deepsort"]["maxIouDist"].as<float>();
    this->gLogger = gLogger;
    init();
}

void DeepSort::init() {
    objTracker = new tracker(maxCosineDist, maxBudget , maxIouDist , maxAge , nInit);
    featureExtractor = new FeatureTensor(batchSize, imgShape, featureDim, gpuID, gLogger);
    int ret = enginePath.find(".onnx");
    if (ret != -1){
        // featureExtractor->loadOnnx(enginePath);
        std::cout << "DeepSort: find .onnx file , convert to .engine file , wait for a while." << std::endl;
        DeepSortEngineGenerator* engG = new DeepSortEngineGenerator(gLogger);
        std::string save_engine_path = "../config/person/deepsort.engine";
        engG->setFP16(true);
        engG->createEngine(enginePath, save_engine_path);
        std::cout << "==============" << std::endl;
        std::cout << "|  SUCCESS!  |" << std::endl;
        std::cout << "==============" << std::endl;
        featureExtractor->loadEngine(save_engine_path);
    }
    else
        featureExtractor->loadEngine(enginePath);
}

DeepSort::~DeepSort() {
    delete objTracker;
}

void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;

    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
    }
    result.clear();
    results.clear();
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
    }
}

//wuhc: implement of the function definition
void DeepSort::sort(cv::Mat& frame, CameraObject::CameraObjects &detection_msg) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;

    for (CameraObject::CameraBox i : detection_msg.boxes()) {
        //tlwh
        DETECTBOX box(i.xpixel() , i.ypixel() , i.wpixel() , i.hpixel());
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.conf();
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.label(), i.conf()));
    }
    result.clear();
    results.clear();
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }
    // postprocess DETECTION -> Mat
    detection_msg.clear_boxes();
    CameraObject::CameraBox *box;
    for (auto r : result) {
        DETECTBOX i = r.second;

        //按照tlwh保存到detection_msg.boxes()
        box = detection_msg.add_boxes();
        box->set_xpixel(i(0));
        box->set_ypixel(i(1));
        box->set_wpixel(i(2));
        box->set_hpixel(i(3));
        box->set_conf(1.);
        box->set_trackid((float)r.first);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        //保存label 和 conf
        detection_msg.mutable_boxes(i)->set_label(c.cls);
        detection_msg.mutable_boxes(i)->set_conf(c.conf);
    }
}


void DeepSort::sort(cv::Mat& frame, DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        //result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detectionsv2);
        result.clear();
        results.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));
        }
    }
}

void DeepSort::sort(vector<DetectBox>& dets) {
    DETECTIONS detections;
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
    }
    if (detections.size() > 0)
        sort(detections);
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2), i(3), 1.);
        b.trackID = r.first;
        dets.push_back(b);
    }
}

void DeepSort::sort(DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}
