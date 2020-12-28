#include <opencv4/opencv2/opencv.hpp>
#include "HelloWorld.hpp"
#include <opencv4/opencv2/imgcodecs.hpp>
#include <dirent.h>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include "FeatureExtraction.hpp"
//#include "dlibWrapper.hpp"

int main(int argc, char **argv) {

    cv::VideoCapture video_capture;
    if (!video_capture.open(0)) {
        return 0;
    }


    // auto image_path = "../trainingData/Ernie/";
    // struct dirent *entry = nullptr;
    // DIR *dp = nullptr;

    // std::cout << "Opening directory: " << image_path << std::endl;
    // std::vector<cv::Mat> images;
    // dp = opendir(image_path);
    // if (dp != nullptr) {
    //     while ((entry = readdir(dp))){
    //         auto fileLocation = std::string(image_path) + std::string(entry->d_name);
    //         auto img = cv::imread(fileLocation, cv::IMREAD_COLOR);
    //         if(!img.empty()){
    //             images.push_back(img);
    //         }
    //     }
    // }
    // closedir(dp);

    //FeatureExtraction::TrainAndSerializeDataset(images);
    
    
    FaceDetector face_detector;

    cv::Mat frame;
    cv::Mat ernie;
    cv::Mat frida;
    cv::Mat dLibInput;

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    while (true) {
        video_capture >> frame;
        dLibWrapper::Analyse(img);

        auto rectangles = face_detector.detect_face_rectangles(frame);
        cv::Scalar color(0, 105, 205);
        for(const auto & r : rectangles){
    
            cv::rectangle(frame, r, color, 4);
        }

        //std::cout << "Breaking" << std::endl;
        int esc_key = 27;
        break;
    }


    cv::destroyAllWindows();
    video_capture.release();

    return 0;
}