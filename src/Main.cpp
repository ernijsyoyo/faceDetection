#include <opencv4/opencv2/opencv.hpp>
#include "HelloWorld.hpp"
#include <opencv4/opencv2/imgcodecs.hpp>
#include <dirent.h>
#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include "FeatureExtraction.hpp"
#include "dlibWrapper.hpp"

void GetAndPrepareDataStructures(std::vector<cv::Mat> &images, std::vector<string> &labels){
    auto image_path = "../trainingData/data/";
    struct dirent *entry = nullptr;
    DIR *dp = nullptr;

    std::cout << "Opening directory: " << image_path << std::endl;
    

    dp = opendir(image_path);
    if (dp != nullptr) {
        while ((entry = readdir(dp))) {
            auto fileName = std::string(entry->d_name);
            auto fileLocation = std::string(image_path) + fileName;
            auto img = cv::imread(fileLocation, cv::IMREAD_COLOR);
            
            // Continue to next iteration and dont bother with labels if img is empty
            if(!img.empty()){
                images.push_back(img);
            } else {
                continue;
            }
            std::string label = fileName.substr(0, fileName.find("_", 0));
            if (!std::count(labels.begin(), labels.end(), label)) {
                labels.push_back(label);
            }
        }
    }
    closedir(dp);
}

int main(int argc, char **argv) {

    
    // // Prepare our binaries which will contain labels and images of our dataset
    // std::vector<cv::Mat> images;
    // std::vector<string> labels;
    // GetAndPrepareDataStructures(images, labels);
    // FeatureExtraction::TrainAndSerializeDataset(images, labels);
    
    // Load in an image to that we wish to recognise
    auto static_path = "../testData/Frida6.jpeg";
    cv::Mat frame;
    frame = cv::imread(static_path, cv::IMREAD_COLOR);
    if(frame.empty()){
        std::cout << "Image not found in path: " << static_path << std::endl;    
    }
    std::cout << "Loading image " << static_path << std::endl;    
    dLibWrapper::Analyse(frame);


    // FaceDetector face_detector;

    // cv::VideoCapture video_capture;
    // if (!video_capture.open(0)) {
    //     return 0;
    // }

    // cv::Mat frame;
    // // cv::Mat ernie;
    // // cv::Mat frida;
    // // cv::Mat dLibInput;

    

    // cin.get();

    // while (true) {
    //     video_capture >> frame;
        
    //     if(frame.empty()){
    //         continue;
    //     }
        

    //     auto rectangles = face_detector.detect_face_rectangles(frame);
    //     cv::Scalar color(0, 105, 205);
    //     for(const auto & r : rectangles){
    
    //         cv::rectangle(frame, r, color, 4);
    //     }

    //     //std::cout << "Breaking" << std::endl;
    //     int esc_key = 27;
    //     break;
    // }


    // cv::destroyAllWindows();
    // video_capture.release();

    return 0;
}