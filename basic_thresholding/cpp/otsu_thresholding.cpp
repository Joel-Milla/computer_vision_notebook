#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

std::string manual_parsing(int argc, char **argv) {
    std::string image_path;

    // Simple manual parsing
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-i" || std::string(argv[i]) == "--image") {
            if (i + 1 < argc) {
                image_path = argv[i + 1];
                i++; // skip next argument
            }
        }
    }

    return image_path;
}

void otsu_thresholding(const std::string &image_path) {
    cv::Mat img = cv::imread(image_path);
    cv::imshow("Original", img);
    cv::waitKey();

    //! Preprocessing
    //* Need grayscal as thresholding only receive on grayscale
    //* Blurring to smoothen image and have less noise so is easier for algorithms to detect the single coin. Because do not care on detecting the specific details inside the coin, only to detect coin from the background
    cv::Mat gray;
    cv::Mat blurred;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(7,7), 0);

    //! Otsu thresholding
    //* What it does, is that it tries all the possible tresholds, and selects the one that better (maximize) separates the background and foreground. That best separates into two homogeneous classes
    cv::Mat otsu_thresh;
    //* @thresh value will be ignored, and then provide otsu's flag
    double otsu_value = cv::threshold(blurred, otsu_thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    std::string otsu_name = "Otsu threshold: " + std::to_string(otsu_value);
    cv::imshow(otsu_name, otsu_thresh);
    cv::waitKey();

    //! Apply mask
    //* Use what we obtained to get a mask
    cv::Mat masked;
    cv::bitwise_and(img, img, masked, otsu_thresh);
    cv::imshow("Masked", masked);
    cv::waitKey();

    cv::destroyAllWindows();
}

int main(int argc, char **argv) {

    std::string image_path = manual_parsing(argc, argv);
    if (image_path.empty()) {
        std::cout << "Using default path"
                  << std::endl; // flushes automatically

        image_path = "/home/joel/Documents/computer_vision/pyimage/basic_thresholding/cpp/images/coins01.png";
    }

    otsu_thresholding(image_path);

    return 0;
}