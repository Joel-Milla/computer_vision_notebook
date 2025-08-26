#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

void simple_thresholding(const std::string &image_path) {
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

    //! Thresholding
    cv::Mat threshold_bin_inv;
    //* @source, @dst, @thresh (if greater than this, converts to zero), @maxval (if previous false, then this value is the one set), @type (controls the behavior of what occurs when @thresh comparison happens)
    cv::threshold(blurred, threshold_bin_inv, 200, 255, cv::THRESH_BINARY_INV);
    //* @maxval set to 255 as usually want the foreground to be white    
    cv::imshow("Threshold binary inverse", threshold_bin_inv);
    cv::waitKey();

    cv::Mat threshold_bin_inv_closed;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::morphologyEx(threshold_bin_inv, threshold_bin_inv_closed, cv::MORPH_CLOSE, kernel);
    cv::imshow("Threshold binary inverse closed", threshold_bin_inv_closed);
    cv::waitKey();

    //* If greater than 200, then specific pixel set to white (255). 
    cv::Mat threshold_bin;
    cv::threshold(blurred, threshold_bin, 200, 255, cv::THRESH_BINARY);
    cv::imshow("Threshold binary", threshold_bin);
    cv::waitKey();

    //! Apply mask
    //* Use what we obtained to get a mask
    cv::Mat masked;
    cv::bitwise_and(img, img, masked, threshold_bin_inv);
    cv::imshow("Masked", masked);
    cv::waitKey();

    cv::Mat masked_closed;
    cv::bitwise_and(img, img, masked_closed, threshold_bin_inv_closed);
    cv::imshow("Masked closed", masked_closed);
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

    simple_thresholding(image_path);

    return 0;
}