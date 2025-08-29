#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
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

//! The purpose of canny edge detection is that after sobel, there is a lot of noise. And we want to remove the noise and effectively create a binary image where the edge is the '255' value, otherwise '0' value
void canny_edge_detection(const std::string &image_path) {
    cv::Mat img = cv::imread(image_path);
    cv::Mat gray, blurred;

    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5,5), -1);

    cv::imshow("Original", img);
    cv::imshow("Blurred", blurred);

    cv::Mat wide, mid, tight;
    cv::Canny(blurred, wide, 80, 200);
    cv::Canny(blurred, mid, 30, 150);
    cv::Canny(blurred, tight, 240, 250);

    cv::imshow("Wide", wide);
    cv::waitKey();
    cv::imshow("Mid", mid);
    cv::waitKey();
    cv::imshow("Tight", tight);
    cv::waitKey();
    cv::destroyAllWindows();
}

int main(int argc, char **argv) {

    std::string image_path = manual_parsing(argc, argv);
    if (image_path.empty()) {
        std::cout << "Using default path"
                  << std::endl; // flushes automatically

        image_path = "/home/joel/Documents/computer_vision/pyimage/edge_detection_canny/cpp/images/clonazepam_1mg.png";
    }

    canny_edge_detection(image_path);

    return 0;
}