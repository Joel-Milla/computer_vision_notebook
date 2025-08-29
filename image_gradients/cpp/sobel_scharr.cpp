#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
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

void sobel(const std::string &image_path) {
    cv::Mat img = cv::imread(image_path);
    //! Computing gradients, we assume working with grayscale images
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    //! Sobel operator
    cv::Mat gx, gy;
    //* Tell in which direction want to compute the gradient
    //* Work with 32 float as convolutions can be less or greater than [0,255]
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    //* Convert the floating point numbers to 8UC1
    cv::convertScaleAbs(gx, gx);
    cv::convertScaleAbs(gy, gy);

    //* Give equal weight to gx and gy when combining them
    cv::Mat result;
    cv::addWeighted(gx, 0.5, gy, 0.5, 0, result);
    cv::imshow("Sobel", result);

    //? change last parameter (ksize) to -1 for using scharr opeartor
    cv::Sobel(gray, gx, CV_32F, 1, 0, -1);
    cv::Sobel(gray, gy, CV_32F, 0, 1, -1);

    cv::convertScaleAbs(gx, gx);
    cv::convertScaleAbs(gy, gy);
    cv::addWeighted(gx, 0.5, gy, 0.5, 0, result);
    cv::imshow("Scharr", result);
    
    cv::waitKey();
    cv::destroyAllWindows();
    
}

int main(int argc, char **argv) {

    std::string image_path = manual_parsing(argc, argv);
    if (image_path.empty()) {
        std::cout << "Using default path"
                  << std::endl; // flushes automatically

        image_path = "/home/joel/Documents/computer_vision/pyimage/image_gradients/cpp/images/clonazepam_1mg.png";
    }

    sobel(image_path);

    return 0;
}