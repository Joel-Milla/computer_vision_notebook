#include <iostream>
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

void adaptive_equalization(const std::string &image_path) {
    cv::Mat img = cv::imread(image_path);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat clahe;
    //* clip is a threshold, and the size is how many tiles will separate the image, more tiles means that can bust more your noise
    cv::Ptr<cv::CLAHE> clahe_ptr = cv::createCLAHE(2.0, cv::Size(8,8));
    clahe_ptr->apply(gray, clahe);

    cv::imshow("Input", gray);
    cv::imshow("Equalized", clahe);
    cv::waitKey();
    cv::destroyAllWindows();
    
}

int main(int argc, char **argv) {

    std::string image_path = manual_parsing(argc, argv);
    if (image_path.empty()) {
        std::cout << "Using default path"
                  << std::endl; // flushes automatically

        image_path = "/home/joel/Documents/computer_vision/pyimage/histogram_equalization/cpp/images/boston.png";
    }

    adaptive_equalization(image_path);

    return 0;
}