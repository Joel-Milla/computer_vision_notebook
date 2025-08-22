#include <cstdio>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

void clean_up(cv::Mat image) {
  //* Clean up screen
  cv::destroyAllWindows();
  cv::imshow("Original", image);
  cv::waitKey(0);
}

/*
If recieve ./something --image hello.txt
argv[0] == ./something
argv[1] == --image
argv[2] == hello.txt
*/
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

/*
Read this after morphological_ops.cpp
*/
void morphological_hat(const std::string &image_path) {
  // load image from file
  cv::Mat img = cv::imread(image_path);
  cv::waitKey();

  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  cv::imshow("Gray", gray);
  // Can then apply thresholding, edge detection, etc.

  //* We are going to analyze license plates, thus need a kernel that adjust to
  //its shape, which is a shape where needs to be longer than is taller
  cv::Mat kernelRectangle =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 5));

  //! Black hat detects black regions in white backgrounds. Or also, It is the
  //! difference between input image and Opening of the image.
  cv::Mat blackHat;
  cv::morphologyEx(gray, blackHat, cv::MORPH_BLACKHAT, kernelRectangle);
  cv::imshow("Black hat ", blackHat);

  cv::Mat opening;
  cv::morphologyEx(gray, opening, cv::MORPH_OPEN, kernelRectangle);
  cv::imshow("Opening ", opening);
  cv::waitKey();

  //! Top hat (also called white hat) finds light regions in dark backgrounds.
  //! It is the difference between the closing of the input image and input
  //! image.
  cv::Mat whiteHat;
  cv::morphologyEx(gray, whiteHat, cv::MORPH_TOPHAT, kernelRectangle);
  cv::imshow("White hat ", whiteHat);

  cv::Mat closing;
  cv::morphologyEx(gray, closing, cv::MORPH_CLOSE, kernelRectangle);
  cv::imshow("Closing ", closing);
  cv::waitKey();
}

int main(int argc, char **argv) {

  std::string image_path = manual_parsing(argc, argv);
  if (image_path.empty()) {
    std::cout << "Using default path"
              << std::endl; // endl flushes automatically

    image_path = "/home/joel/Documents/computer_vision/pyimage/"
                 "morphological-operations/cpp/car.png";
  }

  morphological_hat(image_path);

  return 0;
}