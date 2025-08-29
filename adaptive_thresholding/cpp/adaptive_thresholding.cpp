#include <iostream>
#include <opencv2/core/mat.hpp>
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

void show(std::string name, const cv::Mat &img) {
  cv::imshow(name, img);
  cv::waitKey();
}

//! Otsu thresholding assume that you have a bimodal distribution, that can
//! separate naturally the pixels (like a binary image, have two distinct sets
//! of colors) Adaptive thresholding assumes that smaller regions have similar
//! illumination, and thus apply local thresholding
void adaptive_thresholding(const std::string &image_path) {
  cv::Mat img = cv::imread(image_path);
  cv::imshow("Original", img);
  cv::waitKey();

  //! Preprocessing
  cv::Mat blurred;
  cv::cvtColor(img, blurred, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(blurred, blurred, cv::Size(7, 7), 0);
  cv::imshow("After preprocessing", blurred);
  cv::waitKey();

  //! Global thresholding methods
  cv::Mat thresh_inv;
  cv::threshold(blurred, thresh_inv, 230, 255, cv::THRESH_BINARY_INV);
  show("Thresh inv", thresh_inv);

  cv::Mat otsu_method;
  cv::threshold(blurred, otsu_method, 0, 255,
                cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  show("Otsu method", otsu_method);

  //! Adaptive (local) thresholding
  //* What otsu and other thresholding methods do is that they set (smart or
  //not) a threshold that is applied globally, to every threshold
  //* However, adaptive thresholding have this local, that maybe each 'cuadrant'
  //in an image have different lightning and need a specific threshold to apply
  //to that local
  //! Assumption: smaller regions of an image have uniform ilumination
  cv::Mat adaptive_method;
  //* src, dst, maxValue (similar to other thresh methods), adaptiveMethod (using mean or gaussian to compute threshld t), thresholdType (binary_inv tells that if greater than threhsold 't' found then becomes zero, else max value), block_size (pixel block size to consider the 'local' area. in here is 21x21 pixels), C value (which is the 'C' in the formula t = mean(Li) - C; used to compute threshold)
  cv::adaptiveThreshold(blurred, adaptive_method, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 21, 10);
  cv::imshow("Adaptive mean thresholding", adaptive_method);
  cv::waitKey();

  cv::adaptiveThreshold(blurred, adaptive_method, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 4); //* Change from mean to gaussian, and from C=10 to C=4
  cv::imshow("Adaptive gaussian thresholding", adaptive_method);
  cv::waitKey();

}

int main(int argc, char **argv) {

  std::string image_path = manual_parsing(argc, argv);
  if (image_path.empty()) {
    std::cout << "Using default paths" << std::endl; // flushes automatically

    image_path = "/home/joel/Documents/computer_vision/pyimage/"
                 "adaptive_thresholding/cpp/steve_jobs.png";
  }

  adaptive_thresholding(image_path);

  return 0;
}