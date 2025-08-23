#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

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

void bilateral(const std::string &image_path) {
  //! Bilateral is very useful when you want to blur things that are similar.
  //! When you want to blur but keep the edges intact. Because other types of
  //! blurrings mixed and lose some quality on the edges
  cv::Mat image = cv::imread(image_path);
  cv::imshow("Original", image);

  /*
  Parameters:
  1. img src
  2. diamater: is similar to kernel size, which is to know how many pixels will
  be included
  3. standard deviation: this method, after a certain std threshold, defines
  that the color is too different. Thus, larger std, will have a bigger range of
  colors that will mix up to give the result
  4. sigma space: similar to gaussian blurring, do not wnat the pixels that are
  far off to have the same weight as the kernel center
  */
  std::vector<std::vector<int>> parameters = {
      {11, 21, 7},
      {11, 41, 21},
      {11, 61, 39},
  };

  for (const std::vector<int> &parameter : parameters) {
    cv::Mat blurred;
    cv::bilateralFilter(image, blurred, parameter[0], parameter[1],
                        parameter[2]);

    cv::imshow("Bilateral", blurred);
    cv::waitKey();
  }
}

int main(int argc, char **argv) {

  std::string image_path = manual_parsing(argc, argv);
  if (image_path.empty()) {
    std::cout << "Using default path"
              << std::endl; // endl flushes automatically

    image_path = "/home/joel/Documents/computer_vision/pyimage/"
                 "smoothing_and_blurring/cpp/adrian.png";
  }

  bilateral(image_path);

  return 0;
}