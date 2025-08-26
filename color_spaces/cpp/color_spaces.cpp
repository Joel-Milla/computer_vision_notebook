#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <string>
#include <utility>
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

void color_spaces(const std::string &image_path) {
  cv::Mat img = cv::imread(image_path);
  cv::imshow("Original", img);
  cv::waitKey();

  std::vector<cv::Mat> bgr_channels;
  cv::split(img, bgr_channels);
  std::vector<std::pair<std::string, cv::Mat>> channels = {
      {"B", bgr_channels[0]},
      {"G", bgr_channels[1]},
      {"R", bgr_channels[2]},
  };

  for (const auto &[name, channel] : channels) {
    cv::imshow(name, channel);
    cv::waitKey();
  }
  cv::destroyAllWindows();

  //! HSV: Hue (the type of color: red, green, blue) - saturation (100%
  //! saturation is pure color, and if not has saturation then is white) - value
  //! (light, how light the color is. 0 in here is black). Good to separate the
  //! light of the image, and easilty detecting colors
  cv::Mat hsv;
  cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> hsv_channels;
  cv::split(hsv, hsv_channels);
  channels = {
      {"H", hsv_channels[0]},
      {"S", hsv_channels[1]},
      {"V", hsv_channels[2]},
  };
  for (const auto &[name, channel] : channels) {
    cv::imshow(name, channel);
    cv::waitKey();
  }
  cv::destroyAllWindows();

  //! The LAB - is heavily used in computation, where it is an esphere where the
  //! z axis is 'L' (of lightness, where -1 black and 1 white), then x axis is
  //! 'A' from green to red [-1,1], and y axis 'B' (blue to yellow). The thing
  //! with lab is that all the colors exists in this sphere, and you can
  //! mathematically say that purple is closer to red and blue, than to green.
  //! This is IMPOSSIBLE to do with other color spaces.
  cv::Mat lab;
  cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
  std::vector<cv::Mat> lab_channels;
  cv::split(hsv, lab_channels);
  channels = {
      {"L", lab_channels[0]},
      {"A", lab_channels[1]},
      {"B", lab_channels[2]},
  };
  for (const auto &[name, channel] : channels) {
    cv::imshow(name, channel);
    cv::waitKey();
  }
  cv::destroyAllWindows();

  //! Grayscale - not color space but constantly being used in image processing
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  cv::imshow("Original", img);
  cv::imshow("Gray", gray);
  cv::waitKey();

  cv::destroyAllWindows();
}

int main(int argc, char **argv) {
  std::string image_path = manual_parsing(argc, argv);
  if (image_path.empty()) {
    std::cout << "Using default path"
              << std::endl; // endl flushes automatically

    image_path = "/home/joel/Documents/computer_vision/pyimage/color_spaces/"
                 "cpp/adrian.png";
  }
  color_spaces(image_path);

  return 0;
}