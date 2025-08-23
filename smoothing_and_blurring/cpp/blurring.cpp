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

/*
Use Case:
- Blurring is when pixels are mixed with its surrounding. Its not clear.
- Higher precision confuses more computers, as higher precision include more
noise in the image.
- With this avoid processing the noise that we do not care about
- Also remove the extra detail that we don't care about. Allow to focus on more
largere structural objects in image
*/
void blurring(const std::string &image_path) {
  cv::Mat img = cv::imread(image_path);
  cv::imshow("og", img);
  cv::waitKey();

  //* Larger kernel gets, bigger the blurring
  std::vector<cv::Size> sizes = {
      cv::Size(3, 3),
      cv::Size(9, 9),
      cv::Size(15, 15),
  };

  for (size_t indx = 0; indx < sizes.size(); indx++) {
    const cv::Size &size = sizes[indx];
    cv::Mat blurred;
    cv::blur(img, blurred, size);

    std::string winname = "Average (" + std::to_string(size.width) + ", " +
                          std::to_string(size.height) + ")";
    cv::imshow(winname, blurred);
    cv::waitKey();
  }

  cv::destroyAllWindows();
  cv::imshow("Original", img);

  //! The gaussian blurring is an average, where more empahisis is placed in the
  //! center. A weighted average so closer to center have more 'weight' in the
  //! final result. So is more a natural blur
  for (size_t indx = 0; indx < sizes.size(); indx++) {
    const cv::Size &size = sizes[indx];
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, size,
                     0); //* 0 value tells opencv to automatically compute sigma
                         //value depdending on kernel size

    std::string winname = "Gaussian (" + std::to_string(size.width) + ", " +
                          std::to_string(size.height) + ")";
    cv::imshow(winname, blurred);
    cv::waitKey();
  }
  cv::destroyAllWindows();

  //! With methods above, you can have rectangular matrices (along w/h are odd) but median blurring requires an int, as they require a square matrix
  //! Really good to resolve salt and paper noise
  std::vector<int> sizes_int = {3, 9, 15};
  cv::imshow("og: ", img);
  for (const int& size : sizes_int) {
    cv::Mat blurred;
    cv::medianBlur(img, blurred, size);

    std::string winname = "Median with: " + std::to_string(size);
    cv::imshow(winname, blurred);
    cv::waitKey();
  }
  cv::destroyAllWindows();
}

int main(int argc, char **argv) {

  std::string image_path = manual_parsing(argc, argv);
  if (image_path.empty()) {
    std::cout << "Using default path"
              << std::endl; // endl flushes automatically

    image_path = "/home/joel/Documents/computer_vision/pyimage/"
                 "smoothing_and_blurring/cpp/adrian.png";
  }

  blurring(image_path);

  return 0;
}