#include <cstddef>
#include <cstdio>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

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
Use case:
- When cleaning foreground and background, need to clean the image. Can get some
blobs, and want to clean the image. Can use morphological operators.
- Use this with a binary image, usually with image processing like thresholding.
*/
void morphological_ops(const std::string &image_path) {
  // load image from file
  cv::Mat img = cv::imread(image_path);
  cv::imshow("Original", img);
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  // Can then apply thresholding, edge detection, etc.

  //! Erosion: is like water, it erodes the background and removes the corners,
  //! etc.
  for (int count = 0; count < 3; count++) {
    cv::Mat eroded;
    // anchor Point(-1,-1) tells cv to calculate the default center of kernel
    cv::erode(gray, eroded, cv::Mat(), cv::Point(-1, -1), count + 1);

    std::string winname = "Erosion ";
    winname += std::to_string(count + 1);
    cv::imshow(winname, eroded);
    cv::waitKey(0);
  }

  //* Clean up screen
  clean_up(img);

  //! Dilation: opposite erosion, it grows the foreground. Like going to eye
  //! doctor and he dialates the eyes. Usually apply dialtion after series of
  //! erosion, so the object grows to orignal size but maybe so the corners
  //! don't touch between each other. Maybe to measure the size. So erode to
  //! disconnect noise, and then regrow foreground as close to orignial size,
  //! but not much they actually touch
  for (int count = 0; count < 3; count++) {
    cv::Mat eroded;
    // anchor Point(-1,-1) tells cv to calculate the default center of kernel
    cv::dilate(gray, eroded, cv::Mat(), cv::Point(-1, -1), count + 1);

    std::string winname = "Dilation ";
    winname += std::to_string(count + 1);
    cv::imshow(winname, eroded);
    cv::waitKey(0);
  }

  //* Clean up screen
  clean_up(img);

  //! Opening: erosion + dilation. Usually when want to erode noise or small
  //! blobs, and increase the image again.
  std::vector<cv::Size> sizes = {cv::Size(3, 3), cv::Size(5, 5),
                                 cv::Size(7, 7)};

  for (size_t indx = 0; indx < sizes.size(); indx++) {
    cv::Size size = sizes[indx];
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, size);

    cv::Mat opening;
    cv::morphologyEx(img, opening, cv::MORPH_OPEN, kernel);
    
    std::string winname = "Opening (" + std::to_string(size.height) + ", " + std::to_string(size.width) + ")";
    cv::imshow(winname, opening);
    cv::waitKey(0);
  }

  //* Clean up screen
  clean_up(img);

  //! Closing is exact oppositve of opening. first apply dialtion, then erosion. This can help with closing small holes in the foreground
  for (size_t indx = 0; indx < sizes.size(); indx++) {
    cv::Size size = sizes[indx];
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, size);

    cv::Mat opening;
    cv::morphologyEx(img, opening, cv::MORPH_CLOSE, kernel);
    
    std::string winname = "Closing (" + std::to_string(size.height) + ", " + std::to_string(size.width) + ")";
    cv::imshow(winname, opening);
    cv::waitKey(0);
  }

  //* Clean up screen
  clean_up(img);

  //! Morphological gradient is similar to edge detection, as you get the difference between dilation and erosion. which are just the original borders
  for (size_t indx = 0; indx < sizes.size(); indx++) {
    cv::Size size = sizes[indx];
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, size);

    cv::Mat opening;
    cv::morphologyEx(img, opening, cv::MORPH_GRADIENT, kernel);
    
    std::string winname = "Morphological (" + std::to_string(size.height) + ", " + std::to_string(size.width) + ")";
    cv::imshow(winname, opening);
    cv::waitKey(0);
  }

  //* Clean up screen
  clean_up(img);
}

int main(int argc, char **argv) {

  std::string image_path = manual_parsing(argc, argv);
  if (image_path.empty()) {
    std::cout << "Using default path"
              << std::endl; // endl flushes automatically

    image_path = "/home/joel/Documents/computer_vision/pyimage/"
                 "morphological-operations/cpp/pyimagesearch_logo_noise.png";
  }

  morphological_ops(image_path);

  return 0;
}