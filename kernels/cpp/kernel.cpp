#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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

std::vector<std::pair<std::string, Eigen::MatrixXf>> get_kernelBank() {
  Eigen::MatrixXf smallBlur = Eigen::MatrixXf::Constant(7, 7, 1.0f / 49.0f);
  Eigen::MatrixXf bigBlur =
      Eigen::MatrixXf::Constant(21, 21, 1.0f / (21.0f * 21.0f));

  // Sharpening kernel - enhances edges and details
  Eigen::MatrixXf sharpen(3,3);
  sharpen << 0, -1, 0, -1, 5, -1, 0, -1, 0;
  /* Matrix
  [ 0 -1  0]
  [-1  5 -1]
  [ 0 -1  0]
  */

  // Laplacian kernel - detects edge-like regions
  Eigen::MatrixXf laplacian(3,3);
  laplacian << 0, 1, 0, 1, -4, 1, 0, 1, 0;
  /* Matrix
  [ 0  1  0]
  [ 1 -4  1]
  [ 0  1  0]
  */

  // Sobel x-axis kernel - detects vertical edges
  Eigen::MatrixXf sobelX(3,3);
  sobelX << -1, 0, 1, -2, 0, 2, -1, 0, 1;
  /* Matrix:
  [-1  0  1]
  [-2  0  2]
  [-1  0  1]
  */

  // Sobel y-axis kernel - detects horizontal edges
  Eigen::MatrixXf sobelY(3,3);
  sobelY << -1, -2, -1, 0, 0, 0, 1, 2, 1;
  /* Matrix
  [-1 -2 -1]
  [ 0  0  0]
  [ 1  2  1]
  */

  std::vector<std::pair<std::string, Eigen::MatrixXf>> kernelBank{
      {"small_blur", smallBlur}, {"large_blur", bigBlur}, {"sharpen", sharpen},
      {"laplacian", laplacian},  {"sobel_x", sobelX},     {"sobel_y", sobelY},
  };

  return kernelBank;
}

cv::Mat convolution(cv::Mat image, Eigen::MatrixXf eigen_kernel) {
  //* Height/width of image/kernel
  int iH = image.rows;
  int iW = image.cols;

  int kH = eigen_kernel.rows();
  int kW = eigen_kernel.cols();
  

  int pad = (kW - 1) / 2;
  cv::copyMakeBorder(image, image, pad, pad, pad, pad, cv::BORDER_REPLICATE);

  cv::Mat output = cv::Mat::zeros(iH, iW, CV_8UC1);
  // Eigen::MatrixXf output(iH, iW);
  Eigen::MatrixXf eigen_output(iH, iW);

  /*
  - If originally had 6x6 matrix and apply 5x5 kernel. By border replication,
  new img = 8x8 matrix (applying padding)
  - Then, you want to start in (2,2) original starting place of image, and then
  iterate until (2, 8) which is the 6th position of the original image. You want
  to do that from the row 2, to the row 8 (which was you original ending row)
  */
  for (int y = pad; y < (iH + pad); y++) {
    for (int x = pad; x < (iW + pad); x++) {
      // std::cout << image.size << std::endl;
      // std::cout << eigen_kernel.size() << std::endl;
      // std::cout << y << " " << y+kH << std::endl;
      // std::cout << x << " " << x+kW << std::endl;
      //* (x,y) is the current center. Need to get the range from previous pixels and next pixels (remembering rhs is exclusive)
      cv::Mat slice =
          image(cv::Range(y - pad, y + pad + 1), cv::Range(x - pad, x + pad + 1));

      Eigen::MatrixXf eigen_slice;
      cv::cv2eigen(slice, eigen_slice);

      Eigen::MatrixXf result = eigen_kernel.cwiseProduct(eigen_slice);
      float sum = result.sum();
      //* The result of the sum can be sum<0 or sum>255. So we clip so that the range is between [0, 255]
      sum = std::max(0.0f, std::min(255.0f, sum));
      eigen_output(y - pad, x - pad) = sum;
    }
  }

  //* Convert eigen values to output matrix
  cv::eigen2cv(eigen_output, output);
  output.convertTo(output, CV_8UC1); //* convert to unsigned integers

  return output;
}

/*
Sobel/gaussian/etc they have predefined kernels that were handmade. You know
that after applying those, you get a specific result. With CNN, the NN learns
what filters, what kernels to apply to learn more richer features. Can stack
them and then applies filters at many layers to stack those opearations that are
complex
*/
void kernels(const std::string &image_path) {
  cv::Mat img = cv::imread(image_path);
  std::vector<std::pair<std::string, Eigen::MatrixXf>> kernelBank = get_kernelBank();
  
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

  for (const auto& [name, kernel] : kernelBank) {
    cv::Mat kernel_cv;
    cv::eigen2cv(kernel, kernel_cv);

    cv::Mat convolve_output = convolution(gray, kernel);
    cv::Mat opencv_output;
    cv::filter2D(gray, opencv_output, -1, kernel_cv); //* -1 tells opencv to automatically calculate depth of image

    cv::imshow("original", gray);
    cv::imshow(name, convolve_output);
    cv::imshow("opencv", opencv_output);
    cv::waitKey();
    cv::destroyAllWindows();
  }
}

int main(int argc, char **argv) {

  std::string image_path = manual_parsing(argc, argv);
  if (image_path.empty()) {
    std::cout << "Using default path" << std::endl; // flushes automatically

    image_path = "/home/joel/Documents/computer_vision/pyimage/kernels/cpp/"
                 "3d_pokemon.png";
  }

  kernels(image_path);

  return 0;
}