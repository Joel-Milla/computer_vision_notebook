#include <Eigen/Dense>
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

cv::Mat convolution(cv::Mat image, cv::Mat kernel) {
  //* Convert to eigen kernel
  Eigen::MatrixXf eigen_kernel;
  cv::cv2eigen(kernel, eigen_kernel);

  //* Height/width of image/kernel
  int iH = image.rows;
  int iW = image.cols;

  int kH = kernel.rows;
  int kW = kernel.cols;

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
      cv::Mat slice =
          image(cv::Range(y, y + kernel.rows), cv::Range(x, x + kernel.cols));
      Eigen::MatrixXf eigen_slice;
      cv::cv2eigen(slice, eigen_slice);

      Eigen::MatrixXf result = eigen_kernel.cwiseProduct(eigen_slice);
      float sum = result.sum();

      eigen_output(y, x) = sum;
    }
  }

  //* Convert eigen values to output matrix
  cv::eigen2cv(eigen_output, output);

  //* Normalize to [0, 255]
  cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);

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

  Eigen::MatrixXf smallBlur = Eigen::MatrixXf::Constant(7, 7, 1.0f / 49.0f);
  Eigen::MatrixXf bibBlur =
      Eigen::MatrixXf::Constant(21, 21, 1.0f / (21.0f * 21.0f));

  // Sharpening kernel - enhances edges and details
  Eigen::Matrix3i sharpen;
  sharpen << 0, -1, 0, -1, 5, -1, 0, -1, 0;
  /* Matrix
  [ 0 -1  0]
  [-1  5 -1]
  [ 0 -1  0]
  */

  // Laplacian kernel - detects edge-like regions
  Eigen::Matrix3i laplacian;
  laplacian << 0, 1, 0, 1, -4, 1, 0, 1, 0;
  /* Matrix
  [ 0  1  0]
  [ 1 -4  1]
  [ 0  1  0]
  */

  // Sobel x-axis kernel - detects vertical edges
  Eigen::Matrix3i sobelX;
  sobelX << -1, 0, 1, -2, 0, 2, -1, 0, 1;
  /* Matrix:
  [-1  0  1]
  [-2  0  2]
  [-1  0  1]
  */

  // Sobel y-axis kernel - detects horizontal edges
  Eigen::Matrix3i sobelY;
  sobelY << -1, -2, -1, 0, 0, 0, 1, 2, 1;
  /* Matrix
  [-1 -2 -1]
  [ 0  0  0]
  [ 1  2  1]
  */
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