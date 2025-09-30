#include <CLI/App.hpp>
#include <CLI/CLI.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <vector>

void camera_calibration(const std::string &folder_path) {
  //* Get files
  std::vector<std::string> files;
  for (const auto &entry : std::filesystem::directory_iterator(folder_path))
    files.push_back(entry.path().string());

  int num_cols = 8;
  int num_rows = 6;

  //* Termination criteria
  // Which tells the algorithm to stop after 30 iterations, or after improvement is less than 0.001
  cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);

  //* Data to do calibration
  std::vector<std::vector<cv::Vec3f>> objpoints;
  std::vector<std::vector<cv::Vec2f>> imgpoints;
  cv::Size img_size(1280, 720);
  cv::Size checkboard_size = cv::Size(num_cols, num_rows);

  //* objp its a vector of (0,0,0),(1,0,0),(2,0,0), etc which goes for all rows and all cols. In our ideal world, these are the coordinates of each point. Where one unit, could be in real world 0.55cm or whatever. but the ratio should be always stay the same
  float square_size = 0.55;
  std::vector<cv::Vec3f> objp;
  for (int i = 0; i < num_rows; i++) {    // Rows (y-axis)
    for (int j = 0; j < num_cols; j++) {  // Columns (x-axis)
      objp.push_back(cv::Vec3f(j*square_size, i*square_size, 0)); // (x, y, z=0) where z stays the same, ideally, and camera is the one that changes
    }
  }

  //* Traverse vector
  for (const auto &file : files) {
    cv::Mat image = cv::imread(file);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    //* Find corners of chessboard
    std::vector<cv::Vec2f> corners;

    bool corners_found = cv::findChessboardCorners(
        gray, cv::Size(num_cols, num_rows),
        corners); // Value returned tells whether checkboard with this
                  // specifications was found or not

    if (corners_found) {
      objpoints.push_back(objp);
      // the cornerSubPixe converst the coordinate (100,100) to (100.31, 100.17), a more precise corner by analyzing gradient around it
      cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                       criteria);
      imgpoints.push_back(corners);

      cv::drawChessboardCorners(image, checkboard_size, corners, corners_found);
      // cv::imshow("Image", image);
      // cv::waitKey(500);
    }

    // cv::destroyAllWindows();
  }

  //* Calibration matrix
  cv::Mat camera_matrix;
  cv::Mat dist_coeff;

  // Rotation and translation vectors
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;

  //* This internally soves: P_2d = K [R | t] P_3d
  //* Uses non-linear least squares method to optimize
  cv::calibrateCamera(objpoints, imgpoints, img_size, camera_matrix, dist_coeff,
                      rvecs, tvecs);

  std::cout << "Camera Matrix:\n" << camera_matrix << std::endl;
}

int main(int argc, char **argv) {

  CLI::App app{"camera_calibration"};
  std::string folder_path;
  app.add_option("-f,--folder", folder_path, "Requires an image file")
      ->required();
  CLI11_PARSE(app, argc, argv);

  camera_calibration(folder_path);

  return 0;
}