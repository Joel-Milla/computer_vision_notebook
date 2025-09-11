// ./generating_aruco --id 24 --type DICT_5X5_100 --output
// DICT_5X5_100_id24.png

#include <CLI/App.hpp>
#include <CLI/CLI.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

cv::Ptr<cv::aruco::Dictionary> get_dictionary(const std::string &type) {
  if (type == "DICT_4X4_50")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  if (type == "DICT_4X4_100")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
  if (type == "DICT_4X4_250")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
  if (type == "DICT_4X4_1000")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000);
  if (type == "DICT_5X5_50")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
  if (type == "DICT_5X5_100")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
  if (type == "DICT_5X5_250")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
  if (type == "DICT_5X5_1000")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);
  if (type == "DICT_6X6_50")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
  if (type == "DICT_6X6_100")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_100);
  if (type == "DICT_6X6_250")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  if (type == "DICT_6X6_1000")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000);
  if (type == "DICT_7X7_50")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_50);
  if (type == "DICT_7X7_100")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_100);
  if (type == "DICT_7X7_250")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250);
  if (type == "DICT_7X7_1000")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000);
  if (type == "DICT_ARUCO_ORIGINAL")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
  if (type == "DICT_APRILTAG_16h5")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_16h5);
  if (type == "DICT_APRILTAG_25h9")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_25h9);
  if (type == "DICT_APRILTAG_36h10")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h10);
  if (type == "DICT_APRILTAG_36h11")
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);

  throw std::invalid_argument("Unknown dictionary type: " + type);
}

/*
ArUCo marker can be 4x4, 5x5, etc. And its size correlates to how many bits it
contains. This fiducial markers are surrounded by a black border which makes it
easy for computer vision applications to detect them even when they are rotated,
scaled, etc. Also, by having small squares (bits) it's detection is more robust

A dictionary of markers contains many aruco inside them. Which contains the size
and the internal binaries of the marker. It's a family, where each one has an
id. By having an ArUCo marker, an application can detect it's id from a
family/dictionary. By doing this, you can specifically know the dimensions of
the marker and it's internal properties. Which can then be used to compute the
pose of the camera, and errors in the image for camera correction, etc.
*/
void generating_aruco_markers(const std::string &output, const int &id,
                              const std::string &type) {

  cv::Ptr<cv::aruco::Dictionary> dict = get_dictionary(type);

  cv::Mat markerImage = cv::Mat::zeros(300, 300, CV_8UC1);
  cv::aruco::drawMarker(dict, id, 300, markerImage,
                        1); //* Dictionary with aruco inside

  std::string path = "/home/joel/Documents/computer_vision/pyimage/generating_aruco_markers/cpp/tags/" + output;
  cv::imwrite(path, markerImage); //* Write the image
  cv::imshow("ArUCo Marker", markerImage);
  cv::waitKey();
}

int main(int argc, char **argv) {
  CLI::App app{"Generating aruco markers"};

  // Define options
  std::string output;
  int id;
  std::string type;

  app.add_option("-o,--output", output, "Require an output file")->required();
  app.add_option("-i,--id", id, "ID of ArUCo tag to generate");
  app.add_option("-t,--type", type, "Type of ArUCo tag to generate");

  CLI11_PARSE(app, argc, argv);

  generating_aruco_markers(output, id, type);

  return 0;
}