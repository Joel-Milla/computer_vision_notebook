// ./guess_aruco_type -i ../images/example_01.png

#include <CLI/App.hpp>
#include <CLI/CLI.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

/**
 * @brief Get the dictionary object that contains an arUCo family
 *
 * @param type family type
 */
void detect_aruco_dictionary_matches(const cv::Mat &image) {
  static const std::vector<
      std::pair<std::string, cv::aruco::PREDEFINED_DICTIONARY_NAME>>
      dict_map = {{"DICT_4X4_50", cv::aruco::DICT_4X4_50},
                  {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
                  {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
                  {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
                  {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
                  {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
                  {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
                  {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
                  {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
                  {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
                  {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
                  {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
                  {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
                  {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
                  {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
                  {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
                  {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
                  {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
                  {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
                  {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
                  {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11}};
  cv::Ptr<cv::aruco::DetectorParameters> arucoParams =
      cv::makePtr<cv::aruco::DetectorParameters>(); //* General parameters to
                                                    //detect arUCo markers
  cv::Ptr<cv::aruco::Dictionary> dict;

  for (const auto [name_dict, type] : dict_map) {

    std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;
    std::vector<int> identifiers;

    dict = cv::aruco::getPredefinedDictionary(type);
    cv::aruco::detectMarkers(image, dict, corners, identifiers, arucoParams,
                             rejectedCandidates);

    if (identifiers.size() > 0)
      printf("[INFO] %zu detected markers for %s\n", corners.size(),
             name_dict.c_str());
  }
}

void guess_aruco_type(const std::string &image_path) {
  cv::Mat image = cv::imread(image_path);
  //* Resize for better view. Careful in here if arUCo marker is small, and by
  // resizing it makes it even smaller which then is difficult to detect the
  // marker
  int new_height = (600 * image.rows) / image.cols;
  cv::Size new_size(600, new_height);
  cv::resize(image, image, new_size);

  detect_aruco_dictionary_matches(image);
  cv::imshow("Image", image);
  cv::waitKey();
}

int main(int argc, char **argv) {

  CLI::App app{"guess_aruco_type"};
  std::string image_path;
  app.add_option("-i,--image", image_path, "Requires an image file")
      ->required();
  CLI11_PARSE(app, argc, argv);

  guess_aruco_type(image_path);

  return 0;
}