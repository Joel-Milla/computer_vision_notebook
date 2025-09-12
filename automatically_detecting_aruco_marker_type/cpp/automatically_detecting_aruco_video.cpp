// ./automatically_detecting_aruco_video
// or
// ./automatically_detecting_aruco_video -v ../video.mp4

#include <CLI/App.hpp>
#include <CLI/CLI.hpp>
#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>

/**
 * @brief Loops over all possible dictionary of arUCo markers, and finds the
 * best match
 *
 * @param image
 * @return cv::Ptr<cv::aruco::Dictionary>
 */
cv::Ptr<cv::aruco::Dictionary>
detect_aruco_dictionary_matches(const cv::Mat &image) {
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
  // detect arUCo markers

  cv::Ptr<cv::aruco::Dictionary> dict;
  cv::Ptr<cv::aruco::Dictionary> bestMatch;
  std::string bestMatchName;
  int best_num_matches = 0;
  for (const auto [name_dict, type] : dict_map) {

    std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;
    std::vector<int> identifiers;

    dict = cv::aruco::getPredefinedDictionary(type);
    cv::aruco::detectMarkers(image, dict, corners, identifiers, arucoParams,
                             rejectedCandidates);

    if (identifiers.size() > 0)
      printf("[INFO] %zu detected markers for %s\n", identifiers.size(),
             name_dict.c_str());

    if (identifiers.size() > best_num_matches) {
      best_num_matches = identifiers.size();
      bestMatch = dict;
      bestMatchName = name_dict;
    }
  }

  if (bestMatch.empty()) {
    std::cout << "[WARNING] No ArUco dictionary matched.\n";
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  }

  printf("[INFO] Best Match %s\n", bestMatchName.c_str());
  return bestMatch;
}

/**
 * @brief Given an image and dictionary, will draw in the image the markers
 * detected
 *
 * @param image where the matches will be written
 * @param dict
 */
void detect_aruco_image(cv::Mat &image,
                        const cv::Ptr<cv::aruco::Dictionary> &dict) {
  cv::Ptr<cv::aruco::DetectorParameters> arucoParams =
      cv::makePtr<cv::aruco::DetectorParameters>();

  std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;
  std::vector<int> identifiers;
  //* Pass image and dictionary that contains the arUCo markers. Output will be
  // corners of markers, identifiers (ids inside the dictionary), and the false
  // positives
  cv::aruco::detectMarkers(image, dict, corners, identifiers, arucoParams,
                           rejectedCandidates);

  //* If at least one markers was identified, continue
  if (identifiers.size() > 0) {
    //* Loop over the ids
    for (size_t indx = 0; indx < identifiers.size(); indx++) {
      std::vector<cv::Point2f> corner = corners[indx];
      int id = identifiers[indx];

      //* Here we truncate the float numbers to integers (e.g. 10.7 to 10)
      cv::Point topLeft = corner[0];
      cv::Point topRight = corner[1];
      cv::Point bottomRight = corner[2];
      cv::Point bottomLeft = corner[3];

      //* Draw rectangle of marker
      cv::Scalar green(0, 255, 0);
      cv::line(image, topLeft, topRight, green, 2);
      cv::line(image, topRight, bottomRight, green, 2);
      cv::line(image, bottomRight, bottomLeft, green, 2);
      cv::line(image, bottomLeft, topLeft, green, 2);

      //* Draw the center
      cv::Point center;
      center.x = (topLeft.x + bottomRight.x) / 2;
      center.y = (topLeft.y + bottomRight.y) / 2;
      cv::Scalar right(0, 0, 255);
      cv::circle(image, center, 4, right, -1);

      //* Draw text
      cv::putText(image, std::to_string(id),
                  cv::Point(topLeft.x, topLeft.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, green, 2);
    }
  }
}

void detect_aruco_video(const std::string &video_path) {
  cv::Mat frame;
  cv::VideoCapture cap;
  int deviceID = 0;        // 0 = open default camera
  int apiID = cv::CAP_ANY; // 0 = autodetect default API
                           /*
                           The backend tells how the programs talks with the camera. There are many
                           libraries (or APIs, which are libraries that help communicate two ends) that
                           can talk with the camera, such as: DirectShow (Windows) - the most common on
                           Windows, V4L2 (Linux) - standard Linux camera interface, AVFoundation (macOS)
                           - Apple's camera framework.
                         
                           Different libraries have different performances for specific cameras. Some are
                           capable of zoom, focus. Or some libraries have more efficient protocols to
                           talk with specific cameras in specific environments, like more reliably
                           transmitting frames in linux for integrated camera, etc.
                           */

  //* Can either read from video or open live camera
  if (video_path.empty())
    cap.open(deviceID, apiID);
  else
    cap.open(video_path);

  // check if we succeeded
  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera\n";
  }

  //* Grab and write loop
  cv::Ptr<cv::aruco::Dictionary> dict;
  bool dict_detected = false;

  for (;;) {
    cap.read(frame);

    //* check if we succeed
    if (frame.empty()) {
      std::cerr << "[ERROR] Frame not detected";
      break;
    }

    //* Resize for better view
    int new_height = (600 * frame.rows) / frame.cols;
    cv::Size new_size(600, new_height);
    cv::resize(frame, frame, new_size);

    dict = detect_aruco_dictionary_matches(frame);

    //* Obtain the arUCo detection
    detect_aruco_image(frame, dict);

    //* Show images captured
    cv::imshow("Live video", frame);
    if (cv::waitKey(33) >= 0)
      break;
  }

  //* Camera will be deinitialized after destructor
  return;
}

int main(int argc, char **argv) {

  CLI::App app{"detect_aruco_video"};
  std::string video_path;

  app.add_option("-v,--video", video_path,
                 "Video that contains the aruco marker");

  CLI11_PARSE(app, argc, argv);

  detect_aruco_video(video_path);

  return 0;
}