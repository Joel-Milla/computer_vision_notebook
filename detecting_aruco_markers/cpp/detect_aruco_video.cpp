// ./detect_aruco_video -v ../video.mp4 -t DICT_ARUCO_ORIGINAL

#include <CLI/App.hpp>
#include <CLI/CLI.hpp>
#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

/**
 * @brief Get the dictionary object that contains an arUCo family
 *
 * @param type family type
 * @return cv::Ptr<cv::aruco::Dictionary>
 */
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

void detect_aruco_image(cv::Mat &image, const cv::Ptr<cv::aruco::Dictionary> &dict) {
    //* Resize for better view
    int new_height = (600 * image.rows) / image.cols;
    cv::Size new_size(600, new_height);
    cv::resize(image, image, new_size);

    cv::Ptr<cv::aruco::DetectorParameters> arucoParams = cv::makePtr<cv::aruco::DetectorParameters>();

    std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;
    std::vector<int> identifiers;
    //* Pass image and dictionary that contains the arUCo markers. Output will be corners of markers, identifiers (ids inside the dictionary), and the false positives
    cv::aruco::detectMarkers(image, dict, corners, identifiers, arucoParams, rejectedCandidates);

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
            cv::Scalar green(0,255,0);
            cv::line(image, topLeft, topRight, green, 2);
            cv::line(image, topRight, bottomRight, green, 2);
            cv::line(image, bottomRight, bottomLeft, green, 2);
            cv::line(image, bottomLeft, topLeft, green, 2);

            //* Draw the center
            cv::Point center;
            center.x = (topLeft.x + bottomRight.x) / 2;
            center.y = (topLeft.y + bottomRight.y) / 2;
            cv::Scalar right(0,0,255);
            cv::circle(image, center, 4, right, -1);

            //* Draw text
            cv::putText(image, std::to_string(id), cv::Point(topLeft.x, topLeft.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, green, 2);
        }
    }
}

void detect_aruco_video(const std::string &video_path,
                        const std::string &type) {
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
  cap.open(video_path);
  // cap.open(deviceID, apiID);

  // check if we succeeded
  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera\n";
  }

  //* Grab and write loop
  cv::Ptr<cv::aruco::Dictionary> dict = get_dictionary(type);
  for (;;) {
    cap.read(frame);

    //* check if we succeed
    if (frame.empty()) {
      std::cerr << "[ERROR] Frame not detected";
      break;
    }

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
  std::string type;

  app.add_option("-v,--video", video_path, "Video that contains the aruco marker")
      ->required();
  app.add_option("-t,--type", type, "Type of aruco to detect")->required();
  CLI11_PARSE(app, argc, argv);

  detect_aruco_video(video_path, type);

  return 0;
}