#include <iostream>
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

/*
How histogram equalization works? 
https://www.youtube.com/watch?v=WuVyG4pg9xQ

the graph below, tells the cumulative frequency. Like the point in the graph 100 has a y_value of 0.4. meaning that all points that have intensity 100 or below, will constitute 40% of the intensities in the image. 

By equalizing, you are just making that as you increase your intensity, you uniformly incrementing percentage of the total values considering. Thus, almost all the intensities happen around the same number of times. just multiply it times the real proportion you have and you almost perfectly distribute the intensity values

Limitations is that applies it globally, and can also contrast too much the noise. Adaptive avoids that by separating image in tiles, and applying a function inside each cells. 

If 60% of pixels are at or below intensity 100, then in a "perfect" uniform distribution, those pixels should map to 60% of the intensity range.
*/
void histogram_equalization(const std::string &image_path) {
    cv::Mat img = cv::imread(image_path);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("Gray", gray);
    cv::waitKey();

    cv::Mat equalized;
    cv::equalizeHist(gray, equalized);

    cv::imshow("Equalized", equalized);
    cv::waitKey();
    cv::destroyAllWindows();
}

int main(int argc, char **argv) {

    std::string image_path = manual_parsing(argc, argv);
    if (image_path.empty()) {
        std::cout << "Using default path"
                  << std::endl; // flushes automatically

        image_path = "/home/joel/Documents/computer_vision/pyimage/histogram_equalization/cpp/images/boston.png";
    }

    histogram_equalization(image_path);

    return 0;
}