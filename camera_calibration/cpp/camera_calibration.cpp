#include <CLI/App.hpp>
#include <CLI/CLI.hpp>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

void camera_calibration(const std::string &folder_path) {
    //* Get files
    std::vector<std::string>files;
    for (const auto& entry : std::filesystem::directory_iterator(folder_path))
        files.push_back(entry.path().string());
}

int main(int argc, char **argv) {

    CLI::App app{"camera_calibration"};
    std::string folder_path;
    app.add_option("-f,--folder", folder_path, "Requires an image file")->required();
    CLI11_PARSE(app, argc, argv);


    camera_calibration(folder_path);

    return 0;
}