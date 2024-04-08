#include <dirent.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "gausian_blur.cu"
#include "color.cu"
#include "rotate.cu"
#include "up.cu"
#include "sobel.cu"

using namespace cv;
using namespace cv::cuda;
using namespace std;

int main() {
    string inputDir, outputDir;
    cout << "Enter the path to the input directory: ";
    cin >> inputDir;
    // cout << "Enter the path to the output directory: ";
    // cin >> outputDir;

    DIR* dir;
    struct dirent* ent;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if ((dir = opendir(inputDir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            string fileName = ent->d_name;
            if (fileName.find(".jpg") != string::npos || fileName.find(".png") != string::npos) {
                string inputPath = inputDir + "/" + fileName;
                string outputPath = outputDir + "/blurred_" + fileName;

                Mat image = imread(inputPath, IMREAD_UNCHANGED);
                if (!image.empty()) {
                    cv::Mat blurredImage = cv::Mat::zeros(image.size(), image.type());

                    float sigma = 1.0f; // Standard deviation for Gaussian kernel
                    applyGaussianBlur(image.data, blurredImage.data, image.cols, image.rows, image.channels(), sigma);

                    imwrite("blur/blur_" + fileName, blurredImage);
                    // cout << "Blurred image saved: " << outputPath << endl;

                    double maxRotationAngle = 45.0;
                    cv::Mat rotatedImage(image.rows, image.cols, image.type());
                    applyRandomRotation(image.data, rotatedImage.data, image.cols, image.rows, image.channels(), maxRotationAngle);
                    cv::imwrite("rotation/rot_" + fileName, rotatedImage);

                    int maxColorShift = 50;
                    cv::Mat colorShiftedImage(image.rows, image.cols, image.type());
                    applyRandomColorShift(image.data, colorShiftedImage.data, image.cols, image.rows, image.channels(), maxColorShift);
                    cv::imwrite("color/color_" + fileName, colorShiftedImage);

                    int scaleFactor = 2;
                    int inputWidth = image.cols;
                    int inputHeight = image.rows;
                    int outputWidth = inputWidth * scaleFactor;
                    int outputHeight = inputHeight * scaleFactor;
                    int channels = image.channels();
                    cv::Mat outputImage(outputHeight, outputWidth, image.type());
                    upsampleImage(image.data, outputImage.data, inputWidth, inputHeight, channels, scaleFactor);
                    cv::imwrite("uped/uped_" + fileName, outputImage);

                    cv::Mat grayImage;
                    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
                    cv::Mat sobel(grayImage.rows, grayImage.cols, grayImage.type());
                    applySobel(grayImage.data, sobel.data, inputWidth, inputHeight);
                    cv::imwrite("sobel/sobel_" + fileName, sobel);
                }
            }
        }
        closedir(dir);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Total augmentation time: " << milliseconds << " milliseconds" << endl;

    return 0;
}