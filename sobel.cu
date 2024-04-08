#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

__global__ void sobelFilterKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int dx = 0;
    int dy = 0;

    // Sobel kernel for X-direction
    int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    // Sobel kernel for Y-direction
    int sobelY[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int imageX = min(max(x + j, 0), width - 1);
            int imageY = min(max(y + i, 0), height - 1);
            unsigned char pixelValue = inputImage[imageY * width + imageX];
            dx += pixelValue * sobelX[i + 1][j + 1];
            dy += pixelValue * sobelY[i + 1][j + 1];
        }
    }

    int magnitude = abs(dx) + abs(dy);
    outputImage[y * width + x] = min(magnitude, 255);
}


void applySobel(unsigned char* inputImageData, unsigned char* outputImageData ,int width, int height){
    unsigned char* d_inputImage = nullptr;
    unsigned char* d_outputImage = nullptr;
    cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char));

    cudaMemcpy(d_inputImage, inputImageData, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    sobelFilterKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    cudaMemcpy(outputImageData, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
// int main() {
//     string imagePath;
//     cout << "Enter the path to the input image: ";
//     cin >> imagePath;

//     Mat image = imread(imagePath, IMREAD_GRAYSCALE);
//     if (image.empty()) {
//         cout << "Failed to load the image." << endl;
//         return -1;
//     }

//     int width = image.cols;
//     int height = image.rows;

//     unsigned char* inputImageData = image.data;
//     // unsigned char* outputImageData = new unsigned char[width * height];
//     cv::Mat sobel(image.rows, image.cols, image.type());

//     applySobel(inputImageData, sobel.data, width, height);


//     string outputPath = "asdas.jpg";
//     imwrite(outputPath, sobel);

//     cout << "Sobel result saved as: " << outputPath << endl;

//     // cudaFree(d_inputImage);
//     // cudaFree(d_outputImage);
//     // delete[] outputImageData;

//     return 0;
// }