#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void upsampleKernel(unsigned char *inputImage, unsigned char *outputImage, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputWidth && y < outputHeight) {
        float scaleX = static_cast<float>(inputWidth) / outputWidth;
        float scaleY = static_cast<float>(inputHeight) / outputHeight;

        int inputX = static_cast<int>(x * scaleX);
        int inputY = static_cast<int>(y * scaleY);

        int inputIndex = (inputY * inputWidth + inputX) * channels;
        int outputIndex = (y * outputWidth + x) * channels;

        for (int c = 0; c < channels; c++) {
            outputImage[outputIndex + c] = inputImage[inputIndex + c];
        }
    }
}

void upsampleImage(unsigned char *inputImage, unsigned char *outputImage, int inputWidth, int inputHeight, int channels, int scaleFactor) {
    int outputWidth = inputWidth * scaleFactor;
    int outputHeight = inputHeight * scaleFactor;
    

    unsigned char *d_inputImage, *d_outputImage;
    size_t inputImageSize = inputWidth * inputHeight * channels * sizeof(unsigned char);
    size_t outputImageSize = outputWidth * outputHeight * channels * sizeof(unsigned char);

    cudaMalloc(&d_inputImage, inputImageSize);
    cudaMalloc(&d_outputImage, outputImageSize);

    cudaMemcpy(d_inputImage, inputImage, inputImageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x, (outputHeight + blockDim.y - 1) / blockDim.y);

    upsampleKernel<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, inputWidth, inputHeight, outputWidth, outputHeight, channels);

    cudaMemcpy(outputImage, d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    
}

// int main(int argc, char *argv[]) {
//     if (argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <image_path> <scale_factor>" << std::endl;
//         return 1;
//     }

//     std::string imagePath = argv[1];
//     int scaleFactor = std::stoi(argv[2]);


//     cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);

//     if (inputImage.empty()) {
//         std::cerr << "Failed to open image: " << imagePath << std::endl;
//         return;
//     }

//     int inputWidth = inputImage.cols;
//     int inputHeight = inputImage.rows;
//     int channels = inputImage.channels();

//     int outputWidth = inputWidth * scaleFactor;
//     int outputHeight = inputHeight * scaleFactor;

//     cv::Mat outputImage(outputHeight, outputWidth, inputImage.type());

//     upsampleImage(inputImage.data,outputImage.data,inputWidth,inputHeight,channels, scaleFactor);
//     cv::imwrite("upsampled_image.jpg", outputImage);
//     std::cout << "Upsampled image saved as 'upsampled_image.jpg'" << std::endl;
//     return 0;
// }