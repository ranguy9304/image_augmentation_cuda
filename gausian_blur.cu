#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
using namespace cv;
using namespace cv::cuda;
using namespace std;



#define KERNEL_RADIUS 9
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)

__constant__ float deviceKernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void gaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float blurValue = 0.0f;
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i) {
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; ++j) {
                int currentX = x + j;
                int currentY = y + i;
                if (currentX >= 0 && currentX < width && currentY >= 0 && currentY < height) {
                    int currentPixelPos = (currentY * width + currentX) * channels + c;
                    blurValue += input[currentPixelPos] * deviceKernel[(i + KERNEL_RADIUS) * KERNEL_SIZE + (j + KERNEL_RADIUS)];
                }
            }
        }
        int outIndex = (y * width + x) * channels + c;
        output[outIndex] = static_cast<unsigned char>(blurValue);
    }
}


// Set Gaussian kernel
void setGaussianKernel(float sigma) {
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    float sumKernel = 0.0f;
    float invTwoSigmaSquare = 0.5f / (sigma * sigma);

    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            float exponent = -(i * i + j * j) * invTwoSigmaSquare;
            kernel[(i + KERNEL_RADIUS) * KERNEL_SIZE + (j + KERNEL_RADIUS)] = expf(exponent);
            sumKernel += kernel[(i + KERNEL_RADIUS) * KERNEL_SIZE + (j + KERNEL_RADIUS)];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        kernel[i] /= sumKernel;
    }

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(deviceKernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
}
void applyGaussianBlur(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels, float sigma) {
    setGaussianKernel(sigma); // Set the Gaussian kernel

    unsigned char *d_inputImage, *d_outputImage;
    size_t imageSize = width * height * channels * sizeof(unsigned char);

    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    gaussianBlur<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height, channels);

    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}