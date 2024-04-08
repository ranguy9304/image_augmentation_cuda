

#include <iostream>
#include <opencv2/opencv.hpp>


__global__ void rotateImage(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels, double angle)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        double radians = angle * CV_PI / 180.0;
        double sinTheta = sin(radians);
        double cosTheta = cos(radians);

        double centerX = width / 2.0;
        double centerY = height / 2.0;

        int sourceX = static_cast<int>((x - centerX) * cosTheta - (y - centerY) * sinTheta + centerX);
        int sourceY = static_cast<int>((x - centerX) * sinTheta + (y - centerY) * cosTheta + centerY);

        if (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height)
        {
            int sourceIndex = (sourceY * width + sourceX) * channels;
            int destIndex = (y * width + x) * channels;

            for (int c = 0; c < channels; c++)
            {
                outputImage[destIndex + c] = inputImage[sourceIndex + c];
            }
        }
    }
}

void applyRandomRotation(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels, double maxAngle)
{
    double angle = cv::theRNG().uniform(-maxAngle, maxAngle);

    unsigned char *d_inputImage, *d_outputImage;
    size_t imageSize = width * height * channels * sizeof(unsigned char);

    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    rotateImage<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height, channels, angle);

    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
