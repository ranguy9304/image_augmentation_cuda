
#include <iostream>
#include <opencv2/opencv.hpp>


__global__ void colorShift(unsigned char *image, int width, int height, int channels, int r_shift, int g_shift, int b_shift)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * channels;
        image[index] = min(max(image[index] + b_shift, 0), 255);
        image[index + 1] = min(max(image[index + 1] + g_shift, 0), 255);
        image[index + 2] = min(max(image[index + 2] + r_shift, 0), 255);
    }
}

void applyRandomColorShift(unsigned char *image,unsigned char *out, int width, int height, int channels, int maxShift)
{
    int r_shift = cv::theRNG().uniform(-maxShift, maxShift + 1);
    int g_shift = cv::theRNG().uniform(-maxShift, maxShift + 1);
    int b_shift = cv::theRNG().uniform(-maxShift, maxShift + 1);

    unsigned char *d_image;
    size_t imageSize = width * height * channels * sizeof(unsigned char);

    cudaMalloc(&d_image, imageSize);
    cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    colorShift<<<gridDim, blockDim>>>(d_image, width, height, channels, r_shift, g_shift, b_shift);

    cudaMemcpy(out, d_image, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
}