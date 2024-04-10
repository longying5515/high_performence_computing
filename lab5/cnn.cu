#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA卷积内核函数
__global__ void convolution2D(float* input, float* kernel, float* output,
                              int inputHeight, int inputWidth, int kernelSize,
                              int outputHeight, int outputWidth, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < outputHeight && j < outputWidth) {
        int outputIdx = i * outputWidth + j;
        output[outputIdx] = 0.0;

        for (int ci = 0; ci < 3; ++ci) {
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int inputI = i * stride + ki;
                    int inputJ = j * stride + kj;
                    int inputIdx = (inputI * inputWidth + inputJ) * 3 + ci;

                    int kernelIdx = (ki * kernelSize + kj) * 3 + ci;

                    output[outputIdx] += input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
    }
}

int main() {
    const int inputHeight = 4096;
    const int inputWidth = 4096;
    const int inputChannel = 3;

    const int kernelSize = 3;

    const int stride1 = 1;
    const int stride2 = 2;
    const int stride3 = 3;

    const int outputHeight1 = (inputHeight - kernelSize) / stride1 + 1;
    const int outputWidth1 = (inputWidth - kernelSize) / stride1 + 1;

    const int outputHeight2 = (inputHeight - kernelSize) / stride2 + 1;
    const int outputWidth2 = (inputWidth - kernelSize) / stride2 + 1;

    const int outputHeight3 = (inputHeight - kernelSize) / stride3 + 1;
    const int outputWidth3 = (inputWidth - kernelSize) / stride3 + 1;

    // 分配内存
    float* input = (float*)malloc(inputHeight * inputWidth * inputChannel * sizeof(float));
    float* kernel = (float*)malloc(kernelSize * kernelSize * inputChannel * sizeof(float));
    float* output1 = (float*)malloc(outputHeight1 * outputWidth1 * sizeof(float));
    float* output2 = (float*)malloc(outputHeight2 * outputWidth2 * sizeof(float));
    float* output3 = (float*)malloc(outputHeight3 * outputWidth3 * sizeof(float));

    // 初始化输入和卷积核（随机生成）
    srand(time(NULL));
    for (int i = 0; i < inputHeight * inputWidth * inputChannel; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < kernelSize * kernelSize * inputChannel; ++i) {
        kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    float* d_input;
    float* d_kernel;
    float* d_output1;
    float* d_output2;
    float* d_output3;

    cudaMalloc((void**)&d_input, inputHeight * inputWidth * inputChannel * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernelSize * kernelSize * inputChannel * sizeof(float));
    cudaMalloc((void**)&d_output1, outputHeight1 * outputWidth1 * sizeof(float));
    cudaMalloc((void**)&d_output2, outputHeight2 * outputWidth2 * sizeof(float));
    cudaMalloc((void**)&d_output3, outputHeight3 * outputWidth3 * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_input, input, inputHeight * inputWidth * inputChannel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * inputChannel * sizeof(float), cudaMemcpyHostToDevice);

    // 启动CUDA卷积内核，并使用CUDA自带的计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(16, 16);
    dim3 gridSize((outputHeight1 + blockSize.x - 1) / blockSize.x, (outputWidth1 + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(start);
    convolution2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output1,
                                           inputHeight, inputWidth, kernelSize,
                                           outputHeight1, outputWidth1, stride1);
    convolution2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output2,
                                           inputHeight, inputWidth, kernelSize,
                                           outputHeight2, outputWidth2, stride2);
    convolution2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output3,
                                           inputHeight, inputWidth, kernelSize,
                                           outputHeight3, outputWidth3, stride3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(output1,d_output1,outputHeight1*outputWidth1*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(output2,d_output2,outputHeight2*outputWidth2*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(output3,d_output3,outputHeight3*outputWidth3*sizeof(float),cudaMemcpyDeviceToHost);

    std::cout << "Time of " <<inputHeight<<"*"<<inputWidth<<" Matrix :"<<milliseconds << " ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);

    // 释放主机内存
    free(input);
    free(kernel);
    free(output1);
    free(output2);
    free(output3);

    return 0;
}
