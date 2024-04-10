#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <random>
#define N 8192
#define dim 512
using namespace std;

// 定义矩阵大小
// CUDA核函数：矩阵乘法
__global__ void matrixMul(float* a, float* b, float* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    // 分配矩阵内存并初始化
    float* h_a, * h_b, * h_c;
    float* d_a, * d_b, * d_c;

    size_t size = N * N * sizeof(float);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // 使用 C++11 的随机数生成器
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < N * N; ++i) {
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
    }

    // 在GPU上分配内存
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 将数据从主机内存复制到GPU内存
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 定义CUDA网格和块的大小
    dim3 dimBlock(dim, dim);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // 调用CUDA核函数

    cudaEventRecord(start);

    // 调用 CUDA 核函数
    matrixMul << <dimGrid, dimBlock >> > (d_a, d_b, d_c);

    // 将结果从GPU内存复制回主机内存
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // 打印运行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Matrix Size: " << N << "x" << N << ", Block Size: " << dim << "x" << dim << ", Time: " << milliseconds << " ms" << endl;

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
