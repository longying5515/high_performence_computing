
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N 8192
#define DIM 32
#include <iostream>
#include <cmath>
using namespace std;
// 定义矩阵大小


// CUDA核函数：矩阵乘法
__global__ void matrixMul(int* a, int* b, int* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {

    // 分配矩阵内存并初始化
    int* h_a, * h_b, * h_c;
    int* d_a, * d_b, * d_c;

    size_t size = N * N * sizeof(int);

    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // 在GPU上分配内存
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 将数据从主机内存复制到GPU内存
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 定义CUDA网格和块的大小
    dim3 dimBlock(DIM, DIM);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // 调用CUDA核函数
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 调用 CUDA 核函数
    matrixMul << <dimGrid, dimBlock >> > (d_a, d_b, d_c);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 将结果从GPU内存复制回主机内存
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 打印部分结果进行验证
    //for (int i = 0; i < min(N, 10); ++i) {
    //    for (int j = 0; j < min(N, 10); ++j) {
    //        cout << h_c[i * N + j] << " ";
    //    }
    //    cout << endl;
    //}
     // 打印运行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time: " << milliseconds << " ms" << endl;
    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
