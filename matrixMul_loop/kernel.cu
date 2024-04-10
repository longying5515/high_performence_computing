#include <iostream>
#include <cublas_v2.h>

void matrixMultiplication(const cublasHandle_t& handle, int size) {
    const int n = size;

    // 分配主机内存
    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    float* h_C = new float[n * n];

    // 初始化矩阵
    // TODO: 根据实际需求初始化矩阵 h_A 和 h_B

    // 分配设备内存
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * n * sizeof(float));
    cudaMalloc((void**)&d_C, n * n * sizeof(float));

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // 执行矩阵乘法
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n);

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: 处理矩阵相乘结果

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // 初始化 CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 矩阵规模从 512 增加至 8192（可以根据需要调整）
    for (int size = 512; size <= 8192; size *= 2) {
        std::cout << "Matrix Size: " << size << "x" << size << std::endl;
        matrixMultiplication(handle, size);
        std::cout << std::endl;
    }

    // 销毁 CUBLAS 句柄
    cublasDestroy(handle);

    return 0;
}
