#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// 矩阵转化
__global__ void converttomatrix(float* input, float* outmatrix,
    int inputHeight, int inputWidth, int inputChannels,
    int filterHeight, int filterWidth, int filterChannels,
    int outmatrixwidth, int outmatrixheight,
    int outputheight, int outputwidth) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < outmatrixheight && j < outmatrixwidth) {
        outmatrix[i * outmatrixwidth + j] = input[((i / outputheight + j / inputChannels / filterHeight) * inputWidth + i % outputwidth + j / inputChannels % filterWidth) * filterChannels + j % inputChannels];
    }
}
// 矩阵乘法
__global__ void matrixMultiply(float* a, float* b, float* c, int m, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double sum = 0;
    if (i < m && j < k) {
        for (int x = 0; x < n; x++) {
            sum += a[i * n + x] * b[x * k + j];
        }
        c[i * k + j] = sum;
    }
}
int main() {
    const int inputHeight = 4096;
    const int inputWidth = 4096;
    const int inputChannels = 3;

    const int filterHeight = 3;
    const int filterWidth = 3;
    const int filterChannels = 3;

    const int stride = 1;  // 可以尝试不同的步幅

    const int outputHeight = (inputHeight - filterHeight) / stride + 1;
    const int outputWidth = (inputWidth - filterWidth) / stride + 1;
    const int outmatrixheight = outputWidth * outputHeight;
    const int outmatrixwidth = filterChannels * filterHeight * filterWidth;
    // 分配主机上的内存
    float* h_input = new float[inputHeight * inputWidth * inputChannels];
    float* h_filters = new float[filterHeight * filterWidth * filterChannels];
    float* h_output = new float[outputHeight * outputWidth];
    float* h_outmatrix = new float[outmatrixheight * outmatrixwidth];

    // 初始化输入数据和卷积核
    srand(time(NULL));
    for (int i = 0; i < inputHeight * inputWidth * inputChannels; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < filterHeight * filterWidth * filterChannels; ++i) {
        h_filters[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备上的内存
    float* d_input, * d_filters, * d_output, * d_outmatrix;
    cudaMalloc((void**)&d_input, inputHeight * inputWidth * inputChannels * sizeof(float));
    cudaMalloc((void**)&d_filters, filterHeight * filterWidth * filterChannels * sizeof(float));
    cudaMalloc((void**)&d_output, outputHeight * outputWidth * sizeof(float));
    cudaMalloc((void**)&d_outmatrix, outmatrixheight * outmatrixwidth * sizeof(float));

    // 将输入数据和卷积核从主机拷贝到设备
    cudaMemcpy(d_input, h_input, inputHeight * inputWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filters, h_filters, filterHeight * filterWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice);

    // 计算块和网格的大小
    dim3 blockSize(16, 16);  // 选择适当的块大小
    dim3 gridSize((outmatrixwidth + blockSize.x - 1) / blockSize.x, (outmatrixheight + blockSize.y - 1) / blockSize.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // 调用CUDA核函数
    converttomatrix << <gridSize, blockSize >> > (d_input, d_outmatrix,
        inputHeight, inputWidth, inputChannels,
        filterHeight, filterWidth, filterChannels,
        outmatrixwidth, outmatrixheight,
        outputHeight, outputWidth);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    // 将结果从设备拷贝回主机
    cudaMemcpy(h_outmatrix, d_outmatrix, outmatrixheight * outmatrixwidth * sizeof(float), cudaMemcpyDeviceToHost);
    // 矩阵乘法
    const int m = outmatrixheight;
    const int n = outmatrixwidth;
    const int k = 1;
    float* h_result = new float[m * sizeof(float)];
    float* d_result;
    dim3 blocksize1(16, 1);
    dim3 gridsize1((m + blocksize1.x - 1) / blocksize1.x, 1);
    cudaMalloc((void**)&d_result, m * sizeof(float));
    cudaEventRecord(start1);
    //矩阵乘法
    matrixMultiply << <gridsize1, blocksize1 >> > (d_outmatrix, d_filters, d_result, m, n, k);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    cudaMemcpy(h_result, d_result, m * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 10; i++) {
    //    cout << h_result[i] << " ";
    // }
    // cout << endl;
    //for (int i = 0; i < 48; i++) {

    //    cout << h_input[i] << " "; if ((i + 1) % 12== 0)
    //        cout << endl;
    //}
    //cout << endl;
    //cout << "*********************************************" << endl;
    //// 输出前10个元素
    //for (int i = 0; i < 108; ++i) {

    //    cout << h_outmatrix[i] << " "; if ((i + 1) % 27 == 0)
    //        cout << endl;
    //}
    //cout << endl;
    //for (int i = 0; i < m; i++)
    //{
    //    cout << h_result[i];
    //}
    //cout << endl;
    cout << "Time of " << inputHeight << "*" << inputWidth << " InputMatrix :" << milliseconds +  milliseconds1 << " ms" << endl;
    // 释放内存
    delete[] h_input;
    delete[] h_filters;
    delete[] h_output;
    delete[] h_outmatrix;
    delete[] h_result;
    cudaFree(d_input);
    cudaFree(d_filters);
    cudaFree(d_output);
    cudaFree(d_outmatrix);
    cudaFree(d_result);
    return 0;
}
