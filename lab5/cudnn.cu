#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <chrono>
#include <curand_kernel.h>

using namespace std;

// 检查CUDA函数调用是否成功
#define CUDA_CHECK(call) \
do { \
    cudaError_t cuda_status = call; \
    if (cuda_status != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(cuda_status) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 检查cuDNN函数调用是否成功
#define CUDNN_CHECK(call) \
do { \
    cudnnStatus_t cudnn_status = call; \
    if (cudnn_status != CUDNN_STATUS_SUCCESS) { \
        cerr << "cuDNN Error: " << cudnnGetErrorString(cudnn_status) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 检查cuRAND函数调用是否成功
#define CURAND_CHECK(call) \
do { \
    curandStatus_t curand_status = call; \
    if (curand_status != CURAND_STATUS_SUCCESS) { \
        cerr << "cuRAND Error: " << curandGetErrorString(curand_status) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void initializeData(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;  // Random initialization between 0 and 1
    }
}

int main() {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Loop over different matrix s256
    int matrixSize = 4096;
    cout << "Matrix Size: " << matrixSize << "x" << matrixSize << endl;

    // Set the matrix size
    const int inputHeight = matrixSize;
    const int inputWidth = matrixSize;
    const int inputChannels = 3;

    const int filterHeight = 3;
    const int filterWidth = 3;
    const int filterChannels = 3; 
    const int stride = 1;
    const int outputHeight = (inputHeight - filterHeight) / stride + 1;
    const int outputWidth = (inputWidth - filterWidth) / stride + 1;
   
    float* h_output = new float[outputHeight * outputWidth ];

    // Allocate device memory for input, filter, and output
    float* d_input, * d_filters;
    CUDA_CHECK(cudaMalloc((void**)&d_input, inputHeight * inputWidth * inputChannels * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_filters, filterHeight * filterWidth * filterChannels * sizeof(float)));

    float* h_input = new float[inputHeight * inputWidth * inputChannels];
    float* h_filters = new float[filterHeight * filterWidth * filterChannels ];

    // Randomly initialize matrix
    initializeData(h_input, inputHeight * inputWidth * inputChannels);
    initializeData(h_filters, filterHeight * filterWidth * filterChannels );

    CUDA_CHECK(cudaMemcpy(d_input, h_input, inputHeight * inputWidth * inputChannels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filters, h_filters, filterHeight * filterWidth * filterChannels * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuDNN tensor descriptors, filter descriptor, and convolution descriptor
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, inputChannels, inputHeight, inputWidth));
    cudnnFilterDescriptor_t filterDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           1, filterChannels, filterHeight, filterWidth));

    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, stride, stride, 1, 1,
                                                 CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    int out_n;
    int out_c;
    int out_h;
    int out_w;
      
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
          convDesc, inputDesc, filterDesc,
          &out_n, &out_c, &out_h, &out_w));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           out_n, out_c, out_h, out_w));
    float *out_data;
    CUDA_CHECK(cudaMalloc(
          &out_data, out_n * out_c * out_h * out_w * sizeof(float)));
    // algorithm
    const int requestedAlgoCount = 1;
    int returnedAlgoCount;

    // Use an array to store performance results
    cudnnConvolutionFwdAlgoPerf_t perfResults[1];

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                                       inputDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       outputDesc,
                                                       requestedAlgoCount,
                                                       &returnedAlgoCount,
                                                       perfResults));

    // Get the chosen algorithm
    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    // workspace size && allocate memory
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                        inputDesc,
                                                        filterDesc,
                                                        convDesc,
                                                        outputDesc,
                                                        algo,
                                                        &workspace_size));

    void * workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    std::cerr << "Workspace size: " << (workspace_size/ 1048576.0) << "MB"
          << std::endl;
    // Perform convolution using cuDNN
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Assume alpha is defined
    float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK(cudaEventRecord(start));
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input,
                                        filterDesc, d_filters, convDesc,
                                        algo,
                                        workspace, workspace_size, &beta, outputDesc, out_data));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds_cudnn = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_cudnn, start, stop));
    CUDA_CHECK(cudaMemcpy(h_output, out_data, outputHeight * outputWidth * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print cuDNN convolution time
  
    cout << "cuDNN Convolution Time: " << milliseconds_cudnn << " ms" << endl;

    // Free allocated resources
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filters));
    CUDA_CHECK(cudaFree(out_data));
    CUDA_CHECK(cudaFree(workspace));
    delete[] h_input;
    delete[] h_filters;
    delete[] h_output;

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filterDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));

    CUDNN_CHECK(cudnnDestroy(cudnn));

    return 0;
}
