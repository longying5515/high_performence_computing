#include <iostream>
#include <ctime>
#include <cstdlib>
#include<omp.h>
#include <chrono>
#include <random>
using namespace std;

// 生成随机矩阵
void generateRandomMatrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// 矩阵乘法
void matrixMultiplication(float** A, float** B, float** C, int M, int N, int K) {
#pragma omp parallel num_threads(4)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < K; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int M, N, K;
    M=1024;
    N=1024;
    K=1024;

    if (M < 512 || M > 2048 || N < 512 || N > 2048 || K < 512 || K > 2048) {
        cout << "Invalid input. All values must be between 512 and 2048." << endl;
        return 1;
    }

    // 分配内存并初始化矩阵
    float** A = new float*[M];
    float** B = new float*[N];
    float** C = new float*[M];

    for (int i = 0; i < M; ++i) {
        A[i] = new float[N];
        C[i] = new float[K];
    }

    for (int i = 0; i < N; ++i) {
        B[i] = new float[K];
    }

    // 生成随机矩阵
    srand(static_cast<unsigned>(time(nullptr)));
    generateRandomMatrix(A, M, N);
    generateRandomMatrix(B, N, K);

    // 计算矩阵乘法的时间
auto startTime = chrono::high_resolution_clock::now();
    matrixMultiplication(A, B, C, M, N, K);
    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedTime = endTime - startTime;
    

    cout << "Matrix A:" << endl;
    // 输出矩阵A的内容

    cout << "Matrix B:" << endl;
    // 输出矩阵B的内容

    cout << "Matrix C:" << endl;
    // 输出矩阵C的内容

     cout << "Matrix multiplication took " << elapsedTime.count() << " seconds." << endl;

    // 释放分配的内存
    for (int i = 0; i < M; ++i) {
        delete[] A[i];
        delete[] C[i];
    }

    for (int i = 0; i < N; ++i) {
        delete[] B[i];
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
