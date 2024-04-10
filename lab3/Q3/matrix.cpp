#include <iostream>
#include <ctime>
#include <cstdlib>
#include<panel.h>
#include"parallel_for.h"
using namespace std;
int M, N, K;
float** A,**B,**C;
// 生成随机矩阵
void generateRandomMatrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// 矩阵乘法
void *functor(void* args) {
    struct for_index * idx = (struct for_index *) args;
    int first=idx->start;
	int last=idx->end;
	int increment=idx->increment;
    for (int i = first; i<=last; i+=increment) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < K; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {

    cout << "Enter values for M, N, and K (512-2048): ";
    cin >> M >> N >> K;

    if (M < 512 || M > 2048 || N < 512 || N > 2048 || K < 512 || K > 2048) {
        cout << "Invalid input. All values must be between 512 and 2048." << endl;
        return 1;
    }

    // 分配内存并初始化矩阵
    A = new float*[M];
    B = new float*[N];
    C = new float*[M];

    for (int i = 0; i < M; ++i) {
        A[i] = new float[N];
        C[i] = new float[K];
    }

    for (int i = 0; i < N; ++i) {
        B[i] = new float[K];
    }
    int num_thread=4;
    // 生成随机矩阵
    srand(static_cast<unsigned>(time(nullptr)));
    generateRandomMatrix(A, M, N);
    generateRandomMatrix(B, N, K);

    // 计算矩阵乘法的时间
    clock_t startTime = clock();
    parallel_for(0,M,1,functor,NULL,num_thread);
    clock_t endTime = clock();
    double elapsedTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;

    cout << "Matrix A:" << endl;
    // 输出矩阵A的内容

    cout << "Matrix B:" << endl;
    // 输出矩阵B的内容

    cout << "Matrix C:" << endl;
    // 输出矩阵C的内容

    cout << "Matrix multiplication took " << elapsedTime << " seconds." << endl;

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
