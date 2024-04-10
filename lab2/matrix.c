#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define SIZE 512
#define NUM_THREADS 4 // 定义线程数

int A[SIZE][SIZE];
int B[SIZE][SIZE];
int C[SIZE][SIZE];

// 结构体用于传递给工作线程的参数
typedef struct {
    int thread_id;
    int start_row;
    int end_row;
} ThreadArgs;

// 每个工作线程执行的函数
void *matrixMultiply(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    int start_row = args->start_row;
    int end_row = args->end_row;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < SIZE; j++) {
            int temp = 0;
            for (int k = 0; k < SIZE; k++) {
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    ThreadArgs thread_args[NUM_THREADS];

    // 生成随机矩阵A和B
    srand((unsigned)time(NULL));
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = rand() % 5;
            B[i][j] = rand() % 5;
        }
    }

    clock_t start_time = clock();

    // 创建并启动线程
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].start_row = i * (SIZE / NUM_THREADS);
        thread_args[i].end_row = (i + 1) * (SIZE / NUM_THREADS);
        pthread_create(&threads[i], NULL, matrixMultiply, (void *)&thread_args[i]);
    }

    // 等待线程完成
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("运行时间: %lf秒\n", elapsed_time);

    return 0;
}
