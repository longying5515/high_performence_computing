#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ARRAY_SIZE 1000
#define NUM_THREADS 4

int a[ARRAY_SIZE];
int global_index = 0; // 用于跟踪下一个未加元素的全局下标
int sum = 0; // 用于存储总和
pthread_mutex_t mutex; // 互斥锁
pthread_mutex_t second_mutex;


// 线程函数，计算数组的总和
void *computeSum(void *arg) {
    int local_sum = 0;
    int index;

    while (1) {
        pthread_mutex_lock(&mutex); // 加锁
        index = global_index; // 获取下一个未加元素的全局下标
        global_index++;
        pthread_mutex_unlock(&mutex); // 解锁

        if (index >= ARRAY_SIZE) {
            break; // 所有元素已计算
        }

        local_sum += a[index];
    }

    pthread_mutex_lock(&second_mutex); // 加锁
    sum += local_sum; // 将局部总和添加到全局总和
    pthread_mutex_unlock(&second_mutex); // 解锁

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    pthread_mutex_init(&mutex, NULL); // 初始化互斥锁
    pthread_mutex_init(&second_mutex, NULL);
    // 初始化数组a
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i + 1;
    }

    clock_t start_time = clock(); // 记录开始时间

    // 创建多个线程
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, computeSum, NULL);
    }

    // 等待线程完成
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t end_time = clock(); // 记录结束时间
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // 销毁互斥锁
    pthread_mutex_destroy(&mutex);

    printf("Total sum: %d\n", sum);
    printf("Elapsed time: %lf seconds\n", elapsed_time);

    return 0;
}
