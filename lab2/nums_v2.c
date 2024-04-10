#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ARRAY_SIZE 1000
#define GROUP_SIZE 10
#define NUM_THREADS 100

int a[ARRAY_SIZE+100];
int global_group_index = 0;
int sum = 0;
pthread_mutex_t mutex;
pthread_mutex_t second_mutex;


void *computeSum(void *arg) {
    int local_group_index;
    int group_sum = 0;
 
    while (1) {  
        pthread_mutex_lock(&mutex);
        local_group_index = global_group_index;
        // printf("global_group_index:%d\n",global_group_index);
        global_group_index++;

        pthread_mutex_unlock(&mutex);

        if (local_group_index >= NUM_THREADS) {
            break;
        }
printf("group_sum:%d\n",group_sum);
        int group_start = local_group_index * GROUP_SIZE;
        int group_end = (local_group_index + 1) * GROUP_SIZE;
        group_sum = 0;
        for (int i = group_start; i < group_end; i++) {
            // printf("a[%d]:%d",i,a[i]);

            group_sum += a[i];
        }

        pthread_mutex_lock(&second_mutex);
        
        sum += group_sum;     
        // printf("sum:%d\n",sum);
        pthread_mutex_unlock(&second_mutex);
   
    }

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_init(&second_mutex, NULL);
    for (int i = 0; i <  ARRAY_SIZE ; i++) {
        a[i] = i + 1;
    }

    clock_t start_time = clock();

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, computeSum, NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    pthread_mutex_destroy(&mutex);

    printf("Total sum: %d\n", sum);
    printf("Elapsed time: %lf seconds\n", elapsed_time);

    return 0;
}
