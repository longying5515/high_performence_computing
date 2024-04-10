#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THREAD_NUM 4

pthread_mutex_t mutex;
pthread_cond_t  cond_var;

double a, b, c;
int    count;
double bb, four_ac, two_a, sqrtd, add, sub, a_d2a, s_d2a;

void *thread_root(void *arg) {
    int num_of_thread = (int)arg;
    printf("#%d start.\n", num_of_thread);

    pthread_mutex_lock(&mutex);
    count++;
    if (count == THREAD_NUM) {
        count   = 0;
        bb      = b * b;
        four_ac = 4 * a * c;
        printf("#%d computed b*b and 4ac.\n", num_of_thread);
        pthread_cond_broadcast(&cond_var);
    } else {
        while (pthread_cond_wait(&cond_var, &mutex))
            ;
    }
    pthread_mutex_unlock(&mutex);

    pthread_mutex_lock(&mutex);
    count++;
    if (count == THREAD_NUM) {
        count = 0;
        two_a = 2 * a;
        printf("#%d computed 2a.\n", num_of_thread);
        pthread_cond_broadcast(&cond_var);
    } else {
        while (pthread_cond_wait(&cond_var, &mutex))
            ;
    }
    pthread_mutex_unlock(&mutex);

    pthread_mutex_lock(&mutex);
    count++;
    if (count == THREAD_NUM) {
        count = 0;
        sqrtd = sqrt(bb - four_ac);
        printf("#%d computed sqrt.\n", num_of_thread);
        pthread_cond_broadcast(&cond_var);
    } else {
        while (pthread_cond_wait(&cond_var, &mutex))
            ;
    }
    pthread_mutex_unlock(&mutex);

    pthread_mutex_lock(&mutex);
    count++;
    if (count == THREAD_NUM) {
        count = 0;
        add   = -b + sqrtd;
        sub   = -b - sqrtd;
        printf("#%d computed -b +/-.\n", num_of_thread);
        pthread_cond_broadcast(&cond_var);
    } else {
        while (pthread_cond_wait(&cond_var, &mutex))
            ;
    }
    pthread_mutex_unlock(&mutex);

    pthread_mutex_lock(&mutex);
    count++;
    if (count == THREAD_NUM) {
        count = 0;
        a_d2a = add / two_a;
        s_d2a = sub / two_a;
        printf("#%d computed x1 and x2.\n", num_of_thread);
        pthread_cond_broadcast(&cond_var);
    } else {
        while (pthread_cond_wait(&cond_var, &mutex))
            ;
    }
    pthread_mutex_unlock(&mutex);

    printf("#%d exit.\n", num_of_thread);
    pthread_exit(0);
}
int main() {
    clock_t    start_time, end_time;
    pthread_t *pthread_id = NULL;
    long long  i;

    pthread_id = (pthread_t *)malloc(THREAD_NUM * sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond_var, NULL);
    a = 1, b = 2, c = 0;
    for (i = 0; i < THREAD_NUM; i++) {
        pthread_create(pthread_id + i, NULL, thread_root, (void*)i);
    }
    for (i = 0; i < THREAD_NUM; i++) {
        pthread_join(pthread_id[i], NULL);
    }
    printf("\nx1=%lf\nx2=%lf\n", a_d2a, s_d2a);
    free(pthread_id);

    return 0;
}