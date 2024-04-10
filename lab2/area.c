#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
 
#define intervals 10000     //区间数
int tnum=10;            //进程数
int sum = 0;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;
//计算区间
void* cast (void* args)
{
	int i;
	float x,y;
	int temp=0;
	for(i=0; i< (intervals/tnum); i++){
		x= (float)(rand() % 1001) * 0.001f;
		y= (float)(rand() % 1001) * 0.001f;
		if(y<x*x) temp++;
	}
    //加上
	pthread_mutex_lock(&mutex);
	sum += temp;
	pthread_mutex_unlock(&mutex);
}
 
int main(int argc, char *argv[]){
	printf("Please input the number of thread:\n");
    scanf("%d",&tnum);
	pthread_t thread[tnum];
	int i;
	for (i=0;i<tnum;i++)
	{
		if(pthread_create(&thread[i], NULL, cast, NULL))
			perror("Pthread Create Fails");
	}
	for (i=0;i<tnum;i++)
	{
		if(pthread_join(thread[i], NULL))
			perror("Pthread Join Fails");
	}
	float shadow=(float) sum/(float) intervals;
	printf("sum:%d\n",sum);
	printf("intervals:%d\n",intervals);
	printf("the area of the shadow is:%f\n",shadow);
	return 0;
}
