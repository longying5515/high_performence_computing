#include "parallel_for.h"
#include <pthread.h>
pthread_t pid[1000];

void parallel_for(int start,int end,int increment, void*(*functor)(void*), void *arg, int num_threads){
	int counts=end-start;
	int threads=num_threads;
	if(num_threads>=counts) threads=counts;
	int average_loop=counts/num_threads;
	for(int thread = 0; thread<threads; thread++){
		struct for_index * idx = new for_index;
		idx->start=average_loop*thread;
		idx->increment=increment;
		if(thread < threads-1){		
			idx->end=average_loop*(thread+1)-1;
		}else{
			idx->end=counts-1;		
		
		}
		pthread_create(&(pid[thread]), NULL, functor, (void*) idx);		
	}
	//线程合并进程
	for (int thread=0; thread<threads; thread++)
        pthread_join(pid[thread], NULL);
} 
