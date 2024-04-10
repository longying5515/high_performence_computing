#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define M 500
#define N 500

int main(int argc, char *argv[]) {
  int rank, size;
  double epsilon = 0.001;
  int iterations = 0;
  int iterations_print = 1;
  double diff;
  double mean;
  double my_diff;
  int begin_row=1;
   double startime,wtime =0;
  double u[M][N];
  double w[M][N];
  double local_u[M+2][N];
  double local_w[M+2][N];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_proc = M / size;
  int start_row = rank * rows_per_proc;
  
 int end_row=rows_per_proc+1;
  // Additional buffer for packing/unpacking
  double pack_buffer[rows_per_proc * N];

  // printf("\n");
  // printf("HEATED_PLATE_MPI\n");
  // printf("  C/MPI version with MPI_Pack and MPI_Unpack\n");
  // printf("  A program to solve for the steady state temperature distribution\n");
  // printf("  over a rectangular plate.\n");
  // printf("\n");
  // printf("  Spatial grid of %d by %d points.\n", M, N);
  // printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
  // printf("  Number of processors available = %d\n", size);

  // Set the boundary values, which don't change.
  mean = 0.0;
int right,left;
if(rank>0){
    left=rank-1;
}
else{
    left= MPI_PROC_NULL;
}
if(rank<size-1){
    right=rank+1;
}
else{
    right=MPI_PROC_NULL;
}


if(rank==0){
  for (int i = 1; i < M - 1; i++) {
    w[i][0] = 100.0;
  }

  for (int i = 1; i < M - 1; i++) {
    w[i][N - 1] = 100.0;
  }

  for (int j = 0; j < N; j++) {
    w[M - 1][j] = 100.0;
  }

  for (int j = 0; j < N; j++) {
    w[0][j] = 0.0;
  }

  // Average the boundary values to come up with a reasonable initial value for the interior.
  for (int i = 1; i < M - 1; i++) {
    mean = mean + w[i][0] + w[i][N - 1];
  }

  for (int j = 0; j < N; j++) {
    mean = mean + w[M - 1][j] + w[0][j];
  }

  mean = mean / (double)(2 * M + 2 * N - 4);
  printf("\n");
  printf("  MEAN = %f\n", mean);

  // Initialize the interior solution to the mean value.
  for (int i = 1; i < M - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      w[i][j] = mean;
    }
  }

  
  printf("\n");
  printf(" Iteration  Change\n");
  printf("\n");startime = MPI_Wtime();
}
diff = epsilon;
 
  while (epsilon <= diff) {
    // Scatter the data to each process
    MPI_Scatter(w, rows_per_proc * N, MPI_DOUBLE, pack_buffer, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Unpack the received data
    int position = 0;
  for (int i = 1; i < rows_per_proc+1; i++) {
  for (int j = 0; j < N; j++) {
    local_w[i][j] = pack_buffer[position++];
  }
}
 for (int j = 0; j < N; j++) {
  local_w[0][j]=0.0;
  local_w[ rows_per_proc+1][j]=0.0;
 }
//  for(int i=0;i<=rows_per_proc+1;i++){
//   for(int j=0;j<N;j++){
//     printf(" %f ",local_w[i][j]);
//   }

//   printf("\n");
//  }
   MPI_Sendrecv(&local_w[rows_per_proc][0],N,MPI_DOUBLE,right,1,&local_w[0][0],N,MPI_DOUBLE,left,1,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   MPI_Sendrecv(&local_w[1][0],N,MPI_DOUBLE,left,2,&local_w[rows_per_proc+1][0],N,MPI_DOUBLE,right,2,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  
    // Each process performs its local computation
    my_diff = 0.0;
    if(rank==0){
        begin_row=2;
    }
    if(rank==size-1){
        end_row=rows_per_proc-2;
    }
      for (int i = begin_row; i < end_row; i++) {
      for (int j = 1; j < N - 1; j++) {
        local_u[i][j] = (local_w[i - 1][j] + local_w[i + 1][j] + local_w[i][j - 1] + local_w[i][j + 1]) / 4.0;
        if (my_diff < fabs(local_w[i][j] - local_u[i][j])) {
          my_diff = fabs(local_w[i][j] - local_u[i][j]);
        }
      }
    }
//      for(int i=0;i<=rows_per_proc+1;i++){
//   for(int j=0;j<N;j++){
//     printf(" %f ",local_w[i][j]);
//   }
//   printf("\n");
//  }
//  printf("*********************************************************************************\n");
    // Pack the local result for sending
    for (int i = begin_row; i < end_row; i++) {
      for (int j = 1; j < N-1; j++) {
        local_w[i][j]=local_u[i][j];
      }
    }
//      for(int i=0;i<=rows_per_proc+1;i++){
//   for(int j=0;j<N;j++){
//     printf(" %f ",local_w[i][j]);
//   }
//   printf("\n");
//  }
//  printf("*********************************************************************************\n");
     position = 0;
    for (int i = 1; i < rows_per_proc+1; i++) {
      for (int j = 0; j < N; j++) {
        pack_buffer[position++] = local_w[i][j];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Gather the local results back to the root process
    MPI_Gather(pack_buffer, rows_per_proc * N, MPI_DOUBLE, w, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reduce the maximum difference across all processes
    MPI_Reduce(&my_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Synchronize all processes
    

    iterations++;

    if (rank == 0 && iterations == iterations_print) {
      printf("  %8d  %f\n", iterations, diff);
      iterations_print = 2 * iterations_print;
    }
  }

  

  if (rank == 0) {wtime+=(MPI_Wtime()-startime);
    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  Wallclock time = %f\n", wtime);
  printf("\n");
  printf("HEATED_PLATE_MPI:\n");
  printf("  Normal end of execution.\n");
  }
MPI_Finalize();


  
  return 0;
}
