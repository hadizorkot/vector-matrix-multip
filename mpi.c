#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>


void fullyConnectedLayer(int m, int n, int rank, int size) {
    double *w = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    for (int i = 0; i < m * n; ++i) {
        w[i] = (double)rand() / RAND_MAX;  
    }

    for (int i = 0; i < n; ++i) {
        x[i] = (double)rand() / RAND_MAX; 
    }

    double start_time = MPI_Wtime();
    
    int local_m = m / size;
    int local_start = rank * local_m;
    int local_end = (rank == size - 1) ? m : local_start + local_m;

    double *local_z = (double *)malloc(local_m * sizeof(double));

    for (int i = local_start; i < local_end; ++i) {
        local_z[i - local_start] = 0;
        for (int j = 0; j < n; ++j) {
            local_z[i - local_start] += w[i * n + j] * x[j];
        }
        local_z[i - local_start] = (local_z[i - local_start] > 0) ? local_z[i - local_start] : 0;
    }

    double *z = NULL;
    if (rank == 0) {
        z = (double *)malloc(m * sizeof(double));
    }
    MPI_Gather(local_z, local_m, MPI_DOUBLE, z, local_m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    double end_time = MPI_Wtime();

    if (rank == 0) {
        
        double execution_time = end_time - start_time;

        printf("Parallel Execution Time: %.6f seconds\n", execution_time);

        free(z);
    }

    free(w);
    free(x);
    free(local_z);
}

int main(int argc, char *argv[]) {    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    int m = 1024;
    int n = 2048;

    fullyConnectedLayer(m, n, rank, size);

   
    MPI_Finalize();

    return 0;
}