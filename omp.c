#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


void fullyConnectedLayer(int m, int n) {
    
    double *w = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    
    for (int i = 0; i < m * n; ++i) {
        w[i] = (double)rand() / RAND_MAX;  
    }

    for (int i = 0; i < n; ++i) {
        x[i] = (double)rand() / RAND_MAX;  
    }
   
    double start_time = omp_get_wtime();

    double *z = (double *)malloc(m * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        z[i] = 0;
        for (int j = 0; j < n; ++j) {
            z[i] += w[i * n + j] * x[j];
        }
        z[i] = (z[i] > 0) ? z[i] : 0;  
    }
    
    double end_time = omp_get_wtime();

    double execution_time = end_time - start_time;

    printf("Parallel Execution Time: %.6f seconds\n", execution_time);

    free(w);
    free(x);
    free(z);
}

int main() {
    
    int m = 1024;
    int n = 2048;

    fullyConnectedLayer(m, n);

    return 0;
}