#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void fullyConnectedLayer(int m, int n) {
    
    double *w = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    
    for (int i = 0; i < m * n; ++i) {
        w[i] = (double)rand() / RAND_MAX;   
    }

    for (int i = 0; i < n; ++i) {
        x[i] = (double)rand() / RAND_MAX;  
    }
  
    clock_t start_time = clock();

    double *z = (double *)malloc(m * sizeof(double));

    #pragma acc data copyin(w[0:m*n], x[0:n]) copyout(z[0:m])
    {
        #pragma acc parallel loop present(w[0:m*n], x[0:n]) num_gangs(32) vector_length(256)
        for (int i = 0; i < m; ++i) {
            double partial_sum = 0.0;
            #pragma acc loop reduction(+:partial_sum)
            for (int j = 0; j < n; ++j) {
                partial_sum += w[i * n + j] * x[j];
            }
            z[i] = (partial_sum > 0) ? partial_sum : 0;  
        }

        #pragma acc wait
    }

    
    clock_t end_time = clock();

    
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    
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