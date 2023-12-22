#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 1024
#define N 2048
#define BLOCK_SIZE 256

_global_ void matrixVectorMultiplyWithReLU(double *w, double *x, double *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        z[i] = 0;
        for (int j = 0; j < n; ++j) {
            z[i] += w[i * n + j] * x[j];
        }
        z[i] = (z[i] > 0) ? z[i] : 0; 
    }
}

int main() {
    
    int m = M;
    int n = N;

    
    double *w = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *z = (double *)malloc(m * sizeof(double));

    
    for (int i = 0; i < m * n; ++i) {
        w[i] = (double)rand() / RAND_MAX;  
    }

    for (int i = 0; i < n; ++i) {
        x[i] = (double)rand() / RAND_MAX;  
    }

    
    double *d_w, *d_x, *d_z;
    cudaMalloc((void **)&d_w, m * n * sizeof(double));
    cudaMalloc((void **)&d_x, n * sizeof(double));
    cudaMalloc((void **)&d_z, m * sizeof(double));

    
    cudaMemcpyAsync(d_w, w, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    matrixVectorMultiplyWithReLU<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_x, d_z, n);
    cudaEventRecord(stop);

    
    cudaDeviceSynchronize();

    
    cudaMemcpyAsync(z, d_z, m * sizeof(double), cudaMemcpyDeviceToHost);

    
    cudaDeviceSynchronize();

    
    float execution_time;
    cudaEventElapsedTime(&execution_time, start, stop);
    execution_time /= 1000.0;  

    
    printf("Parallel Execution Time: %.6f seconds\n", execution_time);

    
    free(w);
    free(x);
    free(z);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_z);

    return 0;
}