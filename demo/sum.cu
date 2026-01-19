#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// element-wise vector addition
__global__ void vector_add(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main()
{
    const int N = 1 << 24; // ~16M elements
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    size_t bytes = N * sizeof(float);

    // allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    // copy data to device
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // launch kernel
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CHECK(cudaDeviceSynchronize());

    // copy result back
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // verify
    int error_count = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) error_count++;
    }
    if (error_count == 0)
        printf("Success! All results are correct.\n");
    else
        printf("Error! %d elements are wrong.\n", error_count);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
