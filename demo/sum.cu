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

/*
 * block-level reduction
 */
__global__ void reduce_sum(const float* input, float* output, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load data
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main()
{
    const int N = 1 << 24;  // ~16M elements
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    size_t bytes = N * sizeof(float);
    size_t out_bytes = blocks * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(out_bytes);

    for (int i = 0; i < N; i++)
        h_in[i] = 1.0f;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, out_bytes));

    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // warmup
    reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());

    // measured run
    reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    // final reduction on CPU
    float sum = 0.0f;
    for (int i = 0; i < blocks; i++)
        sum += h_out[i];

    printf("Sum = %.0f (expected %.0f)\n", sum, (float)N);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}

