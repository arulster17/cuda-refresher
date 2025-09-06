#include <cuda_runtime.h>
#include <iostream>

// Kernel: out[x] = a[x] + b[x]
__global__ void add(int n, const int* a, const int* b, int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(int);

    int h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int  bsize = 256;
    dim3 blockSize(bsize);
    dim3 gridSize((N + bsize - 1) / bsize);

    add<<<blockSize, gridSize>>>(N, d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << "\n";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
