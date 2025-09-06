#include <cuda_runtime.h>
#include <iostream>
#include <random>

using namespace std;

// Kernel: out[x][y][z] = a[x][y][z] + b[x][y] + c[x]
__global__ void md_add(int X, int Y, int Z, const int* a, const int* b, const int* c, int* out) {
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    int ib = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z * blockDim.z + threadIdx.z;
    if (ia < X && ib < Y && ic < Z) {
        out[ia*Y*Z + ib*Z + ic] = a[ia*Y*Z + ib*Z + ic] + b[ia*Y + ib] + c[ia];
    }
}
//  
int main() {
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    const int X = 128;
    const int Y = 128;
    const int Z = 128;

    int* h_a = new int[X*Y*Z];
    int* h_b = new int[X*Y];
    int* h_c = new int[X];
    int* h_out = new int[X*Y*Z];


    for (int i = 0; i < X; i++) {
        for(int j = 0; j < Y; j++) {
            for(int k = 0; k < Z; k++) {
                h_a[i*Y*Z + j*Z + k] = k;
            }
            h_b[i*Y + j] = j;
        }
        h_c[i] = i;
    }

    int *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_a, X*Y*Z*sizeof(int));
    cudaMalloc(&d_b, X*Y*sizeof(int));
    cudaMalloc(&d_c, X*sizeof(int));
    cudaMalloc(&d_out, X*Y*Z*sizeof(int));

    cudaMemcpy(d_a, h_a, X*Y*Z*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, X*Y*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, X*sizeof(int), cudaMemcpyHostToDevice);


    dim3 blockSize(8, 8, 8);
    dim3 gridSize((X+7)/8, (Y+7)/8, (Z+7)/8);

    md_add<<<gridSize, blockSize>>>(X, Y, Z, d_a, d_b, d_c, d_out);

    cudaMemcpy(h_out, d_out, X*Y*Z*sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Kernel time: " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print some output values
    random_device rd;                  // non-deterministic seed (if available)
    mt19937 gen(rd());                 // Mersenne Twister engine
    uniform_int_distribution<> dist(0, 127);  // range 0–99

    for(int i = 0; i < 10; i++) {
        int a = dist(gen);
        int b = dist(gen);
        int c = dist(gen);
        printf("out[%d][%d][%d] = %d\n", a, b, c, h_out[a*Y*Z + b*Z + c]);
    }

    
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_out);

}
