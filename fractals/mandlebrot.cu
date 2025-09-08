// nvcc -o gradient gradient.cu
#include <cstdio>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
using namespace std;

__global__ void drawFractal(unsigned char* img, int W, int H) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);   

    if (x >= W || y >= H) return;
    int idx = 3*(y * W + x);

    float xmin = -2.5f, xmax = 1.0f;
    float ymin = -1.0f, ymax = 1.0f;
    float u = xmin + (xmax - xmin) * ((float)x / (W - 1));
    float v = ymin + (ymax - ymin) * ((float)y / (H - 1));

    // x and y are pixel coordinates, compute polar coordinates
    //float r = hypot(x, y);
    //float theta = atan2(y, x);
    
    int maxIters = 500;
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;
    while (zx*zx + zy*zy < 4.0f && iter < maxIters) {
        float tmp = zx*zx - zy*zy + u;
        zy = 2.0f * zx * zy + v;
        zx = tmp;
        iter++;
    }

    if (iter == maxIters) {
        // Inside the set, color black
        img[idx + 0] = 0;
        img[idx + 1] = 0;
        img[idx + 2] = 0;
        return;
    } else {
        float t = (float)iter / maxIters; // normalized iteration count
        img[idx + 0] = t*255;
        img[idx + 1] = t*255;
        img[idx + 2] = t*255;
        // img[idx + 0] = (unsigned char)(9*(1-t)*t*t*t*255);
        // img[idx + 1] = (unsigned char)(15*(1-t)*(1-t)*t*t*255);
        // img[idx + 2] = (unsigned char)(8.5*(1-t)*(1-t)*(1-t)*t*255);
    }


    
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    const int W = 16000, H = 9000;
    size_t imgSize = 3ULL * W * H;

    // Host + device buffers
    unsigned char* d_img;
    unsigned char* h_img = (unsigned char*)malloc(imgSize);
    cudaMalloc(&d_img, imgSize);

    // Launch kernel
    dim3 block(32, 32);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    cudaEventRecord(start);
    drawFractal<<<grid, block>>>(d_img, W, H);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Kernel time: " << ms << " ms\n";


    // Copy back
    cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost);

    // Write PPM image
    // FILE* f = fopen("mandlebrot.ppm", "wb");
    // fprintf(f, "P6\n%d %d\n255\n", W, H);
    // fwrite(h_img, 1, imgSize, f);
    // fclose(f);

    stbi_write_png("mandlebrot_16k_9k.png", W, H, 3, h_img, W*3);

    free(h_img);
    cudaFree(d_img);
    // Cleanup
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
