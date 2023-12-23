#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <windows.h>

using namespace std;

__global__ void matrixMult(const int* A, const int* B, int* C, int size)
{
    int i = size * (blockDim.y * blockIdx.y + threadIdx.y);
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int sum = 0;

    for (int k = 0; k < size; k++)
        sum += A[i + k] * B[k * size + j];

    int ind = size * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    C[ind] = sum;
}

int main(int argc, char** argv) {
    int threads, size;
    cin >> threads >> size;

    int* A = new int [size * size];
    int* B = new int [size * size];
    int* C = new int [size * size];

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = B[i * size + j] = i * j;
        }
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int* calcA = NULL;
    cudaMalloc((void**)&calcA, size * size);

    int* calcB = NULL;
    cudaMalloc((void**)&calcB, size * size);

    int* calcC = NULL;
    cudaMalloc((void**)&calcC, size * size);

    cudaMemcpy(calcA, A, size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(calcB, B, size * size, cudaMemcpyHostToDevice);

    cout << "Ïðîãðàììà íà÷àëà ñâîþ ðàáîòó" << endl;

    dim3 threadsPerBlock = dim3(threads, threads);
    dim3 blocksPerGrid = dim3(size / threads, size / threads);

    cudaEventRecord(start, 0);
    matrixMult <<< blocksPerGrid, threadsPerBlock >>> (calcA, calcB, calcC, size);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, end);

    cout << "Size: " << size << endl;
    cout << "Duration: " << kernelTime / 1000;

    cudaFree(calcA);
    cudaFree(calcB);
    cudaFree(calcC);
    free(A);
    free(B);
    free(C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
