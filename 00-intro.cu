#include <stdio.h>

void helloCPU() {
  printf("Hello from the CPU.\n");
}

__global__ 
void helloGPU() {
  printf("Hello from the GPU.\n");
}

__global__ 
void loop() {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d\n", i);
}

__global__ 
void 

int main() {
  helloGPU<<<1, 1>>>();
  cudaDeviceSynchronize();

  helloCPU();

  helloGPU<<<1, 1>>>();   
  cudaDeviceSynchronize();

  loop<<<2, 5>>>();
  cudaDeviceSynchronize();

  cudaError_t syncErr, asyncErr;
  syncErr = cudaGetLastError();
  asyncErr = cudaDeviceSynchronize();

  if (syncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
}
