#include <stdio.h>

void helloCPU() {
  printf("Hello from the CPU.\n");
}

__global__ void helloGPU() {
  printf("Hello from the GPU.\n");
}

__global__ void loop()
{
  /*
   * This idiomatic expression gives each thread
   * a unique index within the entire grid.
   */

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d\n", i);
}

int main() {
  helloGPU<<<1, 1>>>();
  cudaDeviceSynchronize();

  helloCPU();

  helloGPU<<<1, 1>>>();   
  cudaDeviceSynchronize();

  loop<<<2, 5>>>();
  cudaDeviceSynchronize();
}
