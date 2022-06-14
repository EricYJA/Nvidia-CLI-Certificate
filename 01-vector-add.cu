#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__ 
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  size_t block_num = 32;
  size_t thread_num = 1024;

  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  checkCuda( cudaMallocManaged(&a, size) );
  checkCuda( cudaMallocManaged(&b, size) );
  checkCuda( cudaMallocManaged(&c, size) );

  initWith<<<block_num, thread_num>>>(3, a, N);
  initWith<<<block_num, thread_num>>>(4, b, N);
  initWith<<<block_num, thread_num>>>(0, c, N);
  
  checkCuda( cudaDeviceSynchronize() );

  addVectorsInto<<<block_num, thread_num>>>(c, a, b, N);
  
  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );

  checkElementsAre(7, c, N);

  checkCuda( cudaFree(a) );
  checkCuda( cudaFree(b) );
  checkCuda( cudaFree(c) );
}
