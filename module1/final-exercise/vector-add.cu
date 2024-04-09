#include <stdio.h>
#include <assert.h>

/* For adding error handling in locations where CUDA code might otherwise silently fail. /*
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}
// Augment the addVectorsInto definition so that it is a CUDA kernel
__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{

  /* Refactor the body of addVectorsInto: it will be launched inside of a single thread, and only needs to do one thread's worth of work on the input vectors. Be certain the thread will never try to access elements outside the range of the input vectors, and take care to note whether or not the thread needs to do work on more than one element of the input vectors. */
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
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
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  /* Update memory allocations, and memory freeing to reflect that the 3 vectors a, b, and result need to be accessed by host and device code. */
  checkCuda( cudaMallocManaged(&a, size) );
  checkCuda( cudaMallocManaged(&b, size) );
  checkCuda( cudaMallocManaged(&c, size) );

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  /* Choose and utilize a working execution configuration so that addVectorsInto launches as a CUDA kernel. */
  threadsPerBlock = 256;
  numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
  
  // Error Handling
  checkCuda( cudaGetLastError() );
  checkCuda( cudaDeviceSynchronize() );

  checkElementsAre(7, c, N);
  // Error Handling
  checkCuda( cudaFree(a) );
  checkCuda( cudaFree(b) );
  checkCuda( cudaFree(c) );
}
