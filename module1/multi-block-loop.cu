#include <stdio.h>


__global__ void loop()
{
    int globalIdx = threadIdx.x + blockIdx.x*blockDim.x;
    printf("This is iteration number %d\n", globalIdx);
}

int main()
{

  loop<<<1, 10>>>();
  cudaDeviceSynchronize();
}
