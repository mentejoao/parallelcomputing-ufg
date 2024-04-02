#include <stdio.h>


__global__ void firstParallel()
{
  printf("This should be running in parallel.\n");
}

__global__ void secondParallel()
{
  printf("This should be running in parallel 2.\n");
}

__global__ void thirdParallel()
{
  printf("This should be running in parallel 3.\n");
}

int main()
{

  firstParallel<<<1, 1>>>(); // printará 1 vez

  secondParallel<<<1, 5>>>(); // printará 5 vezes

  thirdParallel<<<5, 5>>>(); // printará 25 vezes

  /*
   * Some code is needed below so that the CPU will wait
   * for the GPU kernels to complete before proceeding.
   */
   cudaDeviceSynchronize();

}
