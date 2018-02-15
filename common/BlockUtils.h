#pragma once
#include "cuda_runtime.h"
#include <stdlib.h>

dim3 *dim3Ctr(int x, int y = 1, int z = 1);
dim3 *dim3Unit();
int dim3Vol(dim3 *a);
void printDim3(dim3 *yow);

__device__ __inline__ int2 NodeInTilePos()
{
	int2 step;
	step.x = threadIdx.x + blockIdx.x * blockDim.x;
	step.y = threadIdx.y + blockIdx.y * blockDim.y;
	return step;
}