#pragma once
#include "cuda_runtime.h"
#include <stdlib.h>

dim3 *dim3Ctr(int x, int y = 1, int z = 1);
dim3 *dim3Unit();
int dim3Vol(dim3 *a);
void printDim3(dim3 *yow);
int SuggestedBlocks(int blocks_needed);
int ThreadChop1d(int width);
int ThreadChop2d(int width);
void GridAndBlocks1d(dim3 &grid, dim3 &block, int size);
void GridAndBlocks2d(dim3 &grid, dim3 &block, int width);

__device__ __inline__ int2 NodeInTilePos()
{
	int2 step;
	step.x = threadIdx.x + blockIdx.x * blockDim.x;
	step.y = threadIdx.y + blockIdx.y * blockDim.y;
	return step;
}

__device__ __inline__ int2 TileSize()
{
	int2 step;
	step.x = blockDim.x* gridDim.x;
	step.y = blockDim.y* gridDim.y;
	return step;
}

__device__ __inline__ int NodeIndexInPly(int2 nitDex, int2 tipDex, int2 tileSize, int2 plySize)
{
	return
		tileSize.y + plySize.x * tipDex.y +
		tileSize.y + tileSize.x * tipDex.x +
		tileSize.x + nitDex.y +
		nitDex.x;
}

__device__ __inline__ int2 NodeInPlyPos(int2 nitDex, int2 tipDex, int2 tileSize)
{
	int2 step;
	step.x = tileSize.x * tipDex.x + nitDex.x;
	step.y = tileSize.y * tipDex.y + nitDex.y;
	return step;
}

