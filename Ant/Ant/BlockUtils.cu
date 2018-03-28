#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#define MAX_BLOCKS_1D  16384
#define MAX_BLOCKS_2D  128

dim3 *dim3Ctr(int x, int y = 1, int z = 1)
{
	dim3 *a;
	a = (dim3 *)malloc(sizeof(dim3));
	a->x = x;
	a->y = y;
	a->z = z;
	return a;
}

dim3 *dim3Unit()
{
	dim3 *a;
	a = (dim3 *)malloc(sizeof(dim3));
	a->x = 1;
	a->y = 1;
	a->z = 1;
	return a;
}

int dim3Vol(dim3 *a)
{
	return a->x * a->y * a->z;
}

void printDim3(dim3 *yow)
{
	printf("yow: {%d, %d, %d}", yow->x, yow->y, yow->z);
}

int SuggestedBlocks(int blocks_needed)
{
	if (blocks_needed > 512)
		return 512;

	if (blocks_needed < 1)
		return 1;

	return blocks_needed;
}

int ThreadChop1d(int size)
{
	if (size > 65536)
		return 256;
	if (size > 16384)
		return 128;
	if (size > 4096)
		return 64;
	if (size > 1024)
		return 32;
	if (size > 255)
		return 16;
	if (size > 63)
		return 8;
	if (size > 15)
		return 4;
	if (size > 3)
		return 2;
	return 1;
}


int ThreadChop2d(int width)
{
	if (width > 255)
		return 16;
	if (width > 63)
		return 8;
	if (width > 15)
		return 4;
	if (width > 3)
		return 2;
	return 1;
}

void GridAndBlocks1d(dim3 &grid, dim3 &block, int size)
{
	int threadCt = ThreadChop1d(size);
	int blockCt = MinInt((size + threadCt - 1) / threadCt, MAX_BLOCKS_1D);
	grid = dim3(blockCt);
	block = dim3(threadCt);
}

void GridAndBlocks2d(dim3 &grid, dim3 &block, int width)
{
	int threadCt = ThreadChop2d(width);
	int blockCt = MinInt((width + threadCt - 1) / threadCt, MAX_BLOCKS_2D);
	grid = dim3(blockCt, blockCt);
	block = dim3(threadCt, threadCt);
}