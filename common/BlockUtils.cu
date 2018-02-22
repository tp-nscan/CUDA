#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>


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
