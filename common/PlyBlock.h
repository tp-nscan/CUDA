#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Rando.h"

//globals needed by the update routine
struct PlyBlock {
	int width;
	int area;

	float           *dev_inSrc;
	float           *dev_outSrc;
	float           *dev_constSrc;

	cudaTextureObject_t *texIn;
	cudaTextureObject_t *texOut;
	cudaTextureObject_t *texConst;

	RandData *randData;
};

void MakePlyBlock(PlyBlock **out, unsigned int width, unsigned int seed);