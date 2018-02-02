#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct RandData {
	curandState *dev_rngs;
	dim3 blocks;
	dim3 threads;
	unsigned int seed;
};

cudaError_t Gpu_InitRNG(curandState **dev_rngs, RandData *randData);

//cudaError_t Gpu_InitRNG(curandState **dev_rngs, dim3 blocks,
//	dim3 threads, unsigned int seed);

cudaError_t Gpu_UniformRandFloats(float **dev_rands, RandData *randData, int numRands);

cudaError_t Gpu_NormalRandFloats(float **dev_rands, RandData *randData, int numRands);