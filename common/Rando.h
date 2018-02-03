#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct RandData {
	curandState *dev_curandStates;
	dim3 blocks;
	dim3 threads;
	unsigned int seed;
};

cudaError_t Gpu_InitRNG_1d(RandData *randData);

cudaError_t Gpu_InitRNG_2d(RandData *randData);

cudaError_t Gpu_UniformRandFloats_1d(float **dev_rands, RandData *randData, int numRands);

cudaError_t Gpu_NormalRandFloats_1d(float **dev_rands, RandData *randData, int numRands);

cudaError_t Gpu_UniformRandFloats_2d(float **dev_rands, RandData *randData, int numRands);

cudaError_t Gpu_NormalRandFloats_2d(float **dev_rands, RandData *randData, int numRands);
