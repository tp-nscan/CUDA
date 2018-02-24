#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


struct RandData {
	curandState *dev_curandStates;
	unsigned int seed;
	unsigned int length;
};

void InitRandData(RandData **out, int seed, int length);

void DeleteRandData(RandData *randData);

void Gpu_UniformRandFloats(float **dev_rands, RandData *randData, int numRands);

void Gpu_NormalRandFloats(float **dev_rands, RandData *randData, int numRands);

//void Gpu_UniformRandFloats_2d(float **dev_rands, RandData *randData, int numRands);
//
//void Gpu_NormalRandFloats_2d(float **dev_rands, RandData *randData, int numRands);
