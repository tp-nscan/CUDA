#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct RandData2 {
	curandGenerator_t rand_gen;
	unsigned int seed;
	unsigned int length;
	unsigned int index;
	float *dev_rands;
};


struct RandData {
	curandState *dev_curandStates;
	unsigned int seed;
	unsigned int length;
};

void InitRandData2(RandData2 **out, int seed, int dataLen, float *dev_rands);

void UpdateRandData2(RandData2 *randData2);

void MakeRandData(RandData **out, int seed, int length);

void DeleteRandData(RandData *randData);

void DeleteRandData2(RandData2 *randData);

void Gpu_UniformRandFloats(float **dev_rands, RandData *randData, int numRands);

void Gpu_NormalRandFloats(float **dev_rands, RandData *randData, int numRands);

void RandTest(int argc, char **argv);

void RandTest2(int argc, char **argv);

