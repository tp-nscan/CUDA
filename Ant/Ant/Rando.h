#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct RandData {
	curandGenerator_t rand_gen;
	unsigned int seed;
	unsigned int length;
	unsigned int index;
	float *dev_rands;
};


void InitRandData(RandData **out, int seed, int dataLen, float *dev_rands);

void UpdateRandData(RandData *randData);

void DeleteRandData(RandData *randData);

void RandTest(int argc, char **argv);
