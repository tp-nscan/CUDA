#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Ply.h"


struct RandPly {
	Ply *ply;
	RandData *randData;
};

void MakeRandPly(RandPly **randPly, float *data, int seed, unsigned int plyLength, unsigned int span);

void RunRandPlyLocalUpdate(RandPly *randPly, int num_steps, float speed, float noise);

void RunRandPlyCume(RandPly *randPly, int num_steps);

void DeleteRandPly(RandPly *randPly);

void RunRandPlyCorrectness(int argc, char **argv);

void RunRandPlyBench(int argc, char **argv);

void BreakDown();