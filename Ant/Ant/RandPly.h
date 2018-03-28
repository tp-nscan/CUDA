#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Ply.h"

//globals needed by the update routine
struct RandPly {
	Ply *ply;
	RandData *randData;
};

void MakeRandPly(RandPly **randPly, float *data, int seed, unsigned int plyLength);

void RunRandPly(RandPly *randPly);

void DeleteRandPly(RandPly *randPly);

void RunRandPlyCorrectness(int argc, char **argv);

void RunRandPlyBench(int argc, char **argv);

//globals needed by the update routine
struct RandPly2 {
	Ply *ply;
	RandData2 *randData;
};

void MakeRandPly2(RandPly2 **randPly, float *data, int seed, unsigned int plyLength);

void RunRandPly2(RandPly2 *randPly);

void DeleteRandPly2(RandPly2 *randPly);

void RunRandPlyCorrectness2(int argc, char **argv);

void RunRandPlyBench2(int argc, char **argv);