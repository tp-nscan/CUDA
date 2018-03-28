#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//globals needed by the update routine
struct Ply {
	unsigned int area;
	unsigned int span;
	bool inToOut;
	float           *dev_inSrc;
	float           *dev_outSrc;
};

void GetPlyData(Ply *texPly, float *data_outSrc);

void RunPly(Ply *texPly, int2 plySize, float speed, int blocks, int threads, float decay);

void MakePly(Ply **out, unsigned int plyLength, unsigned int span);

void MakePly(Ply **out, float *data, unsigned int plyLength, unsigned int span);

void FreePly(Ply *plyBlock);

void RunPlyCorrectness(int argc, char **argv);

void RunPlyBench(int argc, char **argv);