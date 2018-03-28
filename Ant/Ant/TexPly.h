#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//globals needed by the update routine
struct TexPly {
	unsigned int length;
	bool inToOut;
	float           *dev_inSrc;
	float           *dev_outSrc;

	cudaTextureObject_t *texIn;
	cudaTextureObject_t *texOut;
};

void GetTexPlyData(TexPly *texPly, float *data_outSrc);

void RunTexPly(TexPly *texPly, int2 plySize, float speed, int blocks, int threads, float decay);

void MakeTexPly(TexPly **out, unsigned int length);

void MakeTexPly(TexPly **out, float *data, unsigned int length);

void FreeTexPly(TexPly *plyBlock);

void RunTexPlyCorrectness(int argc, char **argv);

void RunTexPlyBench(int argc, char **argv);