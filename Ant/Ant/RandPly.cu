#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "Utils.h"
#include "BlockUtils.h"
#include "Rando.h"
#include "Ply.h"
#include "RandPly.h"

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)


////////////////////////////////////////////////////////////////////////////////
// randPly_local_update
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void randPly_local_update(float *resOut, float *plyIn, float *rands, int2 plySize,
	float speed, float noise) {

	int plyLength = plySize.x * plySize.y;
	int step = blockDim.x * gridDim.x;

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < plyLength; i += step)
	{
		int col = i % plySize.x;
		int row = (i - col) / plySize.x;
		int colm = (col - 1 + plySize.x) % plySize.x;
		int colp = (col + 1 + plySize.x) % plySize.x;
		int rowm = (row - 1 + plySize.y) % plySize.y;
		int rowp = (row + 1 + plySize.y) % plySize.y;

		int left = colm + row * plySize.x;
		int right = colp + row * plySize.x;
		int top = col + rowp * plySize.x;
		int bottom = col + rowm * plySize.x;
		int center = col + row * plySize.x;

		float t = plyIn[top];
		float l = plyIn[left];
		float c = plyIn[center];
		float r = plyIn[right];
		float b = plyIn[bottom];

		float res = c + speed * (t + b + r + l) + noise * rands[i];
		if (res > 1.0) {
			res = 1.0;
		}
		if (res < -1.0) {
			res = -1.0;
		}
		resOut[center] = res;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! MakeRandPly
////////////////////////////////////////////////////////////////////////////////
void MakeRandPly(RandPly **randPly, float *data, int seed, unsigned int plyLength, unsigned int span)
{
	RandPly *rp = (RandPly *)malloc(sizeof(RandPly));
	*randPly = rp;
	MakePly(&(rp->ply), data, plyLength, span);
	float *dev_rands;
	int dataSize = plyLength * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&dev_rands, dataSize));
	InitRandData(&(rp->randData), seed, plyLength, dev_rands);
}

////////////////////////////////////////////////////////////////////////////////
//! RunRandPlyLocalUpdate
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyLocalUpdate(RandPly *randPly, int num_steps, float speed, float noise)
{
	int blocks = SuggestedBlocks((randPly->ply->area + THREADS_1D - 1) / THREADS_1D);
	int2 plySize;
	plySize.x = randPly->ply->span; plySize.y = randPly->ply->span;

	for (int step = 0; step < num_steps; step++)
	{
		UpdateRandData(randPly->randData);

		if (randPly->ply->inToOut)
		{
			randPly_local_update << <blocks, THREADS_1D >> >(
				randPly->ply->dev_outSrc,
				randPly->ply->dev_inSrc,
				randPly->randData->dev_rands,
				plySize,
				speed,
				noise);
		}
		else
		{
			randPly_local_update << <blocks, THREADS_1D >> >(
				randPly->ply->dev_inSrc,
				randPly->ply->dev_outSrc,
				randPly->randData->dev_rands,
				plySize,
				speed,
				noise);
		}

		randPly->ply->inToOut = !randPly->ply->inToOut;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! GridEnergy
////////////////////////////////////////////////////////////////////////////////
float *GridEnergy(float *d_ply, int span, int dataLen)
{
	int blocks = SuggestedBlocks((dataLen + THREADS_1D - 1) / THREADS_1D);

	int plyMemSize = span * span * sizeof(float);
	int2 plySize;
	plySize.x = span; plySize.y = span;

	float *d_energies;
	checkCudaErrors(cudaMalloc((void**)&d_energies, plyMemSize));

	grid_local_energy << <blocks, THREADS_1D >> >(
		d_energies,
		d_ply,
		plySize);

	float *h_energies = (float *)malloc(plyMemSize);
	checkCudaErrors(cudaMemcpy(h_energies, d_energies, plyMemSize, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_energies));
	return h_energies;
}

////////////////////////////////////////////////////////////////////////////////
//! GridEnergyC
////////////////////////////////////////////////////////////////////////////////
float *GridEnergyC(float *d_ply, int span, int dataLen)
{
	int blocks = SuggestedBlocks((dataLen + THREADS_1D - 1) / THREADS_1D);

	int blockEnergySize = blocks * blocks * sizeof(float);
	int2 plySize;
	plySize.x = span; plySize.y = span;

	float *d_energies;
	checkCudaErrors(cudaMalloc((void**)&d_energies, blockEnergySize));

	grid_local_energyC << <blocks, THREADS_1D >> >(
		d_energies,
		d_ply,
		plySize);

	float *h_energies = (float *)malloc(blockEnergySize);
	checkCudaErrors(cudaMemcpy(h_energies, d_energies, blockEnergySize, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_energies));
	return h_energies;
}

////////////////////////////////////////////////////////////////////////////////
//! PlyEnergy
////////////////////////////////////////////////////////////////////////////////
float *PlyEnergy(Ply *ply)
{
	if (ply->inToOut)
	{
		return GridEnergy(ply->dev_inSrc, ply->span, ply->area);
	}
	else
	{
		return GridEnergy(ply->dev_outSrc, ply->span, ply->area);
	}
}


////////////////////////////////////////////////////////////////////////////////
//! PlyEnergyC
////////////////////////////////////////////////////////////////////////////////
float *PlyEnergyC(Ply *ply)
{
	if (ply->inToOut)
	{
		return GridEnergyC(ply->dev_inSrc, ply->span, ply->area);
	}
	else
	{
		return GridEnergyC(ply->dev_outSrc, ply->span, ply->area);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! DeleteRandPly
////////////////////////////////////////////////////////////////////////////////
void DeleteRandPly(RandPly *randPly)
{
	FreePly(randPly->ply);
	DeleteRandData(randPly->randData);
}


////////////////////////////////////////////////////////////////////////////////
//! RunEnergyTest
////////////////////////////////////////////////////////////////////////////////
void RunEnergyTest(int argc, char **argv)
{
	//dataLen=400 span=20 reps=50 seed=1243 speed=0.05 noise=0.01 batch=10
	int dataLen = IntNamed(argc, argv, "dataLen", 81);
	int span = IntNamed(argc, argv, "span", 9);

	float *d_pattern;
	float *h_pattern = CheckerArray(span);
	PrintFloatArray(h_pattern, span, span*span);

	int plyMemSize = span * span * sizeof(float);

	checkCudaErrors(cudaMalloc((void**)&d_pattern, plyMemSize));
	checkCudaErrors(cudaMemcpy(d_pattern, h_pattern, plyMemSize, cudaMemcpyHostToDevice));

	float *h_energies = GridEnergy(d_pattern, span, dataLen);

	printf("Energies:\n");
	PrintFloatArray(h_energies, span, span*span);
}


////////////////////////////////////////////////////////////////////////////////
//! RunEnergyTestC
////////////////////////////////////////////////////////////////////////////////
void RunEnergyTestC(int argc, char **argv)
{
	//dataLen=400 span=20 reps=50 seed=1243 speed=0.05 noise=0.01 batch=10
	int dataLen = IntNamed(argc, argv, "dataLen", 81);
	int span = IntNamed(argc, argv, "span", 9);


	int blocks = SuggestedBlocks((dataLen + THREADS_1D - 1) / THREADS_1D);

	float *d_pattern;
	float *h_pattern = CheckerArray(span);
	PrintFloatArray(h_pattern, span, span*span);

	int plyMemSize = span * span * sizeof(float);

	checkCudaErrors(cudaMalloc((void**)&d_pattern, plyMemSize));
	checkCudaErrors(cudaMemcpy(d_pattern, h_pattern, plyMemSize, cudaMemcpyHostToDevice));

	float *h_energies = GridEnergyC(d_pattern, span, dataLen);

	//float *h_energies = (float *)malloc(plyMemSize);
	//checkCudaErrors(cudaMemcpy(h_energies, d_pattern, plyMemSize, cudaMemcpyDeviceToHost));

	printf("Energies:\n");
	PrintFloatArray(h_energies, 1, blocks);
}

float *RunRandPly(int span, float speed, float noise, int seed, float *h_data, int reps)
{
	RandPly *randPly;
	MakeRandPly(&(randPly), h_data, seed, span*span, span);
	RunRandPlyLocalUpdate(randPly, reps, speed, noise);
	float *h_results = (float *)malloc(span * span * sizeof(float));
	GetPlyData(randPly->ply, h_results);
	DeleteRandPly(randPly);

	return h_results;
}

////////////////////////////////////////////////////////////////////////////////
//! RunRandPlyBench
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyBench(int argc, char **argv)
{
	//dataLen=1048576 span=1024 reps=100 seed=6423 speed=0.1 noise=0.01 batch=1
	//dataLen=400 span=20 reps=50 seed=1243 speed=0.05 noise=0.01 batch=10
	int dataLen = IntNamed(argc, argv, "dataLen", 36);
	int span = IntNamed(argc, argv, "span", 6);
	int seed = IntNamed(argc, argv, "seed", 12);
	int reps = IntNamed(argc, argv, "reps", 16);
	float speed = FloatNamed(argc, argv, "speed", 0.05);
	float noise = FloatNamed(argc, argv, "noise", 0.01);
	int batch = IntNamed(argc, argv, "batch", 10);
	printf("dataLen: %d  batch: %d  speed: %3.4f  noise: %3.4f \n", dataLen, batch, speed, noise);

	float *h_samples;

	int dataSize = dataLen * sizeof(float);
	h_samples = CheckerArray(span);

	RandPly *randPly;
	MakeRandPly(&(randPly), h_samples, seed, dataLen, span);

	cudaEvent_t start, stop;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {
		RunRandPlyLocalUpdate(randPly, batch, speed, noise);		
		float *out = PlyEnergy(randPly->ply);
		printf("Energy:  %3.4f\n", FloatArraySum(out, dataLen));
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time:  %3.1f ms\n", elapsedTime);
	GetPlyData(randPly->ply, h_samples);
	//PrintFloatArray(h_samples, span, dataLen);

	DeleteRandPly(randPly);
}


