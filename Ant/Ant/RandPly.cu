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
		int top = col + rowm * plySize.x;
		int bottom = col + rowp * plySize.x;
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
// randPly_local_energy
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void randPly_local_energy(float *resOut, float *plyIn, int2 plySize) {

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
		int top = col + rowm * plySize.x;
		int bottom = col + rowp * plySize.x;
		int center = col + row * plySize.x;

		float t = plyIn[top];
		float l = plyIn[left];
		float c = plyIn[center];
		float r = plyIn[right];
		float b = plyIn[bottom];

		float res = c * (t + b + r + l);
		resOut[center] = c * res;
	}
}

////////////////////////////////////////////////////////////////////////////////
// randPly_cume
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void randPly_cume(float *resOut, float *arrayIn,
	float *randoms, int plyLength) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;

	for (int i = tid; i < plyLength; i += step)
	{
		resOut[i] = arrayIn[i] + randoms[i];
		//resOut[i] = arrayIn[i] + 1;
	}
}


////////////////////////////////////////////////////////////////////////////////
//! RunRandPlyLocalUpdate
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
//! RunRandPlyCume
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyLocalUpdate(RandPly *randPly, int num_steps, float speed, float noise)
{
	const int threads = 512;
	int dataLen = randPly->ply->area;
	int blocks = SuggestedBlocks(dataLen / threads);
	int2 plySize;
	plySize.x = randPly->ply->span; plySize.y = randPly->ply->span;

	for (int step = 0; step < num_steps; step++)
	{
		UpdateRandData(randPly->randData);

		if (randPly->ply->inToOut)
		{
			randPly_local_update << <blocks, threads >> >(
				randPly->ply->dev_outSrc,
				randPly->ply->dev_inSrc,
				randPly->randData->dev_rands,
				plySize,
				speed,
				noise);
		}
		else
		{
			randPly_local_update << <blocks, threads >> >(
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
//! RunRandPly
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyCume(RandPly *randPly, int num_steps)
{
	const int threads = 512;
	const float speed = 0.1;
	const float noise = 0.01;
	int dataLen = randPly->ply->area;
	int blocks = SuggestedBlocks(dataLen / threads);
	int2 plySize;
	plySize.x = randPly->ply->span; plySize.y = randPly->ply->span;

	for (int step = 0; step < num_steps; step++)
	{
		UpdateRandData(randPly->randData);

		if (randPly->ply->inToOut)
		{
			randPly_cume << <blocks, threads >> >(
				randPly->ply->dev_outSrc,
				randPly->ply->dev_inSrc,
				randPly->randData->dev_rands,
				dataLen);
		}
		else
		{
			randPly_cume << <blocks, threads >> >(
				randPly->ply->dev_inSrc,
				randPly->ply->dev_outSrc,
				randPly->randData->dev_rands,
				dataLen);
		}

		randPly->ply->inToOut = !randPly->ply->inToOut;
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
//! RunRandPlyCorrectness
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyCorrectness(int argc, char **argv)
{
	//dataLen=36 span=6 reps=20 seed=1243 speed=0.05 noise=0.01
	int dataLen = IntNamed(argc, argv, "dataLen", 36);
	int span = IntNamed(argc, argv, "span", 6);
	int seed = IntNamed(argc, argv, "seed", 12);
	int reps = IntNamed(argc, argv, "reps", 16);
	float speed = FloatNamed(argc, argv, "speed", 0.05);
	float noise = FloatNamed(argc, argv, "noise", 0.01);

	float *h_samples;

	int dataSize = dataLen * sizeof(float);
	h_samples = FloatArray(dataLen);

	RandPly *randPly;
	MakeRandPly(&(randPly), h_samples, seed, dataLen, span);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	GetPlyData(randPly->ply, h_samples);
	PrintFloatArray(h_samples, span, dataLen);

	for (int i = 0; i < reps; i++) {
		RunRandPlyLocalUpdate(randPly, 1, speed, noise);
		GetPlyData(randPly->ply, h_samples);
		PrintFloatArray(h_samples, span, dataLen);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time:  %3.1f ms\n", elapsedTime);
	DeleteRandPly(randPly);
}


////////////////////////////////////////////////////////////////////////////////
//! RunRandPlyBench
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyBench(int argc, char **argv)
{
	//dataLen=36 span=6 reps=20 seed=1243 speed=0.05 noise=0.01
	int dataLen = IntNamed(argc, argv, "dataLen", 36);
	int span = IntNamed(argc, argv, "span", 6);
	int seed = IntNamed(argc, argv, "seed", 12);
	int reps = IntNamed(argc, argv, "reps", 16);
	float speed = FloatNamed(argc, argv, "speed", 0.05);
	float noise = FloatNamed(argc, argv, "noise", 0.01);

	float *h_samples;

	int dataSize = dataLen * sizeof(float);
	h_samples = FloatArray(dataLen);

	RandPly *randPly;
	MakeRandPly(&(randPly), h_samples, seed, dataLen, span);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {
		RunRandPlyLocalUpdate(randPly, 1, speed, noise);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time:  %3.1f ms\n", elapsedTime);
	GetPlyData(randPly->ply, h_samples);
	PrintFloatArray(h_samples, span, dataLen);

	DeleteRandPly(randPly);
}


void BreakDown()
{
	int dataLen = 36;
	int span = 6;
	int seed = 1234;
	float *h_samples;

	int dataSize = dataLen * sizeof(float);
	h_samples = FloatArray(dataLen);

	RandPly *randPly;
	MakeRandPly(&(randPly), h_samples, seed, dataLen, span);
	UpdateRandData(randPly->randData);
	//randPly_cume << <2, 2 >> >(
	//	randPly->ply->dev_outSrc,
	//	randPly->ply->dev_inSrc,
	//	randPly->randData->dev_rands,
	//	dataLen);

	checkCudaErrors(cudaMemcpy(h_samples, randPly->ply->dev_inSrc, dataSize, cudaMemcpyDeviceToHost));
	PrintFloatArray(h_samples, span, dataLen);

	checkCudaErrors(cudaMemcpy(h_samples, randPly->ply->dev_outSrc, dataSize, cudaMemcpyDeviceToHost));
	PrintFloatArray(h_samples, span, dataLen);

	checkCudaErrors(cudaMemcpy(h_samples, randPly->randData->dev_rands, dataSize, cudaMemcpyDeviceToHost));
	PrintFloatArray(h_samples, span, dataLen);
}

