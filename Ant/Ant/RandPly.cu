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
// randPly_blend_1d_2d_1d
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void randPly_blend_1d_2d_1d(float *resOut, float *arrayIn, int2 plySize,
	float speed, float decay) {

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

		float t = arrayIn[top];
		float l = arrayIn[left];
		float c = arrayIn[center];
		float r = arrayIn[right];
		float b = arrayIn[bottom];

		resOut[center] = c + speed * (t + b + r + l - decay * c);
	}
}

////////////////////////////////////////////////////////////////////////////////
// randPly_cume_1d_2d_1d
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void randPly_cume_1d_2d_1d(float *resOut, float *arrayIn, 
	curandState *const rngStates, int plyLength) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;

	for (int i = tid; i < plyLength; i += step)
	{
		curandState localState = rngStates[tid];
		resOut[i] = curand_uniform(&localState) + arrayIn[i];
		//resOut[i] = arrayIn[i] + 1;
	}
}

////////////////////////////////////////////////////////////////////////////////
// randPly_cume_1d_2d_1d
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void randPly2_cume_1d_2d_1d(float *resOut, float *arrayIn,
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
//! MakeRandPly
////////////////////////////////////////////////////////////////////////////////
void MakeRandPly(RandPly **randPly, float *data, int seed, unsigned int plyLength)
{
	RandPly *rp = (RandPly *)malloc(sizeof(RandPly));
	*randPly = rp;
	MakePly(&(rp->ply), data, plyLength);
	MakeRandData(&(rp->randData), seed, plyLength);
}

////////////////////////////////////////////////////////////////////////////////
//! MakeRandPly2
////////////////////////////////////////////////////////////////////////////////
void MakeRandPly2(RandPly2 **randPly, float *data, int seed, unsigned int plyLength)
{
	RandPly2 *rp = (RandPly2 *)malloc(sizeof(RandPly2));
	*randPly = rp;
	MakePly(&(rp->ply), data, plyLength);
	float *dev_rands;
	int dataSize = plyLength * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&dev_rands, dataSize));
	InitRandData2(&(rp->randData), seed, plyLength, dev_rands);
}

////////////////////////////////////////////////////////////////////////////////
//! RunRandPly
////////////////////////////////////////////////////////////////////////////////
void RunRandPly(RandPly *randPly)
{
	const int threads = 2; //512;
	int dataLen = randPly->ply->length;
	int blocks = 1; // SuggestedBlocks(dataLen / threads);
	if (randPly->ply->inToOut)
	{
		randPly_cume_1d_2d_1d << <blocks, threads >> >(randPly->ply->dev_outSrc, randPly->ply->dev_inSrc,
			randPly->randData->dev_curandStates, dataLen);
		randPly->ply->inToOut = false;
	}
	else
	{
		randPly_cume_1d_2d_1d << <blocks, threads >> >(randPly->ply->dev_inSrc, randPly->ply->dev_outSrc,
			randPly->randData->dev_curandStates, dataLen);
		randPly->ply->inToOut = true;
	}
	getLastCudaError("RunRandPly execution failed");
}

////////////////////////////////////////////////////////////////////////////////
//! RunRandPly2
////////////////////////////////////////////////////////////////////////////////
void RunRandPly2(RandPly2 *randPly)
{
	const int threads = 512;
	int dataLen = randPly->ply->length;
	int blocks = SuggestedBlocks(dataLen / threads);
	if (randPly->ply->inToOut)
	{
		UpdateRandData2(randPly->randData);
		randPly2_cume_1d_2d_1d << <blocks, threads >> >(randPly->ply->dev_outSrc, randPly->ply->dev_inSrc,
			randPly->randData->dev_rands, dataLen);
		randPly->ply->inToOut = false;
	}
	else 
	{
		UpdateRandData2(randPly->randData);
		randPly2_cume_1d_2d_1d << <blocks, threads >> >(randPly->ply->dev_inSrc, randPly->ply->dev_outSrc,
			randPly->randData->dev_rands, dataLen);
		randPly->ply->inToOut = true;
	}
	getLastCudaError("RunRandPly execution failed");
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
//! DeleteRandPly2
////////////////////////////////////////////////////////////////////////////////
void DeleteRandPly2(RandPly2 *randPly)
{
	FreePly(randPly->ply);
	DeleteRandData2(randPly->randData);
}


////////////////////////////////////////////////////////////////////////////////
//! RunRandPlyCorrectness
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyCorrectness(int argc, char **argv)
{
	//dataLen=12 reps=20 seed=1243
	int dataLen = IntNamed(argc, argv, "dataLen", 12);
	int seed = IntNamed(argc, argv, "seed", 12);
	int reps = IntNamed(argc, argv, "reps", 16);
	float *d_samples, *h_samples;
	curandStatus_t status;
	curandGenerator_t rand_gen;

	int dataSize = dataLen * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&d_samples, dataSize));

	h_samples = (float *)malloc(dataSize);

	status = curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(rand_gen, seed);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {
		status = curandGenerateUniform(rand_gen, d_samples, dataLen);
		checkCudaErrors(cudaMemcpy(h_samples, d_samples, dataSize, cudaMemcpyDeviceToHost));
		PrintFloatArray(h_samples, 1, dataLen);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time:  %3.1f ms\n", elapsedTime);
	curandDestroyGenerator(rand_gen);
}


////////////////////////////////////////////////////////////////////////////////
//! RunRandPlyCorrectness2
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyCorrectness2(int argc, char **argv)
{
	//dataLen=12 reps=20 seed=1243
	int dataLen = IntNamed(argc, argv, "dataLen", 12);
	int seed = IntNamed(argc, argv, "seed", 12);
	int reps = IntNamed(argc, argv, "reps", 16);
	float *h_samples;

	int dataSize = dataLen * sizeof(float);
	h_samples = (float *)malloc(dataSize);

	RandPly2 *randPly2;
	MakeRandPly2(&(randPly2), h_samples, seed, dataLen);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {
		RunRandPly2(randPly2);
		GetPlyData(randPly2->ply, h_samples);
		PrintFloatArray(h_samples, 1, dataLen);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time:  %3.1f ms\n", elapsedTime);
	DeleteRandPly2(randPly2);
}


////////////////////////////////////////////////////////////////////////////////
//! RunRandPlyBench
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyBench(int argc, char **argv)
{
	//dataLen=12 reps=20 seed=1243
	int dataLen = IntNamed(argc, argv, "dataLen", 12);
	int seed = IntNamed(argc, argv, "seed", 12);
	int reps = IntNamed(argc, argv, "reps", 16);
	float *d_samples, *h_samples;
	curandStatus_t status;
	curandGenerator_t rand_gen;

	int dataSize = dataLen * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&d_samples, dataSize));

	h_samples = (float *)malloc(dataSize);

	status = curandCreateGenerator(&rand_gen, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(rand_gen, seed);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {
		curandGenerateUniform(rand_gen, d_samples, dataLen);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time:  %3.1f ms\n", elapsedTime);
	curandDestroyGenerator(rand_gen);
}


////////////////////////////////////////////////////////////////////////////////
//! RunRandPlyBench2
////////////////////////////////////////////////////////////////////////////////
void RunRandPlyBench2(int argc, char **argv)
{
	//dataLen=12 reps=20 seed=1243
	int dataLen = IntNamed(argc, argv, "dataLen", 12);
	int seed = IntNamed(argc, argv, "seed", 12);
	int reps = IntNamed(argc, argv, "reps", 16);
	float *h_samples;

	int dataSize = dataLen * sizeof(float);
	h_samples = (float *)malloc(dataSize);

	RandPly2 *randPly2;
	MakeRandPly2(&(randPly2), h_samples, seed, dataLen);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {
		RunRandPly2(randPly2);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time:  %3.1f ms\n", elapsedTime);
	DeleteRandPly2(randPly2);
}
