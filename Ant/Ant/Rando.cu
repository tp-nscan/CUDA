#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "Utils.h"
#include "BlockUtils.h"
#include "Rando.h"
#include <ctime>

////////////////////////////////////////////////////////////////////////////////
//! RandData Initialization
////////////////////////////////////////////////////////////////////////////////

__global__ void initRNG(curandState *const rngStates, const unsigned int seed,
	const unsigned int length)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int step = gridDim.x * blockDim.x;

	for (unsigned int i = tid; i < length; i += step)
	{
		curand_init(seed, tid, 0, &rngStates[tid]);
		//curand_init((unsigned long long)clock() + i, 0, 0, &rngStates[tid]);
	}
}


void InitRandData(RandData **out, int seed, int dataLen, float *dev_rands)
{
	curandStatus_t status;

	RandData *randData = (RandData *)malloc(sizeof(RandData));
	*out = randData;
	randData->index = 0;
	randData->seed = seed;
	randData->length = dataLen;
	randData->dev_rands = dev_rands;
	int dataSize = dataLen * sizeof(float);

	status = curandCreateGenerator(&(randData->rand_gen), CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(randData->rand_gen, seed);

	getLastCudaError("InitRandData execution failed");
}


void UpdateRandData(RandData *randData)
{
	//curandGenerateUniform(randData->rand_gen, randData->dev_rands, randData->length);
	curandGenerateNormal(randData->rand_gen, randData->dev_rands, randData->length, 0.0, 1.0);
	randData->index++;
	getLastCudaError("MakeRandData execution failed");
}


void DeleteRandData(RandData *randData)
{
	cudaFree(randData->dev_rands);
	curandDestroyGenerator(randData->rand_gen);
	free(randData);
}


////////////////////////////////////////////////////////////////////////////////
// Rand Generation
////////////////////////////////////////////////////////////////////////////////


__global__ void rndGenNormal(float *const results,  curandState *const rngStates,
                                const unsigned int numRands)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int step = gridDim.x * blockDim.x;

	for (unsigned int i = tid; i < numRands; i += step)
	{
		curandState localState = rngStates[tid];
		results[i] = curand_normal(&localState);
	}
}

__global__ void rndGenUniform(float *const results, curandState *const rngStates, 
	                             const unsigned int numRands)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int step = gridDim.x * blockDim.x;

	for (unsigned int i = tid; i < numRands; i += step)
	{
		curandState localState = rngStates[tid];
		results[i] = curand_uniform(&localState);
	}
}


__device__ inline void getPoint(float &x, float &y, curandState &state)
{
	x = curand_uniform(&state);
	y = curand_uniform(&state);
}

__device__ inline void getPoint(double &x, double &y, curandState &state)
{
	x = curand_uniform_double(&state);
	y = curand_uniform_double(&state);
}


////////////////////////////////////////////////////////////////////////////////
//! Make and test RandData
////////////////////////////////////////////////////////////////////////////////
void RandTest(int argc, char **argv)
{
	//dataLen=10 reps=10 seed=1243
	int seed = IntNamed(argc, argv, "seed", 1234);
	int dataLen = IntNamed(argc, argv, "dataLen", 512);
	int reps = IntNamed(argc, argv, "reps", 512);

	float *dev_data;
	int dataSize = dataLen * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&dev_data, dataSize));

	RandData *randData;
	InitRandData(&randData, seed, dataLen, dev_data);

	float *h_samples = (float *)malloc(dataSize);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {
		UpdateRandData(randData);
		checkCudaErrors(cudaMemcpy(h_samples, randData->dev_rands, dataSize, cudaMemcpyDeviceToHost));
		PrintFloatArray(h_samples, 1, dataLen);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Uniform:  %3.1f ms\n", elapsedTime);

	DeleteRandData(randData);
}
