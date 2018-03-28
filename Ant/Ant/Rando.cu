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

void InitRandData2(RandData2 **out, int seed, int dataLen, float *dev_rands)
{
	curandStatus_t status;

	RandData2 *randData2 = (RandData2 *)malloc(sizeof(RandData2));
	*out = randData2;
	randData2->index = 0;
	randData2->seed = seed;
	randData2->length = dataLen;
	randData2->dev_rands = dev_rands;
	int dataSize = dataLen * sizeof(float);

	status = curandCreateGenerator(&(randData2->rand_gen), CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(randData2->rand_gen, seed);

	getLastCudaError("InitRandData2 execution failed");
}


void UpdateRandData2(RandData2 *randData2)
{
	curandGenerateUniform(randData2->rand_gen, randData2->dev_rands, randData2->length);
	randData2->index++;
	getLastCudaError("MakeRandData execution failed");
}


void MakeRandData(RandData **out, int seed, int length)
{
	RandData *randData = (RandData *)malloc(sizeof(RandData));
	*out = randData;

	checkCudaErrors(cudaMalloc((void**)&randData->dev_curandStates, length * sizeof(curandState)));

	dim3 g, b;
	GridAndBlocks1d(g, b, length);
	initRNG <<<g, b>> >(randData->dev_curandStates, seed, length);

	getLastCudaError("MakeRandData execution failed");
	randData->seed = seed;
	randData->length = length;
}


void DeleteRandData(RandData *randData)
{
	cudaFree(randData->dev_curandStates);
	free(randData);
}


void DeleteRandData2(RandData2 *randData)
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


void Gpu_UniformRandFloats(float **dev_rands, RandData *randData, int numRands)
{
	dim3 g, b;
	GridAndBlocks1d(g, b, randData->length);

	float *dev_r;
	checkCudaErrors(cudaMalloc((void**)&dev_r, numRands * sizeof(float)));

	rndGenUniform <<<g, b>>>(dev_r, randData->dev_curandStates, numRands);

	// Check for any errors launching the kernel
	getLastCudaError("Gpu_UniformRandFloats execution failed");

	checkCudaErrors(cudaDeviceSynchronize());
	*dev_rands = dev_r;
}


void Gpu_NormalRandFloats(float **dev_rands, RandData *randData, int numRands)
{
	dim3 g, b;
	GridAndBlocks1d(g, b, randData->length);

	float *dev_r;
	checkCudaErrors(cudaMalloc((void**)&dev_r, numRands * sizeof(float)));

	rndGenNormal << <g, b >> >(dev_r, randData->dev_curandStates, numRands);

	// Check for any errors launching the kernel
	getLastCudaError("Gpu_UniformRandFloats execution failed");

	checkCudaErrors(cudaDeviceSynchronize());
	*dev_rands = dev_r;
}



////////////////////////////////////////////////////////////////////////////////
//! Make and test RandData
////////////////////////////////////////////////////////////////////////////////
void RandTest(int argc, char **argv)
{
	int seed = IntNamed(argc, argv, "seed", 1234);
	int randStateLength = IntNamed(argc, argv, "randStateLength", 512);
	int randLength = IntNamed(argc, argv, "randLength", 512);
	int reps = IntNamed(argc, argv, "reps", 512);

	RandData *randData;
	MakeRandData(&randData, seed, randStateLength);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {

		Gpu_UniformRandFloats(&dev_rands, randData, randLength);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Uniform:  %3.1f ms\n", elapsedTime);

	DeleteRandData(randData);
}

////////////////////////////////////////////////////////////////////////////////
//! Make and test RandData2
////////////////////////////////////////////////////////////////////////////////
void RandTest2(int argc, char **argv)
{
	//dataLen=10 reps=10 seed=1243
	int seed = IntNamed(argc, argv, "seed", 1234);
	int dataLen = IntNamed(argc, argv, "dataLen", 512);
	int reps = IntNamed(argc, argv, "reps", 512);

	float *dev_data;
	int dataSize = dataLen * sizeof(float);
	checkCudaErrors(cudaMalloc((void **)&dev_data, dataSize));

	RandData2 *randData2;
	InitRandData2(&randData2, seed, dataLen, dev_data);

	float *h_samples = (float *)malloc(dataSize);

	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {
		UpdateRandData2(randData2);
		checkCudaErrors(cudaMemcpy(h_samples, randData2->dev_rands, dataSize, cudaMemcpyDeviceToHost));
		PrintFloatArray(h_samples, 1, dataLen);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Uniform:  %3.1f ms\n", elapsedTime);

	///DeleteRandData(randData);
}
