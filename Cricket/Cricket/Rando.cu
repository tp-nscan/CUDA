#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//#include "Utils.h"
#include "BlockUtils.h"
#include "Rando.h"
#include "KbLoop.cuh"

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
	}
}


void InitRandData(RandData **out, int seed, int length)
{
	RandData *randData = (RandData *)malloc(sizeof(RandData));
	*out = randData;

	checkCudaErrors(cudaMalloc((void**)&randData->dev_curandStates, length * sizeof(curandState)));

	dim3 g, b;
	GridAndBlocks1d(g, b, length);
	initRNG <<<g, b>> >(randData->dev_curandStates, seed, length);

	getLastCudaError("InitRandData execution failed");
	randData->seed = seed;
	randData->length = length;
}


void DeleteRandData(RandData *randData)
{
	cudaFree(randData->dev_curandStates);
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

//
//__global__ void rndGenNormal_2d(float *const results, curandState *const rngStates,
//	const unsigned int numRands)
//{
//	unsigned int tid =
//		blockIdx.x * gridDim.y * blockDim.x * blockDim.y +
//		blockIdx.y * blockDim.x * blockDim.y +
//		threadIdx.x * blockDim.y +
//		threadIdx.y;
//	unsigned int step = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
//
//	for (unsigned int i = tid; i < numRands; i += step)
//	{
//		curandState localState = rngStates[tid];
//		results[i] = curand_normal(&localState);
//	}
//}
//
//__global__ void rndGenUniform_2d(float *const results, curandState *const rngStates,
//	const unsigned int numRands)
//{
//	unsigned int tid =
//		blockIdx.x * gridDim.y * blockDim.x * blockDim.y +
//		blockIdx.y * blockDim.x * blockDim.y +
//		threadIdx.x * blockDim.y +
//		threadIdx.y;
//	unsigned int step = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
//
//	for (unsigned int i = tid; i < numRands; i += step)
//	{
//		curandState localState = rngStates[tid];
//		results[i] = curand_uniform(&localState);
//	}
//}


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


// //creates and fills a device pointer with the curandStates
//void Gpu_UniformRandFloats_2d(float **dev_rands, RandData *randData, int numRands)
//{
//	dim3 g, b;
//	GridAndBlocks2d(g, b, randData->length);
//
//	float *dev_r;
//	checkCudaErrors(cudaMalloc((void**)&dev_r, numRands * sizeof(float)));
//
//	rndGenUniform_2d << <g, b >> >(dev_r, randData->dev_curandStates, numRands);
//
//	// Check for any errors launching the kernel
//	getLastCudaError("Gpu_UniformRandFloats execution failed");
//
//	checkCudaErrors(cudaDeviceSynchronize());
//	*dev_rands = dev_r;
//}
//
//void Gpu_NormalRandFloats_2d(float **dev_rands, RandData *randData, int numRands)
//{
//	dim3 g, b;
//	GridAndBlocks2d(g, b, randData->length);
//
//	float *dev_r;
//	checkCudaErrors(cudaMalloc((void**)&dev_r, numRands * sizeof(float)));
//
//	rndGenNormal_2d << <g, b >> >(dev_r, randData->dev_curandStates, numRands);
//
//	// Check for any errors launching the kernel
//	getLastCudaError("Gpu_UniformRandFloats execution failed");
//
//	checkCudaErrors(cudaDeviceSynchronize());
//	*dev_rands = dev_r;
//}