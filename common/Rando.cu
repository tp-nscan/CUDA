#include "common.h"
#include "DimStuff.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <curand_kernel.h>
#include "Rando.h"

__global__ void initRNG(curandState *const rngStates, const unsigned int seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Initialise the RNG
	curand_init(seed, tid, 0, &rngStates[tid]);
}

// Estimator kernel
//template <typename Real>
__global__ void gen_uniform(float *const results, curandState *const rngStates,
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

__global__ void gen_normal(float *const results, curandState *const rngStates,
	const unsigned int numRands)
{
	//// Determine thread ID
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

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// creates and fills a device pointer with the curandStates
cudaError_t Gpu_InitRNG(curandState **dev_rngs, RandData *randData)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	curandState *dev_rngens;
	CHECK_G(cudaMalloc((void**)&dev_rngens, dim3Vol(&randData->blocks) * dim3Vol(&randData->threads) * sizeof(curandState)));
	randData->dev_rngs = dev_rngens;

	initRNG << <randData->blocks, randData->threads >> >(randData->dev_rngs, randData->seed);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	CHECK_G(cudaDeviceSynchronize());

	return cudaStatus;
Error:
	cudaFree(randData->dev_rngs);
	return cudaStatus;
}

// creates and fills a device pointer with the curandStates
cudaError_t Gpu_UniformRandFloats(float **dev_rands, RandData *randData, int numRands)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	float *dev_r;
	cudaStatus = cudaMalloc((void**)&dev_r, numRands * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	randData->dev_rngs;

	gen_uniform << <randData->blocks, randData->threads >> >(dev_r, randData->dev_rngs, numRands);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	CHECK_G(cudaDeviceSynchronize());
	*dev_rands = dev_r;

Error:
	return cudaStatus;
}


// creates and fills a device pointer with the curandStates
cudaError_t Gpu_NormalRandFloats(float **dev_rands, RandData *randData,	int numRands)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	float *dev_r;
	cudaStatus = cudaMalloc((void**)&dev_r, numRands * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	randData->dev_rngs;

	gen_normal << <randData->blocks, randData->threads >> >(dev_r, randData->dev_rngs, numRands);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	CHECK_G(cudaDeviceSynchronize());
	*dev_rands = dev_r;

Error:
	return cudaStatus;
}