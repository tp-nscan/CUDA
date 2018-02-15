#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include "common.h"
#include "Utils.h"
#include "BlockUtils.h"
#include "Rando.h"

__global__ void initRNG_1d(curandState *const rngStates, const unsigned int seed)
{
	// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Initialise the RNG
	curand_init(seed, tid, 0, &rngStates[tid]);
}


__global__ void initRNG_2d(curandState *const rngStates, const unsigned int seed)
{
	unsigned int tid =
		blockIdx.x * gridDim.y * blockDim.x * blockDim.y +
		blockIdx.y * blockDim.x * blockDim.y +
		threadIdx.x * blockDim.y +
		threadIdx.y;
	// Initialise the RNG
	curand_init(seed, tid, 0, &rngStates[tid]);
}

// Estimator kernel
//template <typename Real>
__global__ void gen_uniform_1d(float *const results, curandState *const rngStates,
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

__global__ void gen_uniform_2d(float *const results, curandState *const rngStates,
	const unsigned int numRands)
{
    unsigned int tid = 
		  blockIdx.x * gridDim.y * blockDim.x * blockDim.y +
		  blockIdx.y * blockDim.x * blockDim.y +
		  threadIdx.x * blockDim.y +
		  threadIdx.y;
	unsigned int step = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

	for (unsigned int i = tid; i < numRands; i += step)
	{
		curandState localState = rngStates[tid];
		results[i] = curand_uniform(&localState);
	}
}

__global__ void gen_normal_1d(float *const results, curandState *const rngStates,
	const unsigned int numRands)
{
	//// Determine thread ID
	unsigned int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
	unsigned int step = gridDim.x * blockDim.x;

	for (unsigned int i = tid; i < numRands; i += step)
	{
		curandState localState = rngStates[tid];
		results[i] = curand_normal(&localState);
	}
}

__global__ void gen_normal_2d(float *const results, curandState *const rngStates,
	const unsigned int numRands)
{

	int tid = gridDim.x * blockDim.x * blockDim.y * blockIdx.y +
		blockDim.y * blockDim.x * blockIdx.x +
		blockDim.x * threadIdx.y + threadIdx.x;

	unsigned int step = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

	for (unsigned int i = tid; i < numRands; i += step)
	{
		curandState localState = rngStates[tid];
		results[i] = curand_normal(&localState);
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
cudaError_t Gpu_InitRNG_1d(RandData *randData)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	curandState *dev_rngens;
	CHECK_G(cudaMalloc((void**)&dev_rngens, dim3Vol(&randData->blocks) * dim3Vol(&randData->threads) * sizeof(curandState)));
	randData->dev_curandStates = dev_rngens;

	initRNG_1d << <randData->blocks, randData->threads >> >(randData->dev_curandStates, randData->seed);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	CHECK_G(cudaDeviceSynchronize());

	return cudaStatus;
Error:
	cudaFree(randData->dev_curandStates);
	return cudaStatus;
}

// creates and fills a device pointer with the curandStates
cudaError_t Gpu_InitRNG_2d(RandData *randData)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	curandState *dev_rngens;
	CHECK_G(cudaMalloc((void**)&dev_rngens, dim3Vol(&randData->blocks) * dim3Vol(&randData->threads) * sizeof(curandState)));
	randData->dev_curandStates = dev_rngens;

	initRNG_2d << <randData->blocks, randData->threads >> >(randData->dev_curandStates, randData->seed);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	CHECK_G(cudaDeviceSynchronize());

	return cudaStatus;
Error:
	cudaFree(randData->dev_curandStates);
	return cudaStatus;
}


// creates and fills a device pointer with the curandStates
cudaError_t Gpu_UniformRandFloats_1d(float **dev_rands, RandData *randData, int numRands)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	float *dev_r;
	cudaStatus = cudaMalloc((void**)&dev_r, numRands * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	randData->dev_curandStates;

	gen_uniform_1d << <randData->blocks, randData->threads >> >(dev_r, randData->dev_curandStates, numRands);

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


 //creates and fills a device pointer with the curandStates
cudaError_t Gpu_UniformRandFloats_2d(float **dev_rands, RandData *randData, int numRands)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	float *dev_r;
	cudaStatus = cudaMalloc((void**)&dev_r, numRands * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	randData->dev_curandStates;

	gen_uniform_2d << <randData->blocks, randData->threads >> >(dev_r, randData->dev_curandStates, numRands);

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
cudaError_t Gpu_NormalRandFloats_1d(float **dev_rands, RandData *randData,	int numRands)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	float *dev_r;
	cudaStatus = cudaMalloc((void**)&dev_r, numRands * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	randData->dev_curandStates;

	gen_normal_1d << <randData->blocks, randData->threads >> >(dev_r, randData->dev_curandStates, numRands);

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

//creates and fills a device pointer with the curandStates
cudaError_t Gpu_NormalRandFloats_2d(float **dev_rands, RandData *randData, int numRands)
{
	cudaError_t cudaStatus = cudaSuccess;
	CHECK_G(cudaSetDevice(0));

	float *dev_r;
	cudaStatus = cudaMalloc((void**)&dev_r, numRands * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	randData->dev_curandStates;

	gen_normal_2d << <randData->blocks, randData->threads >> >(dev_r, randData->dev_curandStates, numRands);

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