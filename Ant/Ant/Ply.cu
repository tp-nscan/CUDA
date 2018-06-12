#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include "Utils.h"
#include "BlockUtils.h"
#include "Ply.h"


////////////////////////////////////////////////////////////////////////////////
// grid_local_energy
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void grid_local_energy(float *resOut, float *plyIn, int2 plySize) {

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

		float t = plyIn[top];
		float l = plyIn[left];
		float c = plyIn[i];
		float r = plyIn[right];
		float b = plyIn[bottom];

		resOut[i] = c * (t + b + r + l);
	}
}

////////////////////////////////////////////////////////////////////////////////
// grid_local_energyC
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void grid_local_energyC(float *resOut, float *plyIn, int2 plySize) {

	__shared__ float cache[THREADS_1D];
	int cacheIndex = threadIdx.x;

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

		float t = plyIn[top];
		float l = plyIn[left];
		float c = plyIn[i];
		float r = plyIn[right];
		float b = plyIn[bottom];

		resOut[i] = c * (t + b + r + l);
	}

	__syncthreads();

	int j = blockDim.x / 2;
	while (j != 0) {
		if (cacheIndex < j)
			cache[cacheIndex] += cache[cacheIndex + j];
		__syncthreads();
		j /= 2;
	}
	if (cacheIndex == 0)
		resOut[blockIdx.x] = cache[0];
}



////////////////////////////////////////////////////////////////////////////////
// ply_1d_2d_1d
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void ply_1d_2d_1d(float *resOut, float *arrayIn, int2 plySize,
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
// MakePly
// Make an allocated, uninitialized Ply
////////////////////////////////////////////////////////////////////////////////
void MakePly(Ply **out, unsigned int plyLength, unsigned int span)
{
	Ply *pbRet = (Ply *)malloc(sizeof(Ply));;
	*out = pbRet;

	pbRet->area = plyLength;
	pbRet->span = span;
	pbRet->inToOut = true;

	unsigned int plyMemSize = pbRet->area * sizeof(float);
	cudaError_t cudaResult = cudaSuccess;

	checkCudaErrors(cudaMalloc((void**)&pbRet->dev_inSrc, plyMemSize));
	checkCudaErrors(cudaMalloc((void**)&pbRet->dev_outSrc, plyMemSize));

}

////////////////////////////////////////////////////////////////////////////////
// MakePly
// Make an allocated Ply, having In initialized with *data
////////////////////////////////////////////////////////////////////////////////
void MakePly(Ply **out, float *data, unsigned int plyLength, unsigned int span)
{
	Ply *pbRet = (Ply *)malloc(sizeof(Ply));
	*out = pbRet;

	pbRet->area = plyLength;
	pbRet->span = span;
	pbRet->inToOut = true;

	unsigned int plyMemSize = pbRet->area * sizeof(float);
	cudaError_t cudaResult = cudaSuccess;

	checkCudaErrors(cudaMalloc((void**)&pbRet->dev_inSrc, plyMemSize));
	checkCudaErrors(cudaMalloc((void**)&pbRet->dev_outSrc, plyMemSize));

	checkCudaErrors(cudaMemcpy(pbRet->dev_inSrc, data, plyMemSize, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
// GetPlyData
// Pass the latest data update back to the host
////////////////////////////////////////////////////////////////////////////////
void GetPlyData(Ply *ply, float *host_res)
{
	unsigned int plyMemSize = ply->area * sizeof(float);
	if (ply->inToOut)
	{
		checkCudaErrors(cudaMemcpy(host_res, ply->dev_inSrc, plyMemSize, cudaMemcpyDeviceToHost));
	}
	else
	{
		checkCudaErrors(cudaMemcpy(host_res, ply->dev_outSrc, plyMemSize, cudaMemcpyDeviceToHost));
	}
}

////////////////////////////////////////////////////////////////////////////////
// RunPly
// A 1d ture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
void RunPly(Ply *ply, int2 plySize, float speed, int blocks, int threads, float decay)
{
	if (ply->inToOut)
	{
	    ply_1d_2d_1d << <blocks, threads >> >(ply->dev_outSrc, ply->dev_inSrc, plySize, speed, decay);
		ply->inToOut = false;
	}
	else
	{
		ply_1d_2d_1d << <blocks, threads >> >(ply->dev_inSrc, ply->dev_outSrc, plySize, speed, decay);
		ply->inToOut = true;
	}
	getLastCudaError("RunPly execution failed");
}

////////////////////////////////////////////////////////////////////////////////
// FreePly
// Deallocate all device memory.
////////////////////////////////////////////////////////////////////////////////
void FreePly(Ply *ply)
{
	checkCudaErrors(cudaFree(ply->dev_inSrc));
	checkCudaErrors(cudaFree(ply->dev_outSrc));
}


////////////////////////////////////////////////////////////////////////////////
// RunPlyCorrectness
// Print a few updates to the console.
////////////////////////////////////////////////////////////////////////////////
void RunPlyCorrectness(int argc, char **argv)
{
	//Stride=12 blocks=6 threads=6 reps=20 speed=0.3 decay=3.5 offset=40
	int stride = IntNamed(argc, argv, "stride", 12);
	int area = stride*stride;
	int blocks = IntNamed(argc, argv, "blocks", 6);
	int threads = IntNamed(argc, argv, "threads", 6);
	int2 plySize;  plySize.x = stride; plySize.y = stride;
	int reps = IntNamed(argc, argv, "reps", 16);
	float speed = FloatNamed(argc, argv, "speed", 0.3);
	float decay = FloatNamed(argc, argv, "decay", 3.3);
	int offset = IntNamed(argc, argv, "offset", 40);

	float *da = DotArray(stride, offset);
	PrintFloatArray(da, stride, area);

	Ply *ply;
	MakePly(&ply, da, area, stride);
	float *daOut = (float *)malloc(sizeof(float)*area);

	for (int i = 0; i < reps; i++)
	{
		printf("The other way:\n");
		RunPly(ply, plySize, speed, blocks, threads, decay);
		GetPlyData(ply, daOut);
		PrintFloatArray(daOut, stride, area);
	}
}


////////////////////////////////////////////////////////////////////////////////
// RunPlyBench
////////////////////////////////////////////////////////////////////////////////
void RunPlyBench(int argc, char **argv)
{
	//stride=1024 reps=1000 speed=0.0003 decay=3.5
	int stride = IntNamed(argc, argv, "stride", 12);
	int area = stride*stride;
	int2 plySize;  plySize.x = stride; plySize.y = stride;
	int reps = IntNamed(argc, argv, "reps", 16);
	float speed = FloatNamed(argc, argv, "speed", 0.3);
	float decay = FloatNamed(argc, argv, "decay", 3.3);
	int offset = IntNamed(argc, argv, "offset", 40);


	float *da = DotArray(stride, offset);
	Ply *ply;
	MakePly(&ply, da, area, stride);
	float *daOut = (float *)malloc(sizeof(float)*area);


	cudaEvent_t start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;
	for (int b = 64; b <= 512; b += 64)
	{
		for (int t = 128; t <= 1024; t += 128)
		{
			checkCudaErrors(cudaEventCreate(&start));
			checkCudaErrors(cudaEventCreate(&stop));

			checkCudaErrors(cudaEventRecord(start, 0));

			for (int i = 0; i < reps; i++) {
				RunPly(ply, plySize, speed, b, t, decay);
			}

			checkCudaErrors(cudaEventRecord(stop, 0));
			checkCudaErrors(cudaEventSynchronize(stop));
			checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

			printf("Blocks: %d   Threads: %d   Time:  %3.1f ms\n", b, t, elapsedTime);
		}
	}
	FreePly(ply);
}