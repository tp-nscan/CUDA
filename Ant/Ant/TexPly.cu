#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.h"
#include "BlockUtils.h"
#include "TexPly.h"
#include "Rando.h"
#include "TexUtils.h"

////////////////////////////////////////////////////////////////////////////////
// MakeTexPly
// Make an allocated, uninitialized TexPly
////////////////////////////////////////////////////////////////////////////////
void MakeTexPly(TexPly **out, unsigned int plyLength)
{
	TexPly *pbRet = (TexPly *)malloc(sizeof(TexPly));;
	*out = pbRet;

	pbRet->length = plyLength;
	pbRet->inToOut = true;

	unsigned int plyMemSize = pbRet->length * sizeof(float);
	cudaError_t cudaResult = cudaSuccess;

	checkCudaErrors(cudaMalloc((void**)&pbRet->dev_inSrc, plyMemSize));
	checkCudaErrors(cudaMalloc((void**)&pbRet->dev_outSrc, plyMemSize));

	pbRet->texIn = TexObjFloat1D(pbRet->dev_inSrc, plyMemSize);
	pbRet->texOut = TexObjFloat1D(pbRet->dev_outSrc, plyMemSize);
}

////////////////////////////////////////////////////////////////////////////////
// MakeTexPly
// Make an allocated TexPly, having texIn initialized with *data
////////////////////////////////////////////////////////////////////////////////
void MakeTexPly(TexPly **out, float *data, unsigned int plyLength)
{
	TexPly *pbRet = (TexPly *)malloc(sizeof(TexPly));
	*out = pbRet;

	pbRet->length = plyLength;
	pbRet->inToOut = true;

	unsigned int plyMemSize = pbRet->length * sizeof(float);
	cudaError_t cudaResult = cudaSuccess;

	checkCudaErrors(cudaMalloc((void**)&pbRet->dev_inSrc, plyMemSize));
	checkCudaErrors(cudaMalloc((void**)&pbRet->dev_outSrc, plyMemSize));

	pbRet->texIn = TexObjFloat1D(pbRet->dev_inSrc, plyMemSize);
	pbRet->texOut = TexObjFloat1D(pbRet->dev_outSrc, plyMemSize);

	checkCudaErrors(cudaMemcpy(pbRet->dev_inSrc, data, plyMemSize, cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
// GetTexPlyData
// Pass the latest data update back to the host
////////////////////////////////////////////////////////////////////////////////
void GetTexPlyData(TexPly *texPly, float *host_res)
{
	unsigned int plyMemSize = texPly->length * sizeof(float);
	if (texPly->inToOut)
	{
		checkCudaErrors(cudaMemcpy(host_res, texPly->dev_inSrc, plyMemSize, cudaMemcpyDeviceToHost));
	}
	else
	{
		checkCudaErrors(cudaMemcpy(host_res, texPly->dev_outSrc, plyMemSize, cudaMemcpyDeviceToHost));
	}
}

////////////////////////////////////////////////////////////////////////////////
// RunTexPly
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
void RunTexPly(TexPly *texPly, int2 plySize, float speed, int blocks, int threads, float decay)
{
	if (texPly->inToOut)
	{
		tex_1d_2d_1d <<<blocks, threads>>>(texPly->dev_outSrc, *(texPly->texIn), plySize, speed, decay);
		texPly->inToOut = false;
	}
	else
	{
		tex_1d_2d_1d <<<blocks, threads>>>(texPly->dev_inSrc, *(texPly->texOut), plySize, speed, decay);
		texPly->inToOut = true;
	}
	getLastCudaError("RunTexPly execution failed");
}

////////////////////////////////////////////////////////////////////////////////
// FreeTexPly
// Deallocate all device memory.
////////////////////////////////////////////////////////////////////////////////
void FreeTexPly(TexPly *texPly)
{
	cudaDestroyTextureObject(*(texPly->texIn));
	cudaDestroyTextureObject(*texPly->texOut);
	 
	checkCudaErrors(cudaFree(texPly->dev_inSrc));
	checkCudaErrors(cudaFree(texPly->dev_outSrc));

}


////////////////////////////////////////////////////////////////////////////////
// RunTexPlyCorrectness
// Print a few updates to the console.
////////////////////////////////////////////////////////////////////////////////
void RunTexPlyCorrectness(int argc, char **argv)
{
	//texStride=12 blocks=6 threads=6 reps=20 speed=0.3 decay=3.5 offset=40
	int texStride = IntNamed(argc, argv, "texStride", 12);
	int texArea = texStride*texStride;
	int blocks = IntNamed(argc, argv, "blocks", 6);
	int threads = IntNamed(argc, argv, "threads", 6);
	int2 plySize;  plySize.x = texStride; plySize.y = texStride;
	int reps = IntNamed(argc, argv, "reps", 16);
	float speed = FloatNamed(argc, argv, "speed", 0.3);
	float decay = FloatNamed(argc, argv, "decay", 3.3);
	int offset = IntNamed(argc, argv, "offset", 40);

	float *da = DotArray(texStride, offset);
	PrintFloatArray(da, texStride, texArea);

	TexPly *texPly;
	MakeTexPly(&texPly, da, texArea);
	float *daOut = (float *)malloc(sizeof(float)*texArea);

	for (int i = 0; i < reps; i++)
	{
		printf("The other way:\n");
		RunTexPly(texPly, plySize, speed, blocks, threads, decay);
		GetTexPlyData(texPly, daOut);
		PrintFloatArray(daOut, texStride, texArea);
	}
}


////////////////////////////////////////////////////////////////////////////////
// RunTexPlyBench
////////////////////////////////////////////////////////////////////////////////
void RunTexPlyBench(int argc, char **argv)
{
	//texStride=1024 reps=1000 speed=0.0003 decay=3.5
	int texStride = IntNamed(argc, argv, "texStride", 12);
	int texArea = texStride*texStride;
	int2 plySize;  plySize.x = texStride; plySize.y = texStride;
	int reps = IntNamed(argc, argv, "reps", 16);
	float speed = FloatNamed(argc, argv, "speed", 0.3);
	float decay = FloatNamed(argc, argv, "decay", 3.3);
	int offset = IntNamed(argc, argv, "offset", 40);


	float *da = DotArray(texStride, offset);
	TexPly *texPly;
	MakeTexPly(&texPly, da, texArea);
	float *daOut = (float *)malloc(sizeof(float)*texArea);


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
				RunTexPly(texPly, plySize, speed, b, t, decay);
			}

			checkCudaErrors(cudaEventRecord(stop, 0));
			checkCudaErrors(cudaEventSynchronize(stop));
			checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

			printf("Blocks: %d   Threads: %d   Time:  %3.1f ms\n", b, t, elapsedTime);
		}
	}
	FreeTexPly(texPly);
}