#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "PlyBlock.h"
#include "Rando.h"
#include "TexUtils.h"

int ThreadChop2d(int width)
{
	if (width > 255)
		return 16;
	if (width > 63)
		return 8;
	if (width > 15)
		return 4;
	if (width > 3)
		return 2;
	return 1;
}

void MakePlyBlock(PlyBlock **out, unsigned int plyWidth, unsigned int seed)
{
	PlyBlock *pbRet = (PlyBlock *)malloc(sizeof(PlyBlock));;
	*out = pbRet;

	RandData *randData = (RandData *)malloc(sizeof(RandData));;

	pbRet->width = plyWidth;
	pbRet->area = plyWidth * plyWidth;
	pbRet->randData = randData;

	unsigned int plyMemSize = pbRet->area * sizeof(float);
	cudaError_t cudaResult = cudaSuccess;

	CHECK(cudaMalloc((void**)&pbRet->dev_inSrc, plyMemSize));
	CHECK(cudaMalloc((void**)&pbRet->dev_outSrc, plyMemSize));

	pbRet->texIn = TexObjFloat1D(pbRet->dev_inSrc, plyMemSize);
	pbRet->texOut = TexObjFloat1D(pbRet->dev_outSrc, plyMemSize);

	int chop = ThreadChop2d(pbRet->area);
	randData->blocks = dim3(plyWidth /chop, plyWidth /chop);
	randData->threads = dim3(chop, chop);

	randData->seed = seed;
	CHECK(Gpu_InitRNG_2d(randData));
}

void FreePlyBlock(PlyBlock *plyBlock)
{
	cudaDestroyTextureObject(*(plyBlock->texIn));
	cudaDestroyTextureObject(*plyBlock->texOut);

	CHECK(cudaFree(plyBlock->dev_inSrc));
	CHECK(cudaFree(plyBlock->dev_outSrc));
	CHECK(cudaFree(plyBlock->randData->dev_curandStates));

}