#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "PlyBlock.h"
#include "Rando.h"
#include "TexUtils.h"

void MakePlyBlock(PlyBlock **out, unsigned int width, unsigned int seed)
{
	PlyBlock *pbRet = (PlyBlock *)malloc(sizeof(PlyBlock));;
	*out = pbRet;

	RandData *randData = (RandData *)malloc(sizeof(RandData));;

	pbRet->width = width;
	pbRet->area = width * width;
	pbRet->randData = randData;

	unsigned int plyMemSize = pbRet->area * sizeof(float);
	cudaError_t cudaResult = cudaSuccess;

	CHECK(cudaMalloc((void**)&pbRet->dev_inSrc, plyMemSize));
	CHECK(cudaMalloc((void**)&pbRet->dev_outSrc, plyMemSize));
	//CHECK(cudaMalloc((void**)&pbRet->dev_constSrc, plyMemSize));

	pbRet->texIn = TexObjFloat1D(pbRet->dev_inSrc, plyMemSize);
	pbRet->texOut = TexObjFloat1D(pbRet->dev_outSrc, plyMemSize);
	//pbRet->texConst = TexObjFloat1D(pbRet->dev_constSrc, plyMemSize);

	int chop = 64;
	randData->seed = seed;
	randData->blocks = dim3(width/chop, width/chop);
	randData->threads = dim3(chop/4, chop/4);
	CHECK(Gpu_InitRNG_2d(randData));
}