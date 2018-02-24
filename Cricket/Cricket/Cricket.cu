#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "KbLoop.cuh"
#include "Utils.h"
#include "BlockUtils.h"
#include "rando.h"
#include "GridPlot.h"
#include "Cricket.h"
#include "cpu_anim2.h"


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runRandTest(CricketCmdArgs *args);
void runKbLoop(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// MakeControlBlock
////////////////////////////////////////////////////////////////////////////////
void MakeControlBlock(ControlBlock **out, void *appBlock, int2 winSize, int argc, char **argv)
{
	ControlBlock *controlBlock = (ControlBlock *)malloc(sizeof(ControlBlock));
	controlBlock->argc = argc;
	controlBlock->argv = argv;
	controlBlock->winSize = winSize;
	controlBlock->appBlock = appBlock;


	CPUAnimBitmap *cbm = (CPUAnimBitmap *)malloc(sizeof(CPUAnimBitmap));
	cbm->Init(controlBlock);
	cbm->ClearCommand();
	int sz = cbm->image_size();
	checkCudaErrors(cudaMalloc((void**)&controlBlock->device_bitmap, cbm->image_size()));

	*out = controlBlock;
}


////////////////////////////////////////////////////////////////////////////////
// Get cmd line args
////////////////////////////////////////////////////////////////////////////////
void GetCricketCmdArgs(CricketCmdArgs **out, int argc, char **argv)
{
	int plywidth = 512;
	int imageWidth = 64;
	int zoom = 1;
	int threadChop = 16;
	int xStart = 1;
	int yStart = 1;
	int reps = 1;
	int seed = 1234;
	float speed = 0.1;
	float noise = 0.5;

	if (checkCmdLineFlag(argc, (const char **)argv, "seed"))
	{
		seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "plywidth"))
	{
		plywidth = getCmdLineArgumentInt(argc, (const char **)argv, "plywidth");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "imageWidth"))
	{
		imageWidth = getCmdLineArgumentInt(argc, (const char **)argv, "imageWidth");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "speed"))
	{
		speed = getCmdLineArgumentFloat(argc, (const char **)argv, "speed");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "noise"))
	{
		noise = getCmdLineArgumentFloat(argc, (const char **)argv, "noise");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "zoom"))
	{
		zoom = getCmdLineArgumentInt(argc, (const char **)argv, "zoom");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "threadChop"))
	{
		threadChop = getCmdLineArgumentInt(argc, (const char **)argv, "threadChop");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "xStart"))
	{
		xStart = getCmdLineArgumentInt(argc, (const char **)argv, "xStart");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "yStart"))
	{
		yStart = getCmdLineArgumentInt(argc, (const char **)argv, "yStart");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "reps"))
	{
		reps = getCmdLineArgumentInt(argc, (const char **)argv, "reps");
	}
	CricketCmdArgs *cricketCmdArgs = (CricketCmdArgs *)malloc(sizeof(CricketCmdArgs));
	*out = cricketCmdArgs;
	cricketCmdArgs->plywidth = plywidth;
	cricketCmdArgs->imageWidth = imageWidth;
	cricketCmdArgs->zoom = zoom;
	cricketCmdArgs->threadChop = threadChop;
	cricketCmdArgs->xStart = xStart;
	cricketCmdArgs->yStart = yStart;
	cricketCmdArgs->reps = reps;
	cricketCmdArgs->seed = seed;
	cricketCmdArgs->speed = speed;
	cricketCmdArgs->noise = noise;
}

////////////////////////////////////////////////////////////////////////////////
// runKbLoop
////////////////////////////////////////////////////////////////////////////////
//void runKbLoop(CricketCmdArgs *args)
//{
//	ControlBlock *controlBlock;
//	int2 winSize;
//	winSize.x = args->imageWidth;
//	winSize.y = args->imageWidth;
//	MakeControlBlock(&controlBlock, 0, winSize);
//
//	controlBlock->cPUAnimBitmap->anim_and_exit(gridPlot_anim_gpu, gridPlot_anim_exit);
//}
void runKbLoop(int argc, char **argv)
{
	ControlBlock *controlBlock;
	int2 winSize;
	winSize.x = 512;
	winSize.y = 512;
	MakeControlBlock(&controlBlock, 0, winSize, argc, argv);

	controlBlock->cPUAnimBitmap->anim_and_exit(gridPlot_anim_gpu, gridPlot_anim_exit);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	int wonk = IntNamed(argc, argv, "plywidth", 77);
	wonk = IntNamed(argc, argv, "dsa", 77);

	CricketCmdArgs *cricketCmdArgs;
	GetCricketCmdArgs(&cricketCmdArgs, argc, argv);
	//runRandTest(cricketCmdArgs);
	runKbLoop(argc, argv);

}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runRandTest(CricketCmdArgs *args)
{
	RandData *randData;
	InitRandData(&randData, args->seed, args->imageWidth);

	cudaEvent_t     start, stop;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < args->reps; i++) {

		Gpu_UniformRandFloats(&dev_rands, randData, args->imageWidth * args->plywidth);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Uniform:  %3.1f ms\n", elapsedTime);

	DeleteRandData(randData);
}

