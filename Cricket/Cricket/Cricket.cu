#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include "BlockUtils.h";
#include "rando.h";


struct CricketCmdArgs {
	int plywidth;
	int imageWidth;
	int zoom;
	int threadChop;
	int xStart;
	int yStart;
	int reps;
	int seed;
	float speed;
	float noise;
};

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runRandTest(CricketCmdArgs *args);




////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(float *g_idata, float *g_odata)
{
	// shared memory
	// the size is determined by the host application
	extern  __shared__  float sdata[];

	// access thread id
	const unsigned int tid = threadIdx.x;
	// access number of threads in this block
	const unsigned int num_threads = blockDim.x;

	// read in input data from global memory
	sdata[tid] = g_idata[tid];
	__syncthreads();

	// perform some computations
	sdata[tid] = (float)num_threads * sdata[tid];
	__syncthreads();

	// write data to global memory
	g_odata[tid] = sdata[tid];
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
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	CricketCmdArgs *cricketCmdArgs;
	GetCricketCmdArgs(&cricketCmdArgs, argc, argv);
	runRandTest(cricketCmdArgs);
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runRandTest(CricketCmdArgs *args)
{
	RandData *randData;
	InitRandData(&randData, args->seed, args->imageWidth);

	cudaEvent_t     start, stop;
	float totalTime = 0;
	float frames = 0;
	int arrayLength = 1000000;
	int reps = 1;

	cudaError_t cudaResult = cudaSuccess;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, 0));

	for (int i = 0; i < args->reps; i++) {

		Gpu_UniformRandFloats_2d(&dev_rands, randData, args->imageWidth * args->plywidth);
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	totalTime = elapsedTime;
	printf("Uniform:  %3.1f ms\n", totalTime);

	DeleteRandData(randData);
}

