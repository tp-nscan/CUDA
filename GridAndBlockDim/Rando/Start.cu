#include "../../common/common.h"
#include "../../common/book.h"
#include "../../common/DimStuff.h"
#include "../../common/Rando.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>


void RandBench();
void RandCheck();

int main()
{
	RandCheck();
}


void RandBench()
{
	RandData data;
	data.blocks = dim3(128);
	data.threads = dim3(256);
	data.seed = 123;

	cudaEvent_t     start, stop;
	float totalTime = 0;
	float frames = 0;
	int arrayLength = 1000000;
	int reps = 100;

	cudaError_t cudaResult = cudaSuccess;

	float *dev_rands;
	float *host_rands;
	float   elapsedTime;

	cudaResult = Gpu_InitRNG_1d(&data);

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {

		Gpu_UniformRandFloats_1d(&dev_rands, &data, arrayLength);
	}

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	totalTime = elapsedTime;
	printf("Uniform:  %3.1f ms\n", totalTime); // / data->frames);

	 

	data.threads = dim3(16,16);
	cudaResult = Gpu_InitRNG_2d(&data);

	HANDLE_ERROR(cudaEventRecord(start, 0));

	for (int i = 0; i < reps; i++) {

		Gpu_UniformRandFloats_2d(&dev_rands, &data, arrayLength);
	}

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	totalTime = elapsedTime;
	printf("Normal:  %3.1f ms\n", totalTime); // / data->frames);




	//Gpu_GetFloats(&host_rands, dev_rands, data.arrayLength);

	//PrintFloatArray(host_rands, 10, data.arrayLength);

}


void RandCheck()
{
	RandData data;
	data.blocks = dim3(4,4);
	data.threads = dim3(4, 4);
	data.seed = 123;

	int arrayLength = 10000;
	int reps = 100;

	cudaError_t cudaResult = cudaSuccess;

	float *dev_rands;
	float *host_rands;

	cudaResult = Gpu_InitRNG_2d(&data);
	Gpu_NormalRandFloats_2d(&dev_rands, &data, arrayLength);
	Gpu_GetFloats(&host_rands, dev_rands, arrayLength);

	PrintFloatArray(host_rands, 100, arrayLength);

}