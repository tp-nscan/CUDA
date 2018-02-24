#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GridPlot.h"


void MakeGridPlotAppBlock(GridPlotAppBlock **out, GridPlot *gridPlot, float *host_data)
{
	GridPlotAppBlock *gbaRet = (GridPlotAppBlock *)malloc(sizeof(GridPlotAppBlock));;
	gbaRet->gridPlot = gridPlot;

	gbaRet->totalTime = 0;
	gbaRet->host_Data = host_data;

	checkCudaErrors(cudaMalloc((void**)&gbaRet->dev_Data, gridPlot->dataSize.x * gridPlot->dataSize.y * sizeof(float)));

	*out = gbaRet;
}

void MakeGridPlot(GridPlot **out, int zoom, int2 dataSize, int2 dataUpLeft)
{
	GridPlot *gridPlot = (GridPlot *)malloc(sizeof(GridPlot));
	gridPlot->zoom = zoom;
	gridPlot->dataSize = dataSize;
	gridPlot->dataUpleft = dataUpLeft;
	*out = gridPlot;
}


__device__ unsigned char colorValue(float n1, float n2, int hue) {
	if (hue > 360)      hue -= 360;
	else if (hue < 0)   hue += 360;

	if (hue < 60)
		return (unsigned char)(255 * (n1 + (n2 - n1)*hue / 60));
	if (hue < 180)
		return (unsigned char)(255 * n2);
	if (hue < 240)
		return (unsigned char)(255 * (n1 + (n2 - n1)*(240 - hue) / 60));
	return (unsigned char)(255 * n1);
}

//__global__ void float_to_color2(unsigned char *optr, const float *outSrc, int zoom, int plyWidth) {
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int plyOffset = x + y * plyWidth;
//	//int frameWidth = blockDim.x * gridDim.x;
//
//	float l = outSrc[plyOffset];
//	float s = 1;
//	int h = (180 + (int)(360.0f * outSrc[plyOffset])) % 360;
//	float m1, m2;
//
//	if (l <= 0.5f)
//		m2 = l * (1 + s);
//	else
//		m2 = l + s - l * s;
//	m1 = 2 * l - m2;
//
//	int imgOffset = (x + y * plyWidth * zoom) * zoom;
//	for (int i = 0; i < zoom; i++)
//	{
//		for (int j = 0; j < zoom; j++)
//		{
//			int noff = (imgOffset + i * plyWidth * zoom + j) * 4;
//			optr[noff + 0] = colorValue(m1, m2, h + 120);
//			optr[noff + 1] = colorValue(m1, m2, h);
//			optr[noff + 2] = colorValue(m1, m2, h - 120);
//			optr[noff + 3] = 255;
//		}
//		__syncthreads();
//	}
//}

__global__ void float_to_color2(unsigned char *optr, const float *outSrc, int zoom, int2 plySize, int2 imgSize) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	int plyOffset = x + y * plySize.x;
	float l = outSrc[plyOffset];


	int imgOffset = (x + y * imgSize.x) * zoom;
	for (int i = 0; i < zoom; i++)
	{
		for (int j = 0; j < zoom; j++)
		{
			int noff = (imgOffset + i * imgSize.x + j) * 4;
			//optr[noff + 0] = (x * y) % 256;
			//optr[noff + 1] = (y) % 256;
			//optr[noff + 2] = (x) % 256;
			//optr[noff + 3] = 255;
			optr[noff + 0] = (x * y) % 256;
			optr[noff + 1] = 255;
			optr[noff + 2] = 255;
			optr[noff + 3] = 255;
			
		}
		__syncthreads();
	}
}

void L(ControlBlock *controlBlock, int ticks)
{
	GridPlot *gridPlot;
	int2 dataSize, dataUpLeft;
	float host_data[4] = { -0.99, -0.5, 0.5, 0.99 };

	dataSize.x = 2;
	dataSize.y = 2;
	dataUpLeft.x = 0;
	dataUpLeft.y = 0;
	MakeGridPlot(&gridPlot, 8, dataSize, dataUpLeft);

	GridPlotAppBlock *appBlock;
	MakeGridPlotAppBlock(&appBlock, gridPlot, host_data);
	controlBlock->appBlock = appBlock;

	unsigned int dataMem = dataSize.x * dataSize.y * sizeof(float);
	appBlock->host_Data = host_data;
	checkCudaErrors(cudaMemcpy(appBlock->dev_Data, appBlock->host_Data, dataMem, cudaMemcpyHostToDevice));

	dim3    drawBlocks(8, 8);
	dim3    drawThreads(8, 8);

	float_to_color2<<<drawBlocks, drawThreads >>>(controlBlock->device_bitmap, appBlock->dev_Data,
		                      appBlock->gridPlot->zoom, controlBlock->winSize, controlBlock->winSize);

	checkCudaErrors(cudaDeviceSynchronize());
	int sz = controlBlock->cPUAnimBitmap->image_size();

	checkCudaErrors(cudaMemcpy(controlBlock->cPUAnimBitmap->get_ptr(),
							   controlBlock->device_bitmap, 
							   controlBlock->cPUAnimBitmap->image_size(), cudaMemcpyDeviceToHost));

}

void gridPlot_anim_gpu(ControlBlock *controlBlock, int ticks) {
	char *comm = controlBlock->command;
	GridPlotAppBlock *appBlock = (GridPlotAppBlock *)controlBlock->appBlock;

	if (strcmp(comm, "l") == 0)
	{
		L(controlBlock, ticks);

		controlBlock->cPUAnimBitmap->ClearCommand();
		return;
	}
	if (strcmp(comm, "g") == 0)
	{
		controlBlock->cPUAnimBitmap->ClearCommand();
		//G(controlBlock, ticks);
		return;
	}
	if (strlen(comm) > 0)
	{
		printf("unknown command: %s\n", comm);
		controlBlock->cPUAnimBitmap->ClearCommand();
		return;
	}
}


void gridPlot_anim_exit(ControlBlock *controlBlock) {
	//AppBlock *appBlock = (AppBlock *)controlBlock->appBlock;

	//FreePlyBlock(appBlock->plyBlock);

	//HANDLE_ERROR(cudaEventDestroy(appBlock->start));
	//HANDLE_ERROR(cudaEventDestroy(appBlock->stop));
}
