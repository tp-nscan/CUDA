#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GridPlot.h"
#include "Utils.h"


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

//
__global__ void float_to_color(unsigned char *optr, const float *outSrc, int zoom, 
							   int2 plySize, int2 imgSize, int up, int left) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int xp = x + up;
	int yp = y + left;
	if ((xp >= plySize.x) || (xp < 0) || (yp >= plySize.y) || (yp < 0)) return;

	int plyOffset = yp + xp * plySize.y;
	float l = outSrc[plyOffset];

	int imgOffset = (y + (plySize.x - x - 1) * imgSize.y ) * zoom;

	if ((threadIdx.x == 1) && (threadIdx.y == 2))
	{
		int k = l * 256;
	}

	for (int i = 0; i < zoom; i++)
	{
		for (int j = 0; j < zoom; j++)
		{
			int noff = (imgOffset + i * imgSize.y + j) * 4;
			optr[noff + 0] = (unsigned char)(l * 256) % 256;
			optr[noff + 1] = (unsigned char)(l * 256) % 256;
			optr[noff + 2] = (unsigned char)(l * 256) % 256;
			optr[noff + 3] = 255;

		}
	}
	//__syncthreads();
}

__global__ void float_to_colorW(unsigned char *optr, const float *outSrc, int zoom,
	int2 plySize, int2 imgSize, int up, int left) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int xp = x + up;
	int yp = y + left;
	if ((xp >= plySize.x) || (xp < 0) || (yp >= plySize.y) || (yp < 0)) return;

	int plyOffset = yp + xp * plySize.y;
	float l = outSrc[plyOffset];

	int imageMem = (imgSize.x * imgSize.y - 1) * sizeof(float);
	int tDisp = ((imgSize.y - 1 - y + ((x - 1) * imgSize.y)) * zoom ) * sizeof(float);

	for (int i = 0; i < zoom; i++)
	{
		for (int j = 0; j < zoom; j++)
		{
			int zDisp = (i * imgSize.y + j) * sizeof(float);
			int noff = imageMem - tDisp - zDisp;
			if((noff > -1) && (noff < imageMem -1))
			{
				optr[noff + 0] = (unsigned char)(l * 256) % 256;
				optr[noff + 1] = (unsigned char)(l * 256) % 256;
				optr[noff + 2] = 200; // (unsigned char)(l * 256) % 256;
				optr[noff + 3] = 255;
			}
			else
			{
				int k2 = l * 256;
			}
		}
	}
	//__syncthreads();
}



__global__ void float_to_colorT(unsigned char *optr, const float *outSrc, int zoom,
	int2 plySize, int2 imgSize, int up, int left) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int xp = x + up;
	int yp = y + left;
	if ((xp >= plySize.x) || (xp < 0) || (yp >= plySize.y) || (yp < 0)) return;

	int plyOffset = yp + xp * plySize.y;
	float l = outSrc[plyOffset];

	if (l < 0.5)
	{
		int k3 = l * 256;
	}

	int imageMem = (imgSize.x * imgSize.y - 1) * sizeof(float);
	int tDisp = ((imgSize.y - 1 - y + ((x - 1) * imgSize.y)) * zoom) * sizeof(float);

	if ((threadIdx.x == 1) && (threadIdx.y == 2))
	{
		int k = l * 256;
	}

	for (int i = 0; i < zoom; i++)
	{
		for (int j = 0; j < zoom; j++)
		{
			int zDisp = (i * imgSize.y + j) * sizeof(float);
			int noff = imageMem - tDisp - zDisp;
			if ((noff > -1) && (noff < imageMem - 1))
			{
				if (l < 0.1)
				{
					int k4 = l * 256;
				}
				optr[noff + 0] = (unsigned char)(l * 256) % 256;
				optr[noff + 1] = (unsigned char)(l * 256) % 256;
				optr[noff + 2] = 200; // (unsigned char)(l * 256) % 256;
				optr[noff + 3] = 255;
			}
			else
			{
				int k2 = l * 256;
			}
		}
	}
	//__syncthreads();
}


__global__ void float_to_color2(unsigned char *optr, const float *outSrc, int zoom, 
								int2 plySize, int2 imgSize, int up, int left) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int xp = x + up;
	int yp = y + left;
	if ((xp >= plySize.x) || (xp < 0) || (yp >= plySize.y) || (yp < 0)) return;

	int plyOffset = yp + xp * plySize.y;
	float l = outSrc[plyOffset];

	int tileOffset = (x + y * imgSize.x) * zoom;
	int imageArea = imgSize.x * imgSize.y;
	int tileArea = blockDim.x * blockDim.y * gridDim.x * gridDim.y * zoom * zoom;

	for (int imgOffset = tileOffset; imgOffset < imageArea; imgOffset += tileArea)
	{
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
		}
	}
	__syncthreads();
}

void L(ControlBlock *controlBlock, int ticks)
{
	GridPlot *gridPlot;
	int2 dataSize, dataUpLeft;
	int plywidth = IntNamed(controlBlock->argc, controlBlock->argv, "plywidth", 64);
	int xStart = IntNamed(controlBlock->argc, controlBlock->argv, "xStart", 0);
	int yStart = IntNamed(controlBlock->argc, controlBlock->argv, "yStart", 0);
	dataSize.x = plywidth;
	dataSize.y = plywidth;
	dataUpLeft.x = xStart;
	dataUpLeft.y = yStart;

	float *host_data = BackSlashArray(plywidth, 0.5, 1);

	int zoom = IntNamed(controlBlock->argc, controlBlock->argv, "zoom", 2);
	MakeGridPlot(&gridPlot, zoom, dataSize, dataUpLeft);

	GridPlotAppBlock *appBlock;
	MakeGridPlotAppBlock(&appBlock, gridPlot, host_data);
	controlBlock->appBlock = appBlock;

	unsigned int dataMem = dataSize.x * dataSize.y * sizeof(float);
	unsigned int winMem = controlBlock->winSize.x * controlBlock->winSize.y * sizeof(float);
	appBlock->host_Data = host_data;
	checkCudaErrors(cudaMemcpy(appBlock->dev_Data, appBlock->host_Data, dataMem, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(controlBlock->device_bitmap, 0, winMem));

	int blockSize = IntNamed(controlBlock->argc, controlBlock->argv, "blockSize", 4);
	int gridSize = IntNamed(controlBlock->argc, controlBlock->argv, "gridSize", 4);
	dim3    drawBlocks(gridSize, gridSize);
	dim3    drawThreads(blockSize, blockSize);

	float_to_color2<<<drawBlocks, drawThreads >>>(controlBlock->device_bitmap, appBlock->dev_Data,
		                      appBlock->gridPlot->zoom, appBlock->gridPlot->dataSize, 
							  controlBlock->winSize, appBlock->gridPlot->dataUpleft.x,
							  appBlock->gridPlot->dataUpleft.y);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(controlBlock->cPUAnimBitmap->get_ptr(),
							   controlBlock->device_bitmap, 
							   controlBlock->cPUAnimBitmap->image_size(), cudaMemcpyDeviceToHost));

}

void W(ControlBlock *controlBlock, int ticks)
{
	GridPlot *gridPlot;
	int2 dataSize, dataUpLeft;
	int plywidth = IntNamed(controlBlock->argc, controlBlock->argv, "plywidth", 64);
	int xStart = IntNamed(controlBlock->argc, controlBlock->argv, "xStart", 0);
	int yStart = IntNamed(controlBlock->argc, controlBlock->argv, "yStart", 0);
	dataSize.x = plywidth;
	dataSize.y = plywidth;
	dataUpLeft.x = xStart;
	dataUpLeft.y = yStart;

	float *host_data = BackSlashArray(plywidth, 0.5, 1);

	int zoom = IntNamed(controlBlock->argc, controlBlock->argv, "zoom", 2);
	MakeGridPlot(&gridPlot, zoom, dataSize, dataUpLeft);

	GridPlotAppBlock *appBlock;
	MakeGridPlotAppBlock(&appBlock, gridPlot, host_data);
	controlBlock->appBlock = appBlock;

	unsigned int dataMem = dataSize.x * dataSize.y * sizeof(float);
	unsigned int winMem = controlBlock->winSize.x * controlBlock->winSize.y * sizeof(float);
	appBlock->host_Data = host_data;
	checkCudaErrors(cudaMemcpy(appBlock->dev_Data, appBlock->host_Data, dataMem, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(controlBlock->device_bitmap, 0, winMem));

	int blockSize = IntNamed(controlBlock->argc, controlBlock->argv, "blockSize", 4);
	int gridSize = IntNamed(controlBlock->argc, controlBlock->argv, "gridSize", 4);
	dim3    drawBlocks(gridSize, gridSize);
	dim3    drawThreads(blockSize, blockSize);

	float_to_colorW << <drawBlocks, drawThreads >> >(controlBlock->device_bitmap, appBlock->dev_Data,
		appBlock->gridPlot->zoom, appBlock->gridPlot->dataSize, 
		controlBlock->winSize, appBlock->gridPlot->dataUpleft.x,
		appBlock->gridPlot->dataUpleft.y);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(controlBlock->cPUAnimBitmap->get_ptr(),
		controlBlock->device_bitmap,
		controlBlock->cPUAnimBitmap->image_size(), cudaMemcpyDeviceToHost));

}

void T(ControlBlock *controlBlock, int ticks)
{
	GridPlot *gridPlot;
	int2 dataSize, dataUpLeft;
	int plywidth = IntNamed(controlBlock->argc, controlBlock->argv, "plywidth", 64);
	int xStart = IntNamed(controlBlock->argc, controlBlock->argv, "xStart", 0);
	int yStart = IntNamed(controlBlock->argc, controlBlock->argv, "yStart", 0);
	dataSize.x = plywidth;
	dataSize.y = plywidth;
	dataUpLeft.x = xStart;
	dataUpLeft.y = yStart;

	float *host_data = BackSlashArray(plywidth, 0.5, 1);

	int zoom = IntNamed(controlBlock->argc, controlBlock->argv, "zoom", 2);
	MakeGridPlot(&gridPlot, zoom, dataSize, dataUpLeft);

	GridPlotAppBlock *appBlock;
	MakeGridPlotAppBlock(&appBlock, gridPlot, host_data);
	controlBlock->appBlock = appBlock;

	unsigned int dataMem = dataSize.x * dataSize.y * sizeof(float);
	unsigned int winMem = controlBlock->winSize.x * controlBlock->winSize.y * sizeof(float);
	appBlock->host_Data = host_data;
	checkCudaErrors(cudaMemcpy(appBlock->dev_Data, appBlock->host_Data, dataMem, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(controlBlock->device_bitmap, 0, winMem));

	int blockSize = IntNamed(controlBlock->argc, controlBlock->argv, "blockSize", 4);
	int gridSize = IntNamed(controlBlock->argc, controlBlock->argv, "gridSize", 4);
	dim3    drawBlocks(gridSize, gridSize);
	dim3    drawThreads(blockSize, blockSize);

	float_to_colorT << <drawBlocks, drawThreads >> >(controlBlock->device_bitmap, appBlock->dev_Data,
		appBlock->gridPlot->zoom, appBlock->gridPlot->dataSize,
		controlBlock->winSize, appBlock->gridPlot->dataUpleft.x,
		appBlock->gridPlot->dataUpleft.y);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(controlBlock->cPUAnimBitmap->get_ptr(),
		controlBlock->device_bitmap,
		controlBlock->cPUAnimBitmap->image_size(), cudaMemcpyDeviceToHost));

}

void M(ControlBlock *controlBlock, int ticks)
{
	GridPlot *gridPlot;
	int2 dataSize, dataUpLeft;
	int plywidth = IntNamed(controlBlock->argc, controlBlock->argv, "plywidth", 64);
	int xStart = IntNamed(controlBlock->argc, controlBlock->argv, "xStart", 0);
	int yStart = IntNamed(controlBlock->argc, controlBlock->argv, "yStart", 0);
	dataSize.x = plywidth;
	dataSize.y = plywidth;
	dataUpLeft.x = xStart;
	dataUpLeft.y = yStart;

	float *host_data = BackSlashArray(plywidth, 0.5, 1);

	int zoom = IntNamed(controlBlock->argc, controlBlock->argv, "zoom", 2);
	MakeGridPlot(&gridPlot, zoom, dataSize, dataUpLeft);

	GridPlotAppBlock *appBlock;
	MakeGridPlotAppBlock(&appBlock, gridPlot, host_data);
	controlBlock->appBlock = appBlock;

	unsigned int dataMem = dataSize.x * dataSize.y * sizeof(float);
	unsigned int winMem = controlBlock->winSize.x * controlBlock->winSize.y * sizeof(float);
	appBlock->host_Data = host_data;
	checkCudaErrors(cudaMemcpy(appBlock->dev_Data, appBlock->host_Data, dataMem, cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMemset(controlBlock->device_bitmap, 0, winMem));

	int blockSize = IntNamed(controlBlock->argc, controlBlock->argv, "blockSize", 4);
	int gridSize = IntNamed(controlBlock->argc, controlBlock->argv, "gridSize", 4);
	dim3    drawBlocks(gridSize, gridSize);
	dim3    drawThreads(blockSize, blockSize);

	float_to_color << <drawBlocks, drawThreads >> >(controlBlock->device_bitmap, appBlock->dev_Data,
		appBlock->gridPlot->zoom, appBlock->gridPlot->dataSize, 
		controlBlock->winSize, appBlock->gridPlot->dataUpleft.x,
		appBlock->gridPlot->dataUpleft.y);

	checkCudaErrors(cudaDeviceSynchronize());
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
	if (strcmp(comm, "m") == 0)
	{
		M(controlBlock, ticks);

		controlBlock->cPUAnimBitmap->ClearCommand();
		//G(controlBlock, ticks);
		return;
	}
	if (strcmp(comm, "t") == 0)
	{
		controlBlock->cPUAnimBitmap->ClearCommand();
		T(controlBlock, ticks);
		return;
	}
	if (strcmp(comm, "w") == 0)
	{
		controlBlock->cPUAnimBitmap->ClearCommand();
		W(controlBlock, ticks);
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
