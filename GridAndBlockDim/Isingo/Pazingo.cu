#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include "../../common/book.h"
#include "../../common/cpu_anim2.h"
#include "../../common/Utils.h"
#include "../../common/BlockUtils.h"
#include "../../common/Texutils.h"
#include "../../common/Rando.h"
#include "../../common/PlyBlock.h"
#include "../../common/GridPlot.h"



struct AppBlock {
	cudaEvent_t      start, stop;
	float            totalTime;
	float            frames;
	float            speed;
	float            noise;
	int              threadChop;
	GridPlot        *gridPlot;
	float           *startingData;
	PlyBlock        *plyBlock;
};

void MakeControlBlock(ControlBlock **out, GridPlot *gridPlot, unsigned int plywidth, unsigned int seed, float speed,
	float noise, unsigned int threadChop);

__global__ void loco_kernel(float *dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut, int2 plySize, float speed);

__global__ void noiseTrimKernel(float *dst, curandState *dev_curandStates, float speed);

void anim_gpu(ControlBlock *d, int ticks);

void anim_exit(ControlBlock *d);



__global__ void float_to_color2(unsigned char *optr, const float *outSrc, int zoom, int plyWidth) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int plyOffset = x + y * plyWidth;
	int frameWidth = blockDim.x * gridDim.x;

	float l = outSrc[plyOffset];
	float s = 1;
	int h = (180 + (int)(360.0f * outSrc[plyOffset])) % 360;
	float m1, m2;

	if (l <= 0.5f)
		m2 = l * (1 + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;

	int imgOffset = (x + y * frameWidth * zoom) * zoom;
	for (int i = 0; i < zoom; i++)
	{
		for (int j = 0; j < zoom; j++)
		{
			int noff = (imgOffset + i * frameWidth* zoom + j) * 4;
			optr[noff + 0] = colorValue(m1, m2, h + 120);
			optr[noff + 1] = colorValue(m1, m2, h);
			optr[noff + 2] = colorValue(m1, m2, h - 120);
			optr[noff + 3] = 255;
		}
		__syncthreads();
	}

}

void SetDevicePlyBlockArray(AppBlock *appBlock, float *data)
{
	unsigned int plyMemSize = appBlock->plyBlock->area * sizeof(float);
	appBlock->startingData = data;
	HANDLE_ERROR(cudaMemcpy(appBlock->plyBlock->dev_inSrc, data, plyMemSize, cudaMemcpyHostToDevice));
}

void MakeAppBlock(AppBlock **out, GridPlot *gridPlot, unsigned int plyWidth, unsigned int seed, float speed,
	float noise, unsigned int threadChop)
{
	AppBlock *appBlock = (AppBlock *)malloc(sizeof(AppBlock));
	*out = appBlock;

	MakePlyBlock(&appBlock->plyBlock, plyWidth, seed);

	appBlock->speed = speed;
	appBlock->noise = noise;
	appBlock->gridPlot = gridPlot;
	appBlock->totalTime = 0;
	appBlock->frames = 0;
	appBlock->threadChop = threadChop;

	HANDLE_ERROR(cudaEventCreate(&appBlock->start));
	HANDLE_ERROR(cudaEventCreate(&appBlock->stop));
}

void MakeControlBlock(ControlBlock **out, GridPlot *gridPlot, unsigned int plywidth, unsigned int seed, float speed,
	                  float noise, unsigned int threadChop)
{
	ControlBlock *controlBlock = (ControlBlock *)malloc(sizeof(ControlBlock));

	CPUAnimBitmap *cbm = (CPUAnimBitmap *)malloc(sizeof(CPUAnimBitmap));
	cbm->Init(gridPlot->imageWidth, controlBlock);
	cbm->ClearCommand();
	HANDLE_ERROR(cudaMalloc((void**)&controlBlock->output_bitmap, cbm->image_size()));

	AppBlock *appBlock;
	MakeAppBlock(&appBlock, gridPlot, plywidth, seed, speed, noise, threadChop);
	controlBlock->appBlock = appBlock;
	
	*out = controlBlock;
}

int main(int argc, const char **argv)
{
	int plywidth = 512;
	int imageWidth = 64;
	int zoom = 1;
	int threadChop = 16;
	int xStart = 1;
	int yStart = 1;
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

	ControlBlock *controlBlock;
	GridPlot	 *gridPlot;

	MakeGridPlot(&gridPlot, zoom, imageWidth, xStart, yStart);
	MakeControlBlock(&controlBlock, gridPlot, plywidth, seed, speed, noise, threadChop);

	controlBlock->cPUAnimBitmap->anim_and_exit((void(*)(void*, int))anim_gpu,
		(void(*)(void*))anim_exit);

}


__global__ void noiseTrimKernel(float *dst, curandState *dev_curandStates, float speed) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	curandState localState = dev_curandStates[offset];
	float randy = curand_normal(&localState);

	float oriug = dst[offset];
	float res = oriug + speed * randy;

	if (res < -1)
	{
		dst[offset] = -1;
	}
	else if (res > 1)
	{
		dst[offset] = 1;
	}
	else
	{
		dst[offset] = res;
	}
}

__global__ void noiseTrimKernelR(float *dst, curandState *dev_curandStates, float speed) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	curandState localState = dev_curandStates[offset];
	float randy = curand_normal(&localState);

	float oriug = dst[offset];
	float res = oriug + speed * randy;

	if (res < -1)
	{
		dst[offset] = 1;
	}
	else if (res > 1)
	{
		dst[offset] = -1;
	}
	else
	{
		dst[offset] = res;
	}
}


__global__ void loco_kernel(float *dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut, int2 plySize, float speed) {
	
	int width = plySize.x;
	int2 nitDex = NodeInTilePos();

	//int x = threadIdx.x + blockIdx.x * blockDim.x;
	//int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = nitDex.x + nitDex.y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (nitDex.x == 0)   left++;
	if (nitDex.x == width - 1) right--;

	int top = offset - width;
	int bottom = offset + width;
	if (nitDex.y == 0)   top += width;
	if (nitDex.y == width - 1) bottom -= width;

	float   t, l, c, r, b;
	if (dstOut) {
		t = tex1Dfetch<float>(texIn, top);
		l = tex1Dfetch<float>(texIn, left);
		c = tex1Dfetch<float>(texIn, offset);
		r = tex1Dfetch<float>(texIn, right);
		b = tex1Dfetch<float>(texIn, bottom);

	}
	else {
		t = tex1Dfetch<float>(texOut, top);
		l = tex1Dfetch<float>(texOut, left);
		c = tex1Dfetch<float>(texOut, offset);
		r = tex1Dfetch<float>(texOut, right);
		b = tex1Dfetch<float>(texOut, bottom);
	}

	dst[offset] = c + speed * (t + b + r + l - 4 * c);
}

void L(ControlBlock *controlBlock, int ticks)
{
	AppBlock *appBlock = (AppBlock *)controlBlock->appBlock;

	HANDLE_ERROR(cudaEventRecord(appBlock->start, 0));
	int plyWidth = appBlock->plyBlock->width;

	dim3    simBlocks(plyWidth / appBlock->threadChop, plyWidth / appBlock->threadChop);
	dim3    simThreads(appBlock->threadChop, appBlock->threadChop);
	dim3    drawBlocks(plyWidth / (appBlock->threadChop*appBlock->gridPlot->zoom), plyWidth / (appBlock->threadChop*appBlock->gridPlot->zoom));
	dim3    drawThreads(appBlock->threadChop, appBlock->threadChop);

	CPUAnimBitmap  *cPUAnimBitmap = controlBlock->cPUAnimBitmap;

	volatile bool dstOut = true;
	for (int i = 0; i < 250; i++) {
		float   *in, *out;
		if (dstOut) {
			in = appBlock->plyBlock->dev_inSrc;
			out = appBlock->plyBlock->dev_outSrc;
		}
		else {
			out = appBlock->plyBlock->dev_inSrc;
			in = appBlock->plyBlock->dev_outSrc;
		}

		int2 plySize;
		plySize.x = appBlock->plyBlock->width;
		plySize.y = appBlock->plyBlock->width;

		loco_kernel << <simBlocks, simThreads >> > (
			out, dstOut, *appBlock->plyBlock->texIn, *appBlock->plyBlock->texOut,
			plySize, appBlock->speed);

		noiseTrimKernelR << <simBlocks, simThreads >> > (
			out, appBlock->plyBlock->randData->dev_curandStates,
			appBlock->noise);

		HANDLE_ERROR(cudaDeviceSynchronize());

		dstOut = !dstOut;
	}

	float_to_color2 << <drawBlocks, drawThreads >> >(controlBlock->output_bitmap, appBlock->plyBlock->dev_outSrc, appBlock->gridPlot->zoom, plyWidth);

	//float_to_color<< <simBlocks, simThreads >> >(appBlock->output_bitmap, appBlock->plyBlock->dev_outSrc);


	HANDLE_ERROR(cudaMemcpy(cPUAnimBitmap->get_ptr(), controlBlock->output_bitmap, cPUAnimBitmap->image_size(), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(appBlock->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(appBlock->stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, appBlock->start, appBlock->stop));
	appBlock->totalTime += elapsedTime;
	++appBlock->frames;
	printf("Average Time per frame:  %3.1f ms\n", appBlock->totalTime / appBlock->frames);
	printf("tic:  %d\n\n", ticks);
}


	void G(ControlBlock *controlBlock, int ticks)
	{
		int ta[] = { 1,2,3,4 };
		PrintIntArray(ta, 2, 4);

		//AppBlock *appBlock = (AppBlock *)controlBlock->appBlock;
		//HANDLE_ERROR(cudaEventRecord(appBlock->start, 0));
		//int plyWidth = appBlock->plyBlock->width;

		//dim3    simBlocks(plyWidth / appBlock->threadChop, plyWidth / appBlock->threadChop);
		//dim3    simThreads(appBlock->threadChop, appBlock->threadChop);
		//dim3    drawBlocks(plyWidth / (appBlock->threadChop*appBlock->zoom), plyWidth / (appBlock->threadChop*appBlock->zoom));
		//dim3    drawThreads(appBlock->threadChop, appBlock->threadChop);

		//CPUAnimBitmap  *cPUAnimBitmap = appBlock->cPUAnimBitmap;

		//volatile bool dstOut = true;
		//for (int i = 0; i<250; i++) {
		//	float   *in, *out;
		//	if (dstOut) {
		//		in = appBlock->plyBlock->dev_inSrc;
		//		out = appBlock->plyBlock->dev_outSrc;
		//	}
		//	else {
		//		out = appBlock->plyBlock->dev_inSrc;
		//		in = appBlock->plyBlock->dev_outSrc;
		//	}

		//	int2 plySize;
		//	plySize.x = appBlock->plyBlock->width;
		//	plySize.y = appBlock->plyBlock->width;

		//	loco_kernel << <simBlocks, simThreads >> >(
		//		out, dstOut, *appBlock->plyBlock->texIn, *appBlock->plyBlock->texOut,
		//		plySize, appBlock->speed);

		//	//HANDLE_ERROR(cudaDeviceSynchronize());
		//	//dim3    blocks2(32,32);
		//	//dim3    threads2(256);

		//	noiseTrimKernelR << <simBlocks, simThreads >> >(
		//		out, appBlock->plyBlock->randData->dev_curandStates,
		//		appBlock->noise);

		//	HANDLE_ERROR(cudaDeviceSynchronize());

		//	dstOut = !dstOut;

	}

void anim_gpu(ControlBlock *controlBlock, int ticks) {
	char *comm = controlBlock->command;
	AppBlock *appBlock = (AppBlock *)controlBlock->appBlock;

	if (strcmp(comm, "l") == 0)
	{
		float *init = RndFloat0to1(appBlock->plyBlock->area);
		SetDevicePlyBlockArray(appBlock, init);

		L(controlBlock, ticks);
		return;
	}
	if (strcmp(comm, "g") == 0)
	{
		controlBlock->cPUAnimBitmap->ClearCommand();
		G(controlBlock, ticks);
		return;
	}
	if (strlen(comm) > 0)
	{
		printf("unknown command: %s\n", comm);
		controlBlock->cPUAnimBitmap->ClearCommand();
		return;
	}
}


void anim_exit(ControlBlock *controlBlock) {
	AppBlock *appBlock = (AppBlock *)controlBlock->appBlock;

	FreePlyBlock(appBlock->plyBlock);

	HANDLE_ERROR(cudaEventDestroy(appBlock->start));
	HANDLE_ERROR(cudaEventDestroy(appBlock->stop));
}
