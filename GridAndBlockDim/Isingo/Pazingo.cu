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

struct AppBlock {
	unsigned char   *output_bitmap;
	CPUAnimBitmap  *cPUAnimBitmap;

	cudaEvent_t     start, stop;
	float           totalTime;
	float           frames;
	float           speed;
	int				mag;
	PlyBlock *plyBlock;
};


__device__ __inline__ int2 TileSize();

//__device__ __inline__ int2 NodeInTilePos();

__device__ __inline__ int NodeIndexInPly(int2 nitDex, int2 tipDex, int2 tileSize, int2 plySize);

__device__ __inline__ int2 NodeInPlyPos(int2 nitDex, int2 tipDex, int2 tileSize);

__global__ void loco_kernel(float *dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut, int2 plySize, float speed);

__global__ void noiseTrimKernel(float *dst, curandState *dev_curandStates, float speed);

void anim_gpu(AppBlock *d, int ticks);

void anim_exit(AppBlock *d);

__device__ __inline__ int2 TileSize()
{
	int2 step;
	step.x = blockDim.x* gridDim.x;
	step.y = blockDim.y* gridDim.y;
	return step;
}


__device__ __inline__ int NodeIndexInPly(int2 nitDex, int2 tipDex, int2 tileSize, int2 plySize)
{
	return
		tileSize.y + plySize.x * tipDex.y +
		tileSize.y + tileSize.x * tipDex.x +
		tileSize.x + nitDex.y +
		nitDex.x;
}

__device__ __inline__ int2 NodeInPlyPos(int2 nitDex, int2 tipDex, int2 tileSize)
{
	int2 step;
	step.x = tileSize.x * tipDex.x + nitDex.x;
	step.y = tileSize.y * tipDex.y + nitDex.y;
	return step;
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

__global__ void float_to_color2(unsigned char *optr, const float *outSrc, int mag, int plyWidth) {
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

	int imgOffset = (x + y * frameWidth * mag) * mag;
	for (int i = 0; i < mag; i++)
	{
		for (int j = 0; j < mag; j++)
		{
			int noff = (imgOffset + i * frameWidth* mag + j) * 4;
			optr[noff + 0] = colorValue(m1, m2, h + 120);
			optr[noff + 1] = colorValue(m1, m2, h);
			optr[noff + 2] = colorValue(m1, m2, h - 120);
			optr[noff + 3] = 255;
		}
		__syncthreads();
	}

}


void MakeAppBlock(AppBlock **out, unsigned int width, unsigned int seed, float speed, int mag)
{
	AppBlock *appBlock = (AppBlock *)malloc(sizeof(AppBlock));
	*out = appBlock;

	MakePlyBlock(&appBlock->plyBlock, width, seed);
	unsigned int plyMemSize = appBlock->plyBlock->area * sizeof(float);

	appBlock->speed = speed;
	appBlock->mag = mag;
	appBlock->totalTime = 0;
	appBlock->frames = 0;
	HANDLE_ERROR(cudaEventCreate(&appBlock->start));
	HANDLE_ERROR(cudaEventCreate(&appBlock->stop));

	// intialize the constant data
	float *temp = RndFloat0to1(width*width);
	//HANDLE_ERROR(cudaMemcpy(appBlock->plyBlock->dev_constSrc, temp, plyMemSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(appBlock->plyBlock->dev_inSrc, temp, plyMemSize, cudaMemcpyHostToDevice));
	free(temp);
}

int main(int argc, const char **argv)
{
	int gridWidth = 512;
	int imageWidth = 64;
	int mag = 1;
	float speed = 0.1;
	float noise = 0.5;

	if (checkCmdLineFlag(argc, (const char **)argv, "gridwidth"))
	{
		gridWidth = getCmdLineArgumentInt(argc, (const char **)argv, "gridwidth");
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
	if (checkCmdLineFlag(argc, (const char **)argv, "mag"))
	{
		mag = getCmdLineArgumentInt(argc, (const char **)argv, "mag");
	}

	AppBlock *appBlock;
	MakeAppBlock(&appBlock, gridWidth, 1283, speed, mag);
	CPUAnimBitmap cPUAnimBitmap(imageWidth, imageWidth, appBlock);
	appBlock->cPUAnimBitmap = &cPUAnimBitmap;
	HANDLE_ERROR(cudaMalloc((void**)&appBlock->output_bitmap, cPUAnimBitmap.image_size()));

	cPUAnimBitmap.anim_and_exit((void(*)(void*, int))anim_gpu,
		(void(*)(void*))anim_exit);

}


__global__ void noiseTrimKernel(float *dst, curandState *dev_curandStates, float speed) {
	// map from threadIdx/BlockIdx to pixel position
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
	// map from threadIdx/BlockIdx to pixel position
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


void anim_gpu(AppBlock *appBlock, int ticks) {
	HANDLE_ERROR(cudaEventRecord(appBlock->start, 0));
	const int chop = 16;
	int plyWidth = appBlock->plyBlock->width;

	dim3    simBlocks(plyWidth / chop, plyWidth / chop);
	dim3    simThreads(chop, chop);
	dim3    drawBlocks(plyWidth / (chop*appBlock->mag), plyWidth / (chop*appBlock->mag));
	dim3    drawThreads(chop, chop);

	CPUAnimBitmap  *cPUAnimBitmap = appBlock->cPUAnimBitmap;

	// since tex is global and bound, we have to use a flag to
	// select which is in/out per iteration
	volatile bool dstOut = true;
	for (int i = 0; i<250; i++) {
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

		//loco_kernel << <simBlocks, simThreads >> >(
		//	out, dstOut, *appBlock->plyBlock->texIn, *appBlock->plyBlock->texOut,
		//	plySize, appBlock->speed*0.5);

		//HANDLE_ERROR(cudaDeviceSynchronize());
		//dim3    blocks2(32,32);
		//dim3    threads2(256);

		noiseTrimKernelR << <simBlocks, simThreads >> >(
			out, appBlock->plyBlock->randData->dev_curandStates,
			appBlock->speed);

		HANDLE_ERROR(cudaDeviceSynchronize());

		dstOut = !dstOut;
	}

	float_to_color2 << <drawBlocks, drawThreads >> >(appBlock->output_bitmap, appBlock->plyBlock->dev_outSrc, appBlock->mag, plyWidth);

	//float_to_color<< <simBlocks, simThreads >> >(appBlock->output_bitmap, appBlock->plyBlock->dev_outSrc);

	void *q = cPUAnimBitmap->get_ptr();
	unsigned char  *r = appBlock->output_bitmap;
	int sz = cPUAnimBitmap->image_size();

	HANDLE_ERROR(cudaMemcpy(cPUAnimBitmap->get_ptr(), appBlock->output_bitmap, cPUAnimBitmap->image_size(), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(appBlock->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(appBlock->stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, appBlock->start, appBlock->stop));
	appBlock->totalTime += elapsedTime;
	++appBlock->frames;
	printf("Average Time per frame:  %3.1f ms\n", appBlock->totalTime / appBlock->frames);
	printf("tic:  %d\n\n", ticks);
}


void anim_exit(AppBlock *appBlock) {

	cudaDestroyTextureObject(*(appBlock->plyBlock->texIn));
	cudaDestroyTextureObject(*appBlock->plyBlock->texOut);
	//cudaDestroyTextureObject(*appBlock->plyBlock->texConst);

	HANDLE_ERROR(cudaFree(appBlock->plyBlock->dev_inSrc));
	HANDLE_ERROR(cudaFree(appBlock->plyBlock->dev_outSrc));
	HANDLE_ERROR(cudaFree(appBlock->plyBlock->dev_constSrc));

	HANDLE_ERROR(cudaEventDestroy(appBlock->start));
	HANDLE_ERROR(cudaEventDestroy(appBlock->stop));
}
