#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include "../../common/book.h"
#include "../../common/cpu_anim.h"
#include "../../common/DimStuff.h"
#include "../../common/Texutils.h"
#include "../../common/Rando.h"
#include "../../common/PlyBlock.h"

struct AppBlock {
	unsigned char   *output_bitmap;
	CPUAnimBitmap  *cPUAnimBitmap;

	cudaEvent_t     start, stop;
	float           totalTime;
	float           frames;

	PlyBlock *plyBlock;
};

__global__ void blend_kernel(float *dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut, curandState *dev_curandStates, int width, float speed);

__global__ void loco_kernel(float *dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut, int width, float speed);

__global__ void noiseTrimKernel(float *dst, curandState *dev_curandStates, float speed);

void anim_gpu(AppBlock *d, int ticks);

void anim_exit(AppBlock *d);

void MakeAppBlock(AppBlock **out, unsigned int width, unsigned int seed)
{
	AppBlock *appBlock = (AppBlock *)malloc(sizeof(AppBlock));;
	*out = appBlock;

	MakePlyBlock(&appBlock->plyBlock, width, seed);
	unsigned int plyMemSize = appBlock->plyBlock->area * sizeof(float);

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

int main()
{
	int imageWidth = 512;

	AppBlock *appBlock;
	MakeAppBlock(&appBlock, imageWidth, 1283);
	CPUAnimBitmap cPUAnimBitmap(imageWidth, imageWidth, appBlock);
	appBlock->cPUAnimBitmap = &cPUAnimBitmap;
	HANDLE_ERROR(cudaMalloc((void**)&appBlock->output_bitmap, cPUAnimBitmap.image_size()));

	cPUAnimBitmap.anim_and_exit((void(*)(void*, int))anim_gpu,
		(void(*)(void*))anim_exit);

}


#define SPEED   0.1f

//// this kernel takes in a 2-d array of floats
//// it updates the value-of-interest by a scaled value based
//// on itself and its nearest neighbors
__global__ void blend_kernel(float *dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut, curandState *dev_curandStates, int width, float speed) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0)   left++;
	if (x == width - 1) right--;

	int top = offset - width;
	int bottom = offset + width;
	if (y == 0)   top += width;
	if (y == width - 1) bottom -= width;

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

	curandState localState = dev_curandStates[offset];
	float randy = curand_normal(&localState);

	float res = c + speed * (t + b + r + l - 4 * c) + randy * 0.0015;

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


//// this kernel takes in a 2-d array of floats
//// it updates the value-of-interest by a scaled value based
//// on itself and its nearest neighbors
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

//// on itself and its nearest neighbors
__global__ void loco_kernel(float *dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut, int width, float speed) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0)   left++;
	if (x == width - 1) right--;

	int top = offset - width;
	int bottom = offset + width;
	if (y == 0)   top += width;
	if (y == width - 1) bottom -= width;

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
	int width = appBlock->plyBlock->width;
	dim3    blocks(width / chop, width / chop);
	dim3    threads(chop, chop);
	CPUAnimBitmap  *cPUAnimBitmap = appBlock->cPUAnimBitmap;

	// since tex is global and bound, we have to use a flag to
	// select which is in/out per iteration
	volatile bool dstOut = true;
	for (int i = 0; i<50; i++) {
		float   *in, *out;
		if (dstOut) {
			in = appBlock->plyBlock->dev_inSrc;
			out = appBlock->plyBlock->dev_outSrc;
		}
		else {
			out = appBlock->plyBlock->dev_inSrc;
			in = appBlock->plyBlock->dev_outSrc;
		}

		loco_kernel << <blocks, threads >> >(
			out, dstOut, *appBlock->plyBlock->texIn, *appBlock->plyBlock->texOut,
			width, SPEED*3);

		//HANDLE_ERROR(cudaDeviceSynchronize());

		noiseTrimKernel << <blocks, threads >> >(
			out, appBlock->plyBlock->randData->dev_curandStates,
			SPEED );

		//HANDLE_ERROR(cudaDeviceSynchronize());

		dstOut = !dstOut;
	}

	float_to_color << <blocks, threads >> >(appBlock->output_bitmap, appBlock->plyBlock->dev_outSrc);

	void *q = cPUAnimBitmap->get_ptr();
	unsigned char  *r = appBlock->output_bitmap;
	int sz = cPUAnimBitmap->image_size();

	HANDLE_ERROR(cudaMemcpy(cPUAnimBitmap->get_ptr(), appBlock->output_bitmap, cPUAnimBitmap->image_size(), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(appBlock->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(appBlock->stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,	appBlock->start, appBlock->stop));
	appBlock->totalTime += elapsedTime;
	++appBlock->frames;
	printf("Average Time per frame:  %3.1f ms\n", appBlock->totalTime / appBlock->frames);
	printf("tic:  %d\n\n", ticks);
}


void anim_exit(AppBlock *appBlock) {

	cudaDestroyTextureObject(*(appBlock->plyBlock->texIn));
	cudaDestroyTextureObject(*appBlock->plyBlock->texOut);
	cudaDestroyTextureObject(*appBlock->plyBlock->texConst);

	HANDLE_ERROR(cudaFree(appBlock->plyBlock->dev_inSrc));
	HANDLE_ERROR(cudaFree(appBlock->plyBlock->dev_outSrc));
	HANDLE_ERROR(cudaFree(appBlock->plyBlock->dev_constSrc));

	HANDLE_ERROR(cudaEventDestroy(appBlock->start));
	HANDLE_ERROR(cudaEventDestroy(appBlock->stop));
}
