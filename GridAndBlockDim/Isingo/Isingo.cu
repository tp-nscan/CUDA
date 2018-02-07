//#include <stdlib.h>
//#include <stdio.h>
//#include <curand_kernel.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
////#include "../../common/book.h"
//#include "../../common/cpu_anim.h"
//#include "../../common/common.h"
//#include "../../common/DimStuff.h"
//#include "../../common/TexUtils.h"
//#include "../../common/PlyBlock.h"






//
//#define DIM 1024
//#define PI 3.1415926535897932f
//#define MAX_TEMP 1.0f
//#define MIN_TEMP 0.0001f
//#define SPEED   0.00005f
//
//
////// this kernel takes in a 2-d array of floats
////// it updates the value-of-interest by a scaled value based
////// on itself and its nearest neighbors
//__global__ void blend_kernel(float *dst, bool dstOut, cudaTextureObject_t texIn, cudaTextureObject_t texOut) {
//	// map from threadIdx/BlockIdx to pixel position
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * blockDim.x * gridDim.x;
//
//	int left = offset - 1;
//	int right = offset + 1;
//	if (x == 0)   left++;
//	if (x == DIM - 1) right--;
//
//	int top = offset - DIM;
//	int bottom = offset + DIM;
//	if (y == 0)   top += DIM;
//	if (y == DIM - 1) bottom -= DIM;
//
//	float   t, l, c, r, b;
//	if (dstOut) {
//		t = tex1Dfetch<float>(texIn, top);
//		l = tex1Dfetch<float>(texIn, left);
//		c = tex1Dfetch<float>(texIn, offset);
//		r = tex1Dfetch<float>(texIn, right);
//		b = tex1Dfetch<float>(texIn, bottom);
//
//	}
//	else {
//		t = tex1Dfetch<float>(texOut, top);
//		l = tex1Dfetch<float>(texOut, left);
//		c = tex1Dfetch<float>(texOut, offset);
//		r = tex1Dfetch<float>(texOut, right);
//		b = tex1Dfetch<float>(texOut, bottom);
//	}
//	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
//}
//
//
//
//void anim_gpu(AppBlock *appBlock, int ticks) {
//	//HANDLE_ERROR(cudaEventRecord(appBlock->start, 0));
//	//dim3    blocks(DIM / 16, DIM / 16);
//	//dim3    threads(16, 16);
//	//CPUAnimBitmap  *bitmap = appBlock->bitmap;
//
//	//// since tex is global and bound, we have to use a flag to
//	//// select which is in/out per iteration
//	//volatile bool dstOut = true;
//	//for (int i = 0; i<100; i++) {
//	//	float   *in, *out;
//	//	if (dstOut) {
//	//		in = appBlock->d->dev_inSrc;
//	//		out = appBlock->d->dev_outSrc;
//	//	}
//	//	else {
//	//		out = appBlock->d->dev_inSrc;
//	//		in = appBlock->d->dev_outSrc;
//	//	}
//	//	blend_kernel << <blocks, threads >> >(out, dstOut, *appBlock->d->texIn, *appBlock->d->texOut);
//	//	dstOut = !dstOut;
//	//}
//
//	//float_to_color << <blocks, threads >> >(appBlock->d->output_bitmap, appBlock->d->dev_outSrc);
//
//	//HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),
//	//	d->output_bitmap,
//	//	bitmap->image_size(),
//	//	cudaMemcpyDeviceToHost));
//
//	//HANDLE_ERROR(cudaEventRecord(appBlock->stop, 0));
//	//HANDLE_ERROR(cudaEventSynchronize(d->stop));
//	//float   elapsedTime;
//	//HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
//	//	d->start, d->stop));
//	//d->totalTime += elapsedTime;
//	//++d->frames;
//	//printf("Average Time per frame:  %3.1f ms\n", d->totalTime / d->frames);
//	//printf("tic:  %d\n\n", ticks);
//}
//
//// clean up memory allocated on the GPU
//
//void anim_exit(AppBlock *d) {
//
//	//cudaDestroyTextureObject(*(d->texIn));
//	//cudaDestroyTextureObject(*d->texOut);
//	//cudaDestroyTextureObject(*d->texConst);
//
//	//HANDLE_ERROR(cudaFree(d->dev_inSrc));
//	//HANDLE_ERROR(cudaFree(d->dev_outSrc));
//	//HANDLE_ERROR(cudaFree(d->dev_constSrc));
//
//	//HANDLE_ERROR(cudaEventDestroy(d->start));
//	//HANDLE_ERROR(cudaEventDestroy(d->stop));
//}
//
//
////int main(void)
////{
////	PlyBlock   data;
////	CPUAnimBitmap bitmap(DIM, DIM, &data);
////	data.bitmap = &bitmap;
////	data.totalTime = 0;
////	data.frames = 0;
////	HANDLE_ERROR(cudaEventCreate(&data.start));
////	HANDLE_ERROR(cudaEventCreate(&data.stop));
////
////	int imageSize = bitmap.image_size();
////
////	HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap,
////		imageSize));
////
////	// assume float == 4 chars in size (ie rgba)
////	HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, imageSize));
////	HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, imageSize));
////	HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, imageSize));
////
////
////	data.texIn = TexObjFloat1D(data.dev_inSrc, imageSize);
////	data.texOut = TexObjFloat1D(data.dev_outSrc, imageSize);
////	data.texConst = TexObjFloat1D(data.dev_constSrc, imageSize);
////
////
////
////	// intialize the constant data
////	float *temp = RndFloat0to1(DIM*DIM);
////
////	HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, imageSize,
////		cudaMemcpyHostToDevice));
////
////
////	HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, imageSize,
////		cudaMemcpyHostToDevice));
////	free(temp);
////
////	bitmap.anim_and_exit((void(*)(void*, int))anim_gpu,
////		(void(*)(void*))anim_exit);
////}
