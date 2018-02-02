//
//#include "../../common/common.h"
//#include "../../common/DimStuff.h"
//#include "../../common/TexUtils.h"
//#include "../../common/book.h"
//#include "../../common/cpu_anim.h"
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdlib.h>
//#include <stdio.h>
//
//bool wank();
//bool TestCudaMemRead();
//bool TestCudaTexObjRead();
//bool TestCudaTexRead();
//
//texture<float>  texIn;
//
//
//
//struct TestState {
//	unsigned char   *output_bitmap;
//	float           *dev_inSrc;
//	float           *dev_outSrc;
//	float           *dev_randSrc;
//
//	cudaEvent_t     start, stop;
//	float           totalTime;
//	float           frames;
//};
//
//
//__global__ void Read_texture_kernel(float *iptr) {
//	// map from threadIdx/BlockIdx to pixel position
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * blockDim.x * gridDim.x;
//
//	float c = tex1Dfetch(texIn, offset);
//	iptr[offset] = c;
//}
//
//
////int main(void) {
////
////	printf("wank result: %d\n", wank());
////	printf("TestCudaMemRead result: %d\n", TestCudaMemRead());
////	printf("TestCudaTexRead result: %d\n", TestCudaTexRead());
////	printf("TestCudaTexObjRead result: %d\n", TestCudaTexObjRead());
////}
//
//
//bool wank()
//{
//	int arraySize = 64;
//	float *dev_ptr;
//	cudaTextureObject_t *texObj;
//
//	HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, sizeof(float) * arraySize));
//
//	float *h_randsIn = RndFloat0to1(arraySize);
//
//	HANDLE_ERROR(cudaMemcpy(dev_ptr, h_randsIn, arraySize * sizeof(float),
//		cudaMemcpyHostToDevice));
//
//	texObj = TexObjFloat1D(dev_ptr, arraySize);
//
//	return true;
//}
//
//bool TestCudaMemRead()
//{
//	int arraySize = 64;
//	float *dev_ptr;
//	HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, sizeof(float) * arraySize));
//
//	float *h_randsIn = RndFloat0to1(arraySize);
//
//	float *h_randsOut = (float *)malloc(sizeof(float) * arraySize);
//
//	HANDLE_ERROR(cudaMemcpy(dev_ptr, h_randsIn, arraySize * sizeof(float),
//		cudaMemcpyHostToDevice));
//
//	HANDLE_ERROR(cudaMemcpy(h_randsOut, dev_ptr, arraySize * sizeof(float),
//		cudaMemcpyDeviceToHost));
//
//	cudaFree(dev_ptr);
//
//	return CompFloatArrays(h_randsIn, h_randsOut, arraySize);
//}
//
//bool TestCudaTexRead()
//{
//	int arraySqrt = 8;
//	int arraySize = 64;
//	dim3    blocks(arraySqrt);
//	dim3    threads(arraySqrt);
//	float *dev_ptr;
//	float *dev_ptrCopy;
//
//	HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, sizeof(float) * arraySize));
//	HANDLE_ERROR(cudaMalloc((void**)&dev_ptrCopy, sizeof(float) * arraySize));
//	HANDLE_ERROR(cudaBindTexture(NULL, texIn, dev_ptr, sizeof(float) *arraySize));
//
//	float *h_randsIn = RndFloat0to1(arraySize);
//	float *h_randsOut = (float *)malloc(sizeof(float) * arraySize);
//
//	HANDLE_ERROR(cudaMemcpy(dev_ptr, h_randsIn, arraySize * sizeof(float),
//		cudaMemcpyHostToDevice));
//
//	Read_texture_kernel << <blocks, threads >> > (dev_ptrCopy);
//
//	HANDLE_ERROR(cudaMemcpy(h_randsOut, dev_ptrCopy, arraySize * sizeof(float),
//		cudaMemcpyDeviceToHost));
//
//	cudaFree(dev_ptr);
//	cudaFree(dev_ptrCopy);
//
//	return CompFloatArrays(h_randsIn, h_randsOut, arraySize);
//}
//
//bool TestCudaTexObjRead()
//{
//	int arraySqrt = 8;
//	int arraySize = 64;
//	dim3    blocks(arraySqrt);
//	dim3    threads(arraySqrt);
//
//	float *dev_ptr;
//	float *dev_ptrCopy;
//	cudaTextureObject_t *texObj;
//
//	HANDLE_ERROR(cudaMalloc((void**)&dev_ptr, sizeof(float) * arraySize));
//	HANDLE_ERROR(cudaMalloc((void**)&dev_ptrCopy, sizeof(float) * arraySize));
//
//	float *h_randsIn = RndFloat0to1(arraySize);
//	float *h_randsOut = (float *)malloc(sizeof(float) * arraySize);
//
//	HANDLE_ERROR(cudaMemcpy(dev_ptr, h_randsIn, arraySize * sizeof(float),
//		cudaMemcpyHostToDevice));
//
//	texObj = TexObjFloat1D(dev_ptr, arraySize);
//
//	Read_texture_obj_kernel << <blocks, threads >> > (dev_ptrCopy, *texObj);
//
//	HANDLE_ERROR(cudaMemcpy(h_randsOut, dev_ptrCopy, arraySize * sizeof(float),
//		cudaMemcpyDeviceToHost));
//
//	// free device mems
//	cudaDestroyTextureObject(*texObj);
//	cudaFree(dev_ptr);
//	cudaFree(dev_ptrCopy);
//
//	return CompFloatArrays(h_randsIn, h_randsOut, arraySize);
//}