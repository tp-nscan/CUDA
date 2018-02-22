//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdlib.h>
//#include <stdio.h>
//
//__global__ void Read_texture_obj_kernel(float *iptr, cudaTextureObject_t tex) {
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * blockDim.x * gridDim.x;
//
//	float c = tex1Dfetch<float>(tex, offset);
//	iptr[offset] = c;
//}
//
//
//cudaTextureObject_t *TexObjFloat1D(float *devPtr, int length)
//{
//	// create texture object
//	cudaResourceDesc resDesc;
//	memset(&resDesc, 0, sizeof(resDesc));
//	resDesc.resType = cudaResourceTypeLinear;
//	resDesc.res.linear.devPtr = devPtr;
//	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
//	resDesc.res.linear.desc.x = 32; // bits per channel
//	resDesc.res.linear.sizeInBytes = length * sizeof(float);
//
//	cudaTextureDesc texDesc;
//	memset(&texDesc, 0, sizeof(texDesc));
//	texDesc.readMode = cudaReadModeElementType;
//
//	cudaTextureObject_t *tex = (cudaTextureObject_t *)malloc(sizeof(cudaTextureObject_t));
//	cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
//	return tex;
//}
