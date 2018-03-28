
#include "cuda_runtime.h"

__global__ void copy_tex(float *iptr, cudaTextureObject_t tex);

__global__ void tex_1d_2d_1d(float *resOut, cudaTextureObject_t texIn, int2 plySize, 
	float speed, float decay);

cudaTextureObject_t *TexObjFloat1D(float *devPtr, int length);