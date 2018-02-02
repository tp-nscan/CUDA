
#include "cuda_runtime.h"

__global__ void Read_texture_obj_kernel(float *iptr, cudaTextureObject_t tex);

cudaTextureObject_t *TexObjFloat1D(float *devPtr, int length);