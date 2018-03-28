#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BlockUtils.h"

////////////////////////////////////////////////////////////////////////////////
// tex_1d_2d_1d
// Copy a 1d texture to out_ptr, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void copy_tex(float *out_ptr, cudaTextureObject_t tex) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch<float>(tex, offset);
	out_ptr[offset] = c;
}

////////////////////////////////////////////////////////////////////////////////
// TexObjFloat1D
// Wrap a 1d texture around a float array
////////////////////////////////////////////////////////////////////////////////
cudaTextureObject_t *TexObjFloat1D(float *devPtr, int length)
{
	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = devPtr;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = length * sizeof(float);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t *tex = (cudaTextureObject_t *)malloc(sizeof(cudaTextureObject_t));
	cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
	return tex;
}

////////////////////////////////////////////////////////////////////////////////
// tex_1d_2d_1d
// A 1d texture, working on a 2d torus, using 1d blocks and threads
////////////////////////////////////////////////////////////////////////////////
__global__ void tex_1d_2d_1d(float *resOut, cudaTextureObject_t texIn, int2 plySize, 
	float speed, float decay) {

	int plyLength = plySize.x * plySize.y;
	int step = blockDim.x * gridDim.x;

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < plyLength; i += step)
	{
		int col = i % plySize.x;
		int row = (i - col) / plySize.x;
		int colm = (col - 1 + plySize.x) % plySize.x;
		int colp = (col + 1 + plySize.x) % plySize.x;
		int rowm = (row - 1 + plySize.y) % plySize.y;
		int rowp = (row + 1 + plySize.y) % plySize.y;

		int left = colm + row * plySize.x;
		int right = colp + row * plySize.x;
		int top = col + rowm * plySize.x;
		int bottom = col + rowp * plySize.x;
		int center = col + row * plySize.x;

		float t = tex1Dfetch<float>(texIn, top);
		float l = tex1Dfetch<float>(texIn, left);
		float c = tex1Dfetch<float>(texIn, center);
		float r = tex1Dfetch<float>(texIn, right);
		float b = tex1Dfetch<float>(texIn, bottom);

		resOut[center] = c + speed * (t + b + r + l - decay * c);
	}
}



