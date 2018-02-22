#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "Utils.h"
#include "BlockUtils.h"



//Copies dev_floats to host_floats
cudaError_t Gpu_GetFloats(float **host_floats, float *dev_floats, int float_ct)
{
	cudaError_t cudaStatus = cudaSuccess;

	float *hostRes = (float*)malloc(float_ct * sizeof(float));
	CHECK_G(cudaMemcpy(hostRes, dev_floats, float_ct * sizeof(float), cudaMemcpyDeviceToHost));

	*host_floats = hostRes;
Error:
	return cudaStatus;
}

// Wraps (IxI) -> I cuda func.
cudaError_t Gpu_R2byR1(void(*f)(int *c, int *a, int *b, dim3 *gs, dim3 *bs, int length),
	int *c, int *a, int *b, dim3 *gridSize, dim3 *blockSize, int length)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	int gridVol = dim3Vol(gridSize);

	CHECK_G(cudaSetDevice(0));
	CHECK_G(cudaMalloc((void**)&dev_a, gridVol * sizeof(int)));
	CHECK_G(cudaMalloc((void**)&dev_b, gridVol * sizeof(int)));
	CHECK_G(cudaMalloc((void**)&dev_c, gridVol * sizeof(int)));


	CHECK_G(cudaMemcpy(dev_a, a, gridVol * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_G(cudaMemcpy(dev_b, b, gridVol * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_G(cudaMemcpy(dev_c, c, gridVol * sizeof(int), cudaMemcpyHostToDevice));


	// Launch a kernel on the GPU with one thread for each element.
	(*f)(dev_c, dev_a, dev_b, gridSize, blockSize, length);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	CHECK_G(cudaDeviceSynchronize());
	// Copy output vector from GPU buffer to host memory.
	CHECK_G(cudaMemcpy(c, dev_c, gridVol * sizeof(int), cudaMemcpyDeviceToHost));

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

// Wraps I -> I cuda func.
cudaError_t Gpu_R1byR1(void(*f)(int *c, int *a, dim3 *gs, dim3 *bs, int length),
	int *c, int *a, dim3 *gridSize, dim3 *blockSize, int length)
{
	int *dev_a = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	int gridVol = dim3Vol(gridSize);
 	int blockVol = dim3Vol(blockSize);
	int totalVol = gridVol * blockVol;

	CHECK_G(cudaSetDevice(0));
	CHECK_G(cudaMalloc((void**)&dev_a, totalVol * sizeof(int)));
	CHECK_G(cudaMalloc((void**)&dev_c, totalVol * sizeof(int)));

	CHECK_G(cudaMemcpy(dev_a, a, totalVol * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_G(cudaMemcpy(dev_c, c, totalVol * sizeof(int), cudaMemcpyHostToDevice));


	// Launch a kernel on the GPU with one thread for each element.
	(*f)(dev_c, dev_a, gridSize, blockSize, length);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	CHECK_G(cudaDeviceSynchronize());
	// Copy output vector from GPU buffer to host memory.
	CHECK_G(cudaMemcpy(c, dev_c, totalVol * sizeof(int), cudaMemcpyDeviceToHost));

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);

	return cudaStatus;
}

int *Cpu_R2byR1(dim3 *gridSize, int *av, int *bv)
{
	int gridVol = dim3Vol(gridSize);
	int *c = (int *)malloc(sizeof(int) * gridVol);

	for (int i = 0; i < gridVol; i++)
	{
		c[i] = av[i] + bv[i];
	}
	return c;
}
