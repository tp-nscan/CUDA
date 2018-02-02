
#include "../../common/common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

cudaError_t addWithCuda(int *c, int *a, int *b, dim3 size);

cudaError_t addLongsWithCuda(long *c, long *a, long *b, unsigned long size);

void DoIntAdd();
void DoLongAdd();

void CompCpuGpu(dim3 *arraySize);
void DynLongAdd();

void DoCudaStuff();

__global__ void addKernel_T(int *c, int *a, int *b)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int i = x + y; // * blockDim.x;
	c[i] = a[i] + b[i];
}

__global__ void addKernel_B(int *c, int *a, int *b)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int i = x + y * gridDim.x;
	c[i] = a[i] + b[i];
}

__global__ void addLongKernel(long *c, long *a, long *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void IntVector(int *vec, int ct, int mult, int offset)
{
	for (int i = 0; i < ct; i++)
	{
		vec[i] = offset + i * mult;
	}
}

void LongVector(long *vec, long ct, long mult, long offset)
{
	for (int i = 0; i < ct; i++)
	{
		vec[i] = offset + i * mult;
	}
}


dim3 *dim3Ctr(int x, int y = 1, int z = 1)
{
	dim3 *a;
	a = (dim3 *)malloc(sizeof(dim3));
	a->x = x;
	a->y = y;
	a->z = z;
	return a;
}


void printDim3(dim3 *yow)
{
	printf("yow: {%d, %d, %d}", yow->x, yow->y, yow->z);
}

int main()
{
	DoCudaStuff();
    return 0;
}


void DoCudaStuff()
{
	//int ck = 6400;
	int ck = 400;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{

			dim3 *td = dim3Ctr(ck + ck * i, ck + ck * j);
			CompCpuGpu(td);
		}
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK(cudaDeviceReset());
}

int *Gpu_R2by1(dim3 *arraySize, int *av, int *bv)
{
	int arrayLength = arraySize->x * arraySize->y;
	int *c =  (int *)malloc(sizeof(int) * arrayLength);

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, av, bv, *arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		exit(1);
	}

	return c;
}

int *Cpu_R2by1(dim3 *arraySize, int *av, int *bv)
{
	int arrayLength = arraySize->x * arraySize->y;
	int *c = (int *)malloc(sizeof(int) * arrayLength);

	for (int i = 0; i < arrayLength; i++)
	{
		c[i] = av[i] + bv[i];
	}
	return c;
}

bool CompArrays(int *a, int *b, int length)
{
	for (int i = 0; i < length; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}

void CompCpuGpu(dim3 *arraySize)
{
	int arrayLength = arraySize->x * arraySize->y;

	int *av = (int *)malloc(sizeof(int) * arrayLength);
	int *bv = (int *)malloc(sizeof(int) * arrayLength);
	IntVector(av, arrayLength, 1, 1);
	IntVector(bv, arrayLength, 10, 0);

	int *cGPU = Gpu_R2by1(arraySize, av, bv);
	int *cCPU = Cpu_R2by1(arraySize, av, bv);

	printf("arrayLength: %d ", arrayLength);
	CompArrays(cCPU, cGPU, arrayLength) ? printf("pass\n") : printf("fail\n");
	
	free(cGPU);
	free(cCPU);
	free(av);
	free(bv);
}

void DynLongAdd()
{
	const long arraySize = 500;
	long c[arraySize] = { 0 };
	long av[arraySize], bv[arraySize];
	LongVector(av, arraySize, 1, 1);
	LongVector(bv, arraySize, 10, 0);

	// Add vectors in parallel.
	cudaError_t cudaStatus = addLongsWithCuda(c, av, bv, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		exit(1);
	}

	printf("first five terms of c = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);
}


void DoIntAdd()
{
	const int arraySize = 50;
	int c[arraySize] = { 0 };
	int av[arraySize], bv[arraySize];
	IntVector(av, arraySize, 1, 1);
	IntVector(bv, arraySize, 10, 0);

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, av, bv, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		exit(1);
	}

	printf("first five terms of c = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);
}

void DoLongAdd()
{
	const long arraySize = 500;
	long c[arraySize] = { 0 };
	long av[arraySize], bv[arraySize];
	LongVector(av, arraySize, 1, 1);
	LongVector(bv, arraySize, 10, 0);

	// Add vectors in parallel.
	cudaError_t cudaStatus = addLongsWithCuda(c, av, bv, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		exit(1);
	}

	printf("first five terms of c = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addLongsWithCuda(long *c, long *a, long *b, unsigned long size)
{
	 long *dev_a = 0;
	 long *dev_b = 0;
	 long *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(long), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addLongKernel <<<1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(long), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int *a, int *b, dim3 arraySize)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

	int arrayLength = arraySize.x * arraySize.y;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, arrayLength * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, arrayLength * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, arrayLength * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, arrayLength * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, arrayLength * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	addKernel_T <<<1, arraySize>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, arrayLength * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
