
#include "../../common/common.h"
#include "../../common/DimStuff.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>


__global__ void addKernel_B(int *c, int *a, int *b, int length)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int i = x + y * gridDim.x;
	if (i < length)
	{
		c[i] = a[i] + b[i];
	}
}

// grid 2D block 2D
__global__ void addKernel_BT(int *c, int *a, int *b, int length)
{
	//unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	//unsigned int idx = iy * nx + ix;

	//if (ix < length)
	//	c[idx] = a[idx] + b[idx];
	int xT = threadIdx.x;
	int yT = threadIdx.y;
	int iT = xT + yT * blockDim.x;

	int blockVol = blockDim.x * blockDim.y;

	int xB = blockIdx.x;
	int yB = blockIdx.y;
	int iB = xB + yB * gridDim.x;
	int iBT = iB * blockVol + iT;

	if (iBT < length)
	{
		c[iBT] = a[iBT] + b[iBT];
	}
}

__global__ void Check_BT(int *c, int *a, int length)
{
	unsigned int xB = blockIdx.x;
	unsigned int yB = blockIdx.y;
	unsigned int gV = gridDim.x * gridDim.y * gridDim.z;

	unsigned int xT = threadIdx.x;
	unsigned int yT = threadIdx.y;
	unsigned int tV = blockDim.x * blockDim.y * blockDim.z;

	unsigned int cB = xB + yB * gridDim.x;
	unsigned int cT = xT + yT * blockDim.x;
	unsigned int cI = cT + cB * tV;
	if (cI < length)
	{
		c[cI] = a[cI];
	}
}

__global__ void Check_B(int *c, int *a, int length)
{
	unsigned int xB = blockIdx.x;
	unsigned int yB = blockIdx.y;
	unsigned int gV = gridDim.x;

	unsigned int xT = threadIdx.x;
	unsigned int yT = threadIdx.y;

	unsigned int iT = xT + yT * blockDim.x;
	if (iT < length)
	{
		c[iT] = a[iT];
	}


	//unsigned int iB = xB + yB * gridDim.x;
	//if (iB < length)
	//{
	//	c[iB] = a[iB];
	//}
}


void GridAndBlockCheck(int *dev_c, int *dev_a, dim3 *gridSize, dim3 *blockSize, int arrayLen)
{
	Check_BT<<<*gridSize, *blockSize>>>(dev_c, dev_a, arrayLen);
}

void TestGridAndBlock()
{
	dim3 *gridSize = dim3Ctr(2, 2);
	dim3 *blockSize = dim3Ctr(16, 16);

	//dim3 *gridSize = dim3Ctr(3, 2);
	//dim3 *blockSize = dim3Unit();

	int bTVol = dim3Vol(gridSize) * dim3Vol(blockSize);

	int *a = IntArray(bTVol, 0, 1);
	int *cGPU = IntArray(bTVol, 0, 0);

	// Add vectors in parallel.
	cudaError_t cudaStatus = Gpu_R1byR1(&GridAndBlockCheck, cGPU, a, gridSize, blockSize, bTVol);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
	}

	PrintIntArray(a, 40, bTVol);

	PrintIntArray(cGPU, 40, bTVol);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK(cudaDeviceReset());
}

int main()
{
	TestGridAndBlock();
    return 0;
}


//__global__ void Matty(int *C) 
//{ 
//	int i = threadIdx.x;
//	C[i] = i; 
//} 
//
//int main() 
//{
//	const int N = 5;
//	int numBlocks = 1;
//	int C[N] = { 0,1,2,3,4 };
//	dim3 threadsPerBlock(N); 
//	Matty <<<numBlocks, threadsPerBlock>>>(C);
//	PrintIntArray(C, N, N);
//}



void AddGpuByBlock(int *dev_c, int *dev_a, int *dev_b, dim3 *gridSize, dim3 *blockSize, int arrayLen)
{
	addKernel_B<<<*gridSize, *blockSize >>>(dev_c, dev_a, dev_b, arrayLen);
}

void AddGpuByThread(int *dev_c, int *dev_a, int *dev_b, dim3 *gridSize, dim3 *blockSize, int arrayLen)
{
	addKernel_BT<<<*gridSize, *blockSize >>>(dev_c, dev_a, dev_b, arrayLen);
}


void TestR2byR1()
{
	dim3 *gridSize = dim3Unit(); // dim3Ctr(3, 2);
	dim3 *blockSize = dim3Ctr(3, 2);
	int bTVol = dim3Vol(gridSize) * dim3Vol(blockSize);

	int *a = IntArray(bTVol, 0, 1);
	int *b = IntArray(bTVol, 0, 10);
	int *cGPU = IntArray(bTVol, 0, 0);

	// Add vectors in parallel.
	cudaError_t cudaStatus = Gpu_R2byR1(&AddGpuByThread, cGPU, a, b, gridSize, blockSize, bTVol);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
	}

	int *cCPU = Cpu_R2byR1(gridSize, a, b);
	CompIntArrays(cCPU, cGPU, bTVol) ? printf("pass\n") : printf("fail\n");
	PrintIntArray(cGPU, 10, 30);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	CHECK(cudaDeviceReset());
}