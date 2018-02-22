#pragma once
#include "cuda_runtime.h"
#include <stdlib.h>


//#ifndef DIMSTUFF_H_   /* Include guard */
//#define DIMSTUFF_H_



///**********/
///* iDivUp */
///**********/
//int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

///***********************/
///* CUDA ERROR CHECKING */
///***********************/
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
//{
//	if (code != cudaSuccess)
//	{
//		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//		if (abort) exit(code);
//	}
//}


cudaError_t Gpu_R1byR1(void(*f)(int *c, int *a, dim3 *gs, dim3 *bs, int length),
	int *c, int *a, dim3 *gridSize, dim3 *blockSize, int length);
	int *Cpu_R2byR1(dim3 *gridSize, int *av, int *bv);

cudaError_t Gpu_R2byR1(void (*f)(int *c, int *a, int *b, dim3 *gs, dim3 *bs, int length),
	int *c, int *a, int *b, dim3 *gridSize, dim3 *blockSize, int length);

cudaError_t Gpu_GetFloats(float **host_floats, float *dev_floats, int float_ct);

//#endif // DIMSTUFF_H_
