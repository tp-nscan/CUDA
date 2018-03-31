#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <helper_functions.h>


int IntNamed(int argc, char ** argv, const char *name, int default)
{
	if (checkCmdLineFlag((const int)argc, (const char **)argv, (const char *)name))
	{
		return getCmdLineArgumentInt((const int)argc, (const char **)argv, (const char *)name);
	}
	return default;
}

float FloatNamed(int argc, char ** argv, const char *name, float default)
{
	if (checkCmdLineFlag((const int)argc, (const char **)argv, (const char *)name))
	{
		return getCmdLineArgumentFloat((const int)argc, (const char **)argv, (const char *)name);
	}
	return default;
}

int MaxInt(int a, int b)
{
	if (a > b) return a;
	return b;
}

int MinInt(int a, int b)
{
	if (a < b) return a;
	return b;
}

float MaxFloat(float a, float b)
{
	if (a > b) return a;
	return b;
}

float MinFloat(float a, float b)
{
	if (a < b) return a;
	return b;
}

float FloatArraySum(float *array, int length)
{
	float cume = 0;
	for (int i = 0; i < length; i++)
	{
		cume += array[i];
	}
	return cume;
}

float *FloatArray(int length)
{
	float *av = (float *)malloc(sizeof(float) * length);
	for (int i = 0; i < length; i++)
	{
		av[i] = 0.0;
	}
	return av;
}

int *IntArray(int length, int first = 0, int step = 0)
{
	int *av = (int *)malloc(sizeof(int) * length);
	for (int i = 0; i < length; i++)
	{
		av[i] = first + step * i;
	}
	return av;
}

float *BackSlashArray(int span, float minval, float maxval)
{
	float* av = new float[span*span];
	float step = (maxval - minval) / (float)span;
	float fi, fj;
	for (int i = 0; i < span; i++)
	{
		fi = minval + i * step;
		for (int j = 0; j < span; j++)
		{
			fj = minval + (j * 1.5) * step;
			av[i + j * span] = MinFloat(fi, fj);
		}
	}
	return av;
}

float *DotArray(int span, int offset)
{
	int area = span * span;
	float *av = (float *)malloc(sizeof(float) * area);
	for (int i = 0; i < span; i++)
	{
		for (int j = 0; j < span; j++)
		{
			av[i + j * span] = 0.0;
		}
	}
	av[((span/2) * (span + 1) + offset) % area] = 1.0;
	return av;
}

float *DashArray(int span, int offset)
{
	int area = span * span;
	float *av = (float *)malloc(sizeof(float) * area);
	for (int i = 0; i < span; i++)
	{
		for (int j = 0; j < span; j++)
		{
			av[i + j * span] = 0.0;
		}
	}
	av[((span / 2) * (span + 1) + offset) % area] = 1.0;
	av[((span / 2) * (span + 1) + offset + 1) % area] = 1.0;
	return av;
}

float *CheckerArray(int span)
{
	int area = span * span;
	float *av = (float *)malloc(sizeof(float) * area);
	for (int i = 0; i < span; i++)
	{
		for (int j = 0; j < span; j++)
		{
			av[i + j * span] = (float)(((i - j + 4096) % 2) *2 - 1);
		}
	}
	return av;
}

bool CompIntArrays(int *a, int *b, int length)
{
	for (int i = 0; i < length; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}

bool CompFloatArrays(float *a, float *b, int length)
{
	for (int i = 0; i < length; i++)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}

void PrintFloatArray(float *aa, int width, int length)
{
	for (int i = 0; i < length; i++) {
		printf("%3.3f ", aa[i]);
		if ((i>0) && ((i + 1) % width == 0)) printf("\n");
	}
	printf("\n");
}

void PrintIntArray(int *aa, int width, int length)
{
	for (int i = 0; i < length; i++) {
		printf("%d ", aa[i]);
		if ((i>0) && ((i + 1) % width == 0)) printf("\n");
	}
	printf("\n");
}

float *RndFloat0to1(int arraySize)
{
	float *temp = (float*)malloc(arraySize * sizeof(float));
	for (int i = 0; i<arraySize; i++) {
		temp[i] = (float)rand() / (float)(RAND_MAX);
	}
	return temp;
}
