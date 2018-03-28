#pragma once
#include "cuda_runtime.h"
#include <stdlib.h>
int IntNamed(int argc, char ** argv, const char *name, int default);
float FloatNamed(int argc, char ** argv, const char *name, float default);
int MaxInt(int a, int b);
int MinInt(int a, int b);
float MaxFloat(float a, float b);
float MinFloat(float a, float b);
float *FloatArray(int length);
int *IntArray(int length, int first = 0, int step = 0);
float *DotArray(int span, int offset);
float *BackSlashArray(int span, float minval, float maxval);
bool CompIntArrays(int *a, int *b, int length);
bool CompFloatArrays(float *a, float *b, int length);
void PrintFloatArray(float *aa, int width, int length);
void PrintIntArray(int *aa, int width, int length);
float *RndFloat0to1(int arraySize);