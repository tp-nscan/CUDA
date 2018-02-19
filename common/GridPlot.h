#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct GridPlot
{
	unsigned int				 zoom;
	unsigned int				 imageWidth;
	unsigned int              xStart;
	unsigned int              yStart;
};

void MakeGridPlot(GridPlot **out, int zoom, int imageWidth, int xStart, int yStart);

__device__ unsigned char colorValue(float n1, float n2, int hue);