#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cpu_anim2.h"

struct GridPlot
{
	unsigned int	zoom;
	int2            dataSize;
	int2			dataUpleft;
};

struct GridPlotAppBlock {
	cudaEvent_t      start, stop;
	float            totalTime;
	float           *host_Data;
	float           *dev_Data;
	GridPlot		*gridPlot;
};


void gridPlot_anim_exit(ControlBlock *controlBlock);

void gridPlot_anim_gpu(ControlBlock *controlBlock, int ticks);

void MakeGridPlotAppBlock(GridPlotAppBlock **out, GridPlot *gridPlot, float *host_data);

void MakeGridPlot(GridPlot **out, int zoom, int2 dataSize, int2 dataUpLeft);

__device__ unsigned char colorValue(float n1, float n2, int hue);