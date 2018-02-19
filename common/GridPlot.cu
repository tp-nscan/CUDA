#include <stdlib.h>
#include <stdio.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "GridPlot.h"

void MakeGridPlot(GridPlot **out, int zoom, int imageWidth, 
	int xStart, int yStart)
{
	GridPlot *gridPlot = (GridPlot *)malloc(sizeof(GridPlot));
	gridPlot->zoom = zoom;
	gridPlot->imageWidth = imageWidth;
	gridPlot->xStart = xStart;
	gridPlot->yStart = yStart;
	*out = gridPlot;
}


__device__ unsigned char colorValue(float n1, float n2, int hue) {
	if (hue > 360)      hue -= 360;
	else if (hue < 0)   hue += 360;

	if (hue < 60)
		return (unsigned char)(255 * (n1 + (n2 - n1)*hue / 60));
	if (hue < 180)
		return (unsigned char)(255 * n2);
	if (hue < 240)
		return (unsigned char)(255 * (n1 + (n2 - n1)*(240 - hue) / 60));
	return (unsigned char)(255 * n1);
}