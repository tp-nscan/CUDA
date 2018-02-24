#pragma once
#include <curand_kernel.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct CricketCmdArgs {
	int plywidth;
	int imageWidth;
	int zoom;
	int threadChop;
	int xStart;
	int yStart;
	int reps;
	int seed;
	float speed;
	float noise;
};