#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "Utils.h"
#include "BlockUtils.h"
#include "rando.h"
#include "Ply.h"
#include "TexPly.h"
#include "RandPly.h"
#include "Ant.h"


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	int wonk = IntNamed(argc, argv, "plywidth", 77);
	wonk = IntNamed(argc, argv, "dsa", 77);

	RunRandPlyBench2(argc, argv);
	return 0;
}
