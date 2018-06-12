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

//10	1024
//11	2048
//12	4096
//13	8192
//14	16384
//15	32768
//16	65536
//17	131072
//18	262144
//19	524288
//20	1048576
//21	2097152
//22	4194304

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	int wonk = IntNamed(argc, argv, "plywidth", 77);
	wonk = IntNamed(argc, argv, "dsa", 77);

	RunRandPlyBench(argc, argv);
	return 0;
}
