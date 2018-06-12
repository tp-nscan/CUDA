all: Ising_2D Ising_3D Potts_2D Potts_3D clock_2D clock_3D

Ising_2D: Ising_2D.cu
	nvcc $+ -o $@.out -arch=sm_35 -lcurand

Potts_2D: Potts_2D.cu
	nvcc $+ -o $@.out -arch=sm_35 -lcurand

clock_2D: clock_2D.cu
	nvcc $+ -o $@.out -arch=sm_35 -lcurand

Ising_3D: Ising_3D.cu
	nvcc $+ -o $@.out -arch=sm_35 -lcurand

Potts_3D: Potts_3D.cu
	nvcc $+ -o $@.out -arch=sm_35 -lcurand

clock_3D: clock_3D.cu
	nvcc $+ -o $@.out -arch=sm_35 -lcurand

clean:
	rm -f *.out

