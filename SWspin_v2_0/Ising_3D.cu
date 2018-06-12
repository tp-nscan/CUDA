/******************************************************************************/
//
//  GPU-based Swendsen-Wang multi-cluster algorithm for the simulation 
//  of classical spin systems
//
//    1. 2D Ising model
//    2. 2D q-state Potts model
//    3. 2D q-state clock model
//    4. 3D Ising model
//    5. 3D q-state Potts model
//    6. 3D q-state clock model
//
//  Copyright (C) 2015  Yukihiro Komura, Yutaka Okabe
//  RIKEN, Advanced Institute for Computational Science
//  Department of Physics, Tokyo Metropolitan University
//
//  This program is subject to the Standard CPC licence, 
//        (http://cpc.cs.qub.ac.uk/licence/licence.html)
//
//  Prerequisite: the NVIDIA CUDA Toolkit 5.0 or newer
//
//  Related publication:
//    Y. Komura and Y. Okabe, Comput. Phys. Commun. 183, 1155-1161 (2012).
//    doi: 10.1016/j.cpc.2012.01.017
//    Y. Komura,              Comput. Phys. Commun. 194, 54-58 (2015).
//    doi:10.1016/j.cpc.2015.04.015
//
//  We use the CUDA implementation for the cluster-labeling based on 
//  the work by Hawick et al., that by Kalentev et al., and that by Komura.
//    K.A. Hawick, A. Leist, and D. P. Playne, 
//      Parallel Computing 36, 655-678 (2010).
//    O. Kalentev, A. Rai, S. Kemnitzb, and R. Schneider,
//      J. Parallel Distrib. Comput. 71, 615-620 (2011).
//    Y. Komura,             
//      Comput. Phys. Commun. 194, 54-58 (2015).
//
//  We use the NVIDIA CUDA Random Number Generation library (cuRAND) 
//
/******************************************************************************/
//
//    4. 3D Ising model
//
//    model: Ising model on the square lattice
//           with periodic boundary conditions
//
//    system size: nx*nx*nx  (nx should be a multiple of 8;
//                 maximum size of nx depends on memory size)
//
//    choice of cluster-labeling algorithm:
//         Hawick et al.   - algorithm = 0
//         Kalentev et al. - algorithm = 1
//         Komura          - algorithm = 2
//    number of Monte Carlo steps per spin for discard:     mcs1
//    number of Monte Carlo steps per spin for measurement: mcs2
//
//    outputs:
//        2nd moment of magnetization
//        4th moment of magnetization
//        energy per spin
//        specific heat per spin
//        computational time for spin-flip  (milliseconds per a single MCS)
//        computational time for spin-flip + measurement
//                                          (milliseconds per a single MCS)
//
/******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <curand.h>

__global__ void device_function_spinset    (int*                        );
__global__ void device_function_init_H     (double, int*, double*, 
                                            unsigned int*, unsigned int*);
__global__ void device_function_init_K     (double, int*, double*, 
                                            unsigned int*               );
__global__ void device_function_init_YK    (double, int*, int*, double*, 
                                            unsigned int*               );
__global__ void device_function_scanning_H (unsigned int*, unsigned int*, 
                                            int*                        );
__global__ void device_function_scanning_K (unsigned int*, int*         );
__global__ void device_function_analysis_H (unsigned int*, unsigned int*);
__global__ void device_function_analysis_K (unsigned int*               );
__global__ void device_function_analysis_YK(unsigned int*               );
__global__ void device_function_labeling_H (unsigned int*, unsigned int*);
__global__ void device_ReduceLabels        (unsigned int*, int*         );
__device__ unsigned int root_find          (unsigned int*, int          );
__global__ void device_function_spin_select(int*, double*               );
__global__ void device_function_spin_flip  (int*, int*, unsigned int*   );
__global__ void device_function_spin_flip_YK(int*, int*, unsigned int*  );
__global__ void device_function_sum        (int*, int*, int*            );


#define nx 32   // linear system size

const int nla = nx*nx*nx;

int main(int argc,char** argv) {

/*------------ Device Init --------------------------------------------*/

    cudaSetDevice(0);


/*------------ Variables for GPU command ------------------------------*/

    unsigned threads, grid;
    threads = 512;
    grid    = nla/512;

    float flip_timer, total_timer;
    cudaEvent_t start_timer, stop_timer;
    cudaEventCreate( &start_timer );
    cudaEventCreate( &stop_timer  );

//    int algorithm = 0; // Hawick
//    int algorithm = 1; // Kalentev
    int algorithm = 2; // Komura

/*------------ Size of array ------------------------------------------*/

    unsigned int mem_N                 = sizeof(int)*(nla);
    unsigned int mem_rand              = sizeof(double)*(4*nla);
    unsigned int mem_1                 = sizeof(int)*(1);
    unsigned int mem_measured_quantity = sizeof(int)*(grid);
    unsigned int mem_measured_magnet   = sizeof(int)*(grid);


/*------------ Assignment of memory for host --------------------------*/

    int* h_flag        = (int*) malloc(mem_1);
    int* h_out_energy  = (int*) malloc(mem_measured_quantity);
    int* h_out_magnet  = (int*) malloc(mem_measured_magnet  );


/*------------ Assignment of memory for device ------------------------*/

    int* d_spin;
    int* d_spin_new;
    int* d_bond;
    unsigned int* d_label;
    unsigned int* d_R;
    double* d_random_data;
    int* d_flag;
    int* d_out_energy;
    int* d_out_magnet;

    cudaMalloc((void**) &d_spin       , mem_N   );
    cudaMalloc((void**) &d_spin_new   , mem_N   );
    cudaMalloc((void**) &d_bond       , mem_N   );
    cudaMalloc((void**) &d_label      , mem_N   );
    cudaMalloc((void**) &d_R          , mem_N   );
    cudaMalloc((void**) &d_random_data, mem_rand);
    cudaMalloc((void**) &d_flag       , mem_1   );
    cudaMalloc((void**) &d_out_energy , mem_measured_quantity);
    cudaMalloc((void**) &d_out_magnet , mem_measured_magnet  );


/*------------ Set of initial random numbers --------------------------*/

    int iri1 = 12345;

    curandGenerator_t gen;
//
//  You can choose several random number generations: 
//  CURAND_RNG_PSEUDO_XORWOW(=CURAND_RNG_PSEUDO_DEFAULT), 
//  CURAND_RNG_PSEUDO_MRG32K3A, CURAND_RNG_PSEUDO_MTGP32,
//  CURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_PSEUDO_MT19937
//
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    cudaDeviceSynchronize();
    curandSetPseudoRandomGeneratorSeed(gen, (long long)iri1);
    cudaDeviceSynchronize();

/*------------ Temperature and Monte Carlo steps ----------------------*/

    double t, beta;
    double boltz;

    const double tstart =  4.44, tunit = 0.02;
    const int    tnum = 8;

    const int mcs1 = 5000, mcs2 = 20000;


/*------------ Variables for measured quantities ----------------------*/

    double H, M, f2order, fm[5];
    double fm2, fm4, fe1, fe2, cv;

    double fnla2 = (double)nla*(double)nla;

/*------------ Start of GPU calculation -------------------------------*/

    printf("#\n");
    printf("#  GPU-based Swendsen-Wang multi-cluster algorithm ");
    printf("for 3D Ising model\n");
    if(algorithm==0){
      printf("#      (Hawick algorithm for cluster-labeling)\n");
    } else if(algorithm==1) {
      printf("#      (Kalentev algorithm for cluster-labeling)\n");
    } else if(algorithm==2){
      printf("#      (Komura algorithm for cluster-labeling)\n");    
    }
    printf("#\n");
    printf("#  Related publication:\n");
    printf("#    Y. Komura and Y. Okabe, Comput. Phys. Commun. ");
    printf("183, 1155-1161 (2012).\n");
    printf("#    doi: 10.1016/j.cpc.2012.01.017\n");
    printf("#    Y. Komura, Comput. Phys. Commun. ");
    printf("194, 54-58 (2015).\n");
    printf("#    doi: 10.1016/j.cpc.2015.04.015\n");
    printf("#\n");
    printf("#  nx= %d\n",nx);
    printf("#  T_start= %f  T_unit= %f  'No. of T'= %d\n", tstart, tunit, tnum);
    printf("#  mcs1(discard)= %d  mcs2(measure)= %d\n", mcs1, mcs2);
    printf("#  iri1(seed)= %d\n", iri1);
    printf("#\n");
    printf("#  T, m2, m4, energy, cv, flip_time(ms/MCS), total_time(ms/MCS)\n");
    printf("#\n");


    for(int itemp=1; itemp<=tnum; itemp++){
      t = tstart+tunit*(itemp-1);
      beta = 1.0/t;
      boltz = exp(-2*beta);

      cudaThreadSynchronize();
      cudaEventRecord( start_timer, 0 );

      device_function_spinset<<<grid,threads>>>(d_spin);


/*------------ Equilibration ------------------------------------------*/

      for(int mcs=1; mcs <= mcs1; mcs++){

        // Random number generation
        curandGenerateUniformDouble(gen, d_random_data, 4*nla);

        // Bond connection
        if(algorithm==0){
          device_function_init_H<<<grid,threads>>>
                  (boltz, d_spin, d_random_data, d_label, d_R);
        } else if(algorithm==1){
          device_function_init_K<<<grid,threads>>>
                  (boltz, d_spin, d_random_data, d_label);
        } else if(algorithm==2){
          device_function_init_YK<<<grid,threads>>>
                  (boltz, d_spin, d_bond, d_random_data, d_label);        
        }

        // Cluster formation
        if(algorithm<=1){
         h_flag[0] = 1;
         while(h_flag[0] != 0){
          h_flag[0] = 0;
          // Copy of "flag" from Host to Device
          cudaMemcpy(d_flag,h_flag,mem_1,cudaMemcpyHostToDevice); 
          if(algorithm==0){
            device_function_scanning_H<<<grid,threads>>>(d_label, d_R, d_flag);
            device_function_analysis_H<<<grid,threads>>>(d_label, d_R        );
            device_function_labeling_H<<<grid,threads>>>(d_label, d_R        );
          } else if(algorithm==1) {
            device_function_scanning_K<<<grid,threads>>>(d_label, d_flag);
            device_function_analysis_K<<<grid,threads>>>(d_label        );
          } 

          // Copy of "flag" from Device to Host
          cudaMemcpy(h_flag,d_flag,mem_1,cudaMemcpyDeviceToHost); 
         }
        }else if(algorithm==2){
            device_function_analysis_YK<<<grid,threads>>>(d_label);
            device_ReduceLabels<<<grid,threads>>>(d_label,d_bond);
            device_function_analysis_YK<<<grid,threads>>>(d_label);
        }

        // Spin flip process
        device_function_spin_select<<<grid,threads>>>(d_spin_new, 
                                                      d_random_data  );
        if(algorithm<=1){
         device_function_spin_flip  <<<grid,threads>>>(d_spin, d_spin_new, 
                                                       d_label);
        }else if(algorithm==2){
         device_function_spin_flip_YK<<<grid,threads>>>(d_spin, d_spin_new, 
                                                       d_label);        
        }
      } // "mcs1" loop end

      cudaEventRecord( stop_timer, 0 );
      cudaEventSynchronize( stop_timer );
      cudaEventElapsedTime( &flip_timer, start_timer, stop_timer );
      flip_timer = flip_timer/mcs1;

      cudaEventRecord( start_timer, 0 );

/*------------ Main calculation ---------------------------------------*/

      for(int i=1; i<=4; i++){
        fm[i]= 0;
      }

      for(int mcs=1; mcs <= mcs2; mcs++){

        // Random number generation
        curandGenerateUniformDouble(gen, d_random_data, 4*nla);

        // Bond connection
        if(algorithm==0){
          device_function_init_H<<<grid,threads>>>
                  (boltz, d_spin, d_random_data, d_label, d_R);
        } else if (algorithm==1) {
          device_function_init_K<<<grid,threads>>>
                  (boltz, d_spin, d_random_data, d_label);
        } else{
          device_function_init_YK<<<grid,threads>>>
                  (boltz, d_spin, d_bond, d_random_data, d_label);        
        }

        // Cluster formation
        if(algorithm<=1){
         h_flag[0] = 1;
         while(h_flag[0] != 0){
          h_flag[0] = 0;
          // Copy of "flag" from Host to Device
          cudaMemcpy(d_flag,h_flag,mem_1,cudaMemcpyHostToDevice); 
          if(algorithm==0){
            device_function_scanning_H<<<grid,threads>>>(d_label, d_R, d_flag);
            device_function_analysis_H<<<grid,threads>>>(d_label, d_R        );
            device_function_labeling_H<<<grid,threads>>>(d_label, d_R        );
          } else if(algorithm==1) {
            device_function_scanning_K<<<grid,threads>>>(d_label, d_flag);
            device_function_analysis_K<<<grid,threads>>>(d_label        );
          } 

          // Copy of "flag" from Device to Host
          cudaMemcpy(h_flag,d_flag,mem_1,cudaMemcpyDeviceToHost); 
         }
        }else if(algorithm==2){
            device_function_analysis_YK<<<grid,threads>>>(d_label);
            device_ReduceLabels<<<grid,threads>>>(d_label,d_bond);
            device_function_analysis_YK<<<grid,threads>>>(d_label);
        }

        // Spin flip process
        device_function_spin_select<<<grid,threads>>>(d_spin_new, 
                                                      d_random_data);
        if(algorithm<=1){
         device_function_spin_flip  <<<grid,threads>>>(d_spin, d_spin_new, 
                                                       d_label);
        }else if(algorithm==2){
         device_function_spin_flip_YK<<<grid,threads>>>(d_spin, d_spin_new, 
                                                       d_label);        
        }


/*------------ Measurement --------------------------------------------*/

        device_function_sum <<<grid,threads>>>(d_spin,d_out_energy,
                                               d_out_magnet);

        H = 0;
        M = 0;

        cudaMemcpy(h_out_energy,d_out_energy,mem_measured_quantity,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out_magnet,d_out_magnet,mem_measured_magnet,
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for(int i=0; i<grid; i++){
          H += h_out_energy[i];
          M += h_out_magnet[i];
        }
        f2order = M*M;

        fm[1] += f2order;
        fm[2] += f2order*f2order;
        fm[3] += H;
        fm[4] += H*H;

      } // "mcs2" loop end

      for(int i=1; i<=4; i++){
        fm[i] = fm[i]/mcs2;
      }

      fm2 = fm[1]/fnla2;
      fm4 = fm[2]/(fnla2*fnla2);
      fe1 = fm[3]/nla;
      fe2 = fm[4]/fnla2;
      cv  = beta*beta*(fe2-fe1*fe1)*nla;

      cudaEventRecord( stop_timer, 0 );
      cudaEventSynchronize( stop_timer );
      cudaEventElapsedTime( &total_timer, start_timer, stop_timer );
      total_timer = total_timer/mcs2;

      printf("%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n",
        t,fm2,fm4,fe1,cv,flip_timer,total_timer);

    } // "t" loop end


/*------------- Cleaning of memory ------------------------------------*/

    free(h_flag);
    free(h_out_energy);
    free(h_out_magnet);

    curandDestroyGenerator(gen);
    cudaEventDestroy( start_timer );
    cudaEventDestroy( stop_timer );
    cudaFree(d_spin);
    cudaFree(d_spin_new);
    cudaFree(d_bond);
    cudaFree(d_label);
    cudaFree(d_R);
    cudaFree(d_random_data);
    cudaFree(d_flag);
    cudaFree(d_out_energy);
    cudaFree(d_out_magnet);
}


/*****************************************************************************
        Device functions (GPU)
*****************************************************************************/


//****************************************************************************
__global__ void device_function_spinset(int* d_spin)
/*
     Initial spin set
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    d_spin[index] = 1;
}


//****************************************************************************
__global__ void device_function_init_H(double d_t, int* d_spin, 
                double* d_random_data, unsigned int* d_label, unsigned int* d_R)
/*
     Bond connection
        (Hawick algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    int la, i;
    int spin, bond;
    __shared__ double boltz;

    spin = d_spin[index];
    bond = 0;
    if(threadIdx.x==0){
      boltz = d_t;
    }
    __syncthreads();

/*------------ Bond connections with left and top sites ---------------*/

    for(i=0; i<3; i++){
     if(i==0)la = (index-1+nx)%nx+((int)(index/nx))*nx;
     if(i==1)la = (index-nx+nx*nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
     if(i==2)la = (index-nx*nx+nla)%nla;
     if( spin == d_spin[la] ){
       if( boltz < d_random_data[index+i*nla] ) {
         bond |=  0x01<<i;
       }
     }
    }

/*------------ Transfer to global memories ----------------------------*/

    // Transfer "label" and "R" to a global memory
    d_label[index] = (index<<3) | bond;
    d_R[index] = index;
}


//****************************************************************************
__global__ void device_function_init_K(double d_t, int* d_spin, 
                double* d_random_data, unsigned int* d_label)
/*
     Bond connection
        (Kalentev algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    int la, i;
    int spin, bond;
    __shared__ double boltz;

    spin = d_spin[index];
    bond = 0;
    if(threadIdx.x==0){
      boltz = d_t;
    }
    __syncthreads();

/*------------ Bond connections with left and top sites ---------------*/

    for(i=0; i<3; i++){
     if(i==0)la = (index-1+nx)%nx+((int)(index/nx))*nx;
     if(i==1)la = (index-nx+nx*nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
     if(i==2)la = (index-nx*nx+nla)%nla;
     if( spin == d_spin[la] ){
       if( boltz < d_random_data[index+i*nla] ) {
         bond |=  0x01<<i;
       }
     }
    }

/*------------ Transfer to global memories ----------------------------*/

    // Transfer "label" to a global memory
    d_label[index] = (index<<3) | bond;
}


//****************************************************************************
__global__ void device_function_init_YK(double d_t, int* d_spin, 
              int* d_bond, double* d_random_data, unsigned int* d_label)
/*
     Bond connection
        (Komura algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    int la, i, index_min;
    int spin, bond;
    __shared__ double boltz;

    spin = d_spin[index];
    bond = 0;
    index_min = index;
    if(threadIdx.x==0){
      boltz = d_t;
    }
    __syncthreads();

/*------------ Bond connections with left and top sites ---------------*/

    for(i=0; i<3; i++){
     if(i==0)la = (index-1+nx)%nx+((int)(index/nx))*nx;
     if(i==1)la = (index-nx+nx*nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
     if(i==2)la = (index-nx*nx+nla)%nla;
     if( spin == d_spin[la] ){
       if( boltz < d_random_data[index+i*nla] ) {
         bond |=  0x01<<i;
         index_min = min(index_min, la);
       }
     }
    }

/*------------ Transfer to global memories ----------------------------*/

    // Transfer "label" and "bond" to a global memory
    d_bond[index]  = bond;
    d_label[index] = index_min;
}


//****************************************************************************
__global__ void device_function_scanning_H(unsigned int* d_label, 
                unsigned int* d_R, int* d_flag)
/*
     Comparison of "label"s with nearest neighbor sites
        (Hawick algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int la, label_1, label_2, i;
    __shared__ unsigned int label;

    label = d_label[index];
    label_1 = label>>3;
    label_2 = nla+1;   // MAX_INT

/*------------ Comparison with "label"s of left and top sites -----------*/

    for(i=0; i<3; i++){
     if((label>>i)&1==1){
      if(i==0)la = (index-1+nx)%nx+((int)(index/nx))*nx;
      if(i==1)la = (index-nx+nx*nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
      if(i==2)la = (index-nx*nx+nla)%nla;
      label_2 = min((d_label[la]>>3),label_2);
     }
    }

/*------------ Comparison with "label"s of right and bottom sites -------*/

    for(i=0; i<3; i++){
     if(i==0) la = (index+1)%nx+((int)(index/nx))*nx;
     if(i==1) la = (index+nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
     if(i==2) la = (index+nx*nx)%nla;
     if((d_label[la]>>i)&1==1){
      label_2 = min((d_label[la]>>3),label_2);
     }
    }

/*------------ Comparison between old and new "label"s ----------------*/

    if(label_2 < label_1){
      atomicMin( &d_R[label_1], label_2); 
            // Atomic operations are performed without interence 
            // with any other threads. 
      d_flag[0] = 1;
    }
}


//****************************************************************************
__global__ void device_function_scanning_K(unsigned int* d_label, 
                int* d_flag)
/*
     Comparison of "label"s with nearest neighbor sites
        (Kalentev algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int la, label_1, label_2, label_3, i;
    unsigned int t_label,label;

    label = d_label[index];
    label_1 = (label>>3);
    label_2 = nla+1;   // MAX_INT

    __syncthreads();

/*------------ Comparison with "label"s of left and top sites -----------*/

    for(i=0; i<3; i++){
     if((label>>i)&1==1){
      if(i==0)la = (index-1+nx)%nx+((int)(index/nx))*nx;
      if(i==1)la = (index-nx+nx*nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
      if(i==2)la = (index-nx*nx+nla)%nla;
      label_2 = min((d_label[la]>>3),label_2);
     }
    }

/*------------ Comparison with "label"s of right and bottom sites -------*/

    for(i=0; i<3; i++){
     if(i==0) la = (index+1)%nx+((int)(index/nx))*nx;
     if(i==1) la = (index+nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
     if(i==2) la = (index+nx*nx)%nla;
     if((d_label[la]>>i)&1==1){
      label_2 = min((d_label[la]>>3),label_2);
     }
    }

/*------------ Comparison between old and new "label"s ----------------*/

    if(label_2 < label_1){
      t_label = d_label[label_1];
      label_3 = (t_label>>3);
      label_3 = min(label_3,label_2);
      d_label[label_1] = (t_label&0x07) | ( label_3<<3 );
      d_flag[0] = 1;
    }
}


//****************************************************************************
__global__ void device_function_analysis_H(unsigned int* d_label, 
                unsigned int* d_R)
/*
     Equivalence chain of "R": repeat the calculation up to the point 
           such that d_R[d_R[...[index]...]] = d_R[...[index]...];
        (Hawick algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ref, label;

    if( (d_label[index]>>3)==index ){
      label = index;
      ref = d_R[label];

      while( ref != label){
        label = ref;
        ref = d_R[label];
      }
      d_R[index] = ref;
    }
}


//****************************************************************************
__global__ void device_function_analysis_K(unsigned int* d_label)
/*
     Equivalence chain of "label": repeat the calculation up to the point 
           such that d_label[d_label[...[index]...]] = d_label[...[index]...];
        (Kalentev algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ref, t_label, bond;

    bond = d_label[index];
    t_label = index;
    ref = (d_label[t_label]>>3);

    while( ref != t_label){
      t_label = ref;
      ref = (d_label[t_label]>>3);
    }

    d_label[index] = (bond&0x07) | ( ref<<3 );
}


//****************************************************************************
__global__ void  device_function_labeling_H(unsigned int* d_label, 
                unsigned int* d_R)
/*
     Update of "label" from "R"
        (Hawick algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int label;

    label = d_label[index];
    d_label[index] = (label&0x07) |(d_R[label>>3] <<3);
}


//****************************************************************************
__global__ void device_function_analysis_YK(unsigned int* d_label)
/*
     Equivalence chain of "label": repeat the calculation up to the point 
           such that d_label[d_label[...[index]...]] = d_label[...[index]...];
        (Komura algorithm)
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    d_label[index] = root_find(d_label, index);
}


//****************************************************************************
__global__ void device_ReduceLabels(unsigned int* d_label, 
                                                    int* d_bond )
/*
     Label reduction method. 
     Please see the pseudo-code of Algorithm 1 in 
     Comput. Phys. Commun. 194, 54-58 (2015).
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int la, i;
    unsigned int label_1, label_2, label_3, flag;
    unsigned int bond;

    bond = d_bond[index];

    __syncthreads();

/*------------ Comparison with "label" of left and y-top sites ------------*/
    
    for(i=0; i<2; i++){
     flag = 0;

     label_1 = root_find(d_label, index);
     if( (bond>>i)&0x01 == 1 ){
      if(i==0)la = (index-1+nx)%nx+((int)(index/nx))*nx;
      if(i==1)la = (index-nx+nx*nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
      label_2 = root_find(d_label, la);
      if(label_1!=label_2)flag = 1;
     }

     if(label_1 < label_2){ 
      label_3=label_1; 
      label_1=label_2; 
      label_2=label_3; 
     }
     while( flag == 1 ){
      label_3 = atomicMin(&d_label[label_1], label_2);
      if(label_3==label_2){ flag = 0;        }
      else if(label_3>label_2){ label_1=label_3;                 }
      else if(label_3<label_2){ label_1=label_2; label_2=label_3;}
     }
    }

/*------------ Comparison with "label" of z-top boudanry site ---------*/

   if(index< (nx*nx)){
     flag = 0;

     label_1 = root_find(d_label, index);
     if( (bond>>2)&0x01 == 1 ){
      la = (index-nx*nx+nla)%nla;
      label_2 = root_find(d_label, la);
      if(label_1!=label_2)flag = 1;
     }

     if(label_1 < label_2){ 
      label_3=label_1; 
      label_1=label_2; 
      label_2=label_3; 
     }
     while( flag == 1 ){
      label_3 = atomicMin(&d_label[label_1], label_2);
      if(label_3==label_2){ flag = 0;        }
      else if(label_3>label_2){ label_1=label_3;                 }
      else if(label_3<label_2){ label_1=label_2; label_2=label_3;}
     }
   }
}

//****************************************************************************
__device__ unsigned int  root_find(
  unsigned int* d_label, int index )
/*
     Equivalence chain of "label": repeat the calculation up to the point 
           such that d_label[d_label[...[index]...]] = d_label[...[index]...];
*/
{

    unsigned int ref, t_label;

    t_label = index;
    ref = d_label[t_label];

    while( ref != t_label){
      t_label = ref;
      ref = d_label[t_label];
    }

  return ref;
}


//****************************************************************************
__global__ void device_function_spin_select(int* d_spin_new, 
                double* d_random_data)
/*
     Set a new spin state for each "label"
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    int spin_new;


    if( 0.5 < d_random_data[index+3*nla] ) {
      spin_new = 1;
    } else {
      spin_new = -1;
    }
    d_spin_new[index] = spin_new;
}


//****************************************************************************
__global__ void device_function_spin_flip(int* d_spin, int* d_spin_new, 
                unsigned int* d_label)
/*
     Set a new spin state for each site with chosen "label"
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    d_spin[index] = d_spin_new[ ( d_label[index]>>3 ) ];
}

//****************************************************************************
__global__ void device_function_spin_flip_YK(int* d_spin, int* d_spin_new, 
                unsigned int* d_label)
/*
     Set a new spin state for each site with chosen "label"
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    d_spin[index] = d_spin_new[ ( d_label[index] ) ];
}


//****************************************************************************
__global__ void device_function_sum(int* d_spin, int* out_energy, 
                int* out_magnet)
/*
     Functions for measuring physical quantities
*/
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int la[3];
    int S;
    volatile __shared__ int dH[512], dM[512];
    unsigned int dx;

/*------------ Calculation of energy ----------------------------------*/

    la[0] = (index+1)%nx + ((int)(index/nx))*nx;
    la[1] = (index+nx)%(nx*nx) + (int)(index/(nx*nx))*nx*nx;
    la[2] = (index+nx*nx)%nla;

    S = d_spin[index];

    dH[threadIdx.x] = -S*
                      (d_spin[la[0]]+d_spin[la[1]]+d_spin[la[2]]);
    __syncthreads();

    //reduction
    dx = 512;

    while(dx > 64){
      dx = dx/2;

      if(threadIdx.x<dx) {
        dH[threadIdx.x] += dH[threadIdx.x+dx];
      }
      __syncthreads();
    }

    if(threadIdx.x<32) {
      dH[threadIdx.x] += dH[threadIdx.x+32];
      dH[threadIdx.x] += dH[threadIdx.x+16];
      dH[threadIdx.x] += dH[threadIdx.x+8];
      dH[threadIdx.x] += dH[threadIdx.x+4];
      dH[threadIdx.x] += dH[threadIdx.x+2];
      dH[threadIdx.x] += dH[threadIdx.x+1];
    }

    if(threadIdx.x==0){ 
      out_energy[blockIdx.x] = dH[0];  
    }

    __syncthreads();

/*------------ Calculation of magnetization ---------------------------*/


    dM[threadIdx.x] = S;
    __syncthreads();

    //reduction
    dx = 512;

    while(dx > 64){
      dx = dx/2;

      if(threadIdx.x<dx) {
        dM[threadIdx.x] += dM[threadIdx.x+dx];
      }
      __syncthreads();
    }

    if(threadIdx.x<32) {
      dM[threadIdx.x] += dM[threadIdx.x+32];
      dM[threadIdx.x] += dM[threadIdx.x+16];
      dM[threadIdx.x] += dM[threadIdx.x+8 ];
      dM[threadIdx.x] += dM[threadIdx.x+4 ];
      dM[threadIdx.x] += dM[threadIdx.x+2 ];
      dM[threadIdx.x] += dM[threadIdx.x+1 ];
    }

    if(threadIdx.x==0){
      out_magnet[blockIdx.x] = dM[0];
    }
    __syncthreads();
}
