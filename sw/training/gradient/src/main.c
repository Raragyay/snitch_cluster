#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "snrt.h"
#include "gradient.h"
#include "data.h"

//#define SINGLE_CORE

#define MAX_DIM 100000.0f / sizeof(DATA_TYPE)

uint32_t n_iter[2]__attribute__ ((aligned (4096)));
uint32_t fix_size_k;
uint32_t fix_size_n;

int main(int argc, char *argv[]) {
    #ifdef SINGLE_CORE
    if (snrt_cluster_core_idx() == 0){
        backpropagation_one_core(I,W,B,E,e,M,N,K);
        snrt_fpu_fence();

    }
    #else

    float div;
    uint32_t fix_tiling_size_k,new_tiling_size_k;
    uint32_t fix_tiling_size_n, new_tiling_size_n; 
    uint32_t size_i, size_w, size_b, size_e;
    void *local_I, *local_W, *local_B, *local_E;
    void *remote_B,*remote_I,*remote_E,*remote_W;
    uint32_t ub_i, ub_j,i ,j;

    //find best values for fix sizes of tiling
    if(K>=N){
        div =(float) (K/N); //K=div*N

        //K*N+k+n<=MAX_DIM ---> div*N*N+(div+1)*N-MAX_DIM == 0 ----> N =(-(div+1)+sqrt((div+1)*(div+1)-4*div*(-MAX_DIM))) / 2*div
        fix_tiling_size_n = ((uint32_t)(-(div+1.0f)+(float)sqrtf((div+1.0f)*(div+1.0f)-4.0f*div*(-MAX_DIM)))/(uint32_t)(2*div));

        fix_tiling_size_k = fix_tiling_size_n * (uint32_t)div;
        
    }else{
        div =(float) (N/K); 

        fix_tiling_size_k = ((uint32_t)(-(div+1.0f)+(float)sqrtf((div+1.0f)*(div+1.0f)-4.0f*div*(-MAX_DIM)))/(uint32_t)(2*div));

        fix_tiling_size_n = fix_tiling_size_k * (uint32_t)div;

    }

    size_i = fix_tiling_size_k * sizeof(DATA_TYPE);
    size_w = fix_tiling_size_k * fix_tiling_size_n * sizeof(DATA_TYPE);
    size_b = fix_tiling_size_n * sizeof(DATA_TYPE);
    size_e = fix_tiling_size_n * sizeof(DATA_TYPE);

    local_I = (void *)snrt_l1_next();
    local_W = local_I + size_i;
    local_B = local_W + size_w;
    local_E = local_B + size_b;


    remote_B = B;
    remote_I = I;
    remote_W = W;
    remote_E = E;

    //the following values dont work if division is perfect number. Fix with round

    ub_i =(K%fix_tiling_size_k==0)? K/fix_tiling_size_k : K/fix_tiling_size_k+1;
    ub_j =(N%fix_tiling_size_n==0)? N/fix_tiling_size_n : N/fix_tiling_size_n+1;
  
    if(snrt_cluster_core_idx()==0){
        n_iter[0]=ub_i;
        n_iter[1]= ub_j;
        fix_size_k = fix_tiling_size_k;
        fix_size_n = fix_tiling_size_n;
        printf("Number of iterations is: %u\n",ub_i*ub_j);
    }

    for(i=0;i<ub_i;i++){
        new_tiling_size_k = (i!=ub_i-1) ? fix_tiling_size_k : (K-i*fix_tiling_size_k);
        if(snrt_cluster_core_idx()==0)
            printf("Iteration number %u out of %u\n",i*ub_j,ub_i*ub_j);
        for(j=0;j<ub_j;j++){
        //if(j%4==0 && snrt_cluster_core_idx()==0)
            //printf("Iteration number %u out of %u\n",j,ub_i*ub_j);
            //all the cycles but the last one  with fixed size
            new_tiling_size_n = (j!=ub_j-1) ? fix_tiling_size_n : (N-j*fix_tiling_size_n);
        

            // Copy data in TCDM
            if (snrt_is_dm_core()) {

                if(j==0){
                    snrt_dma_start_1d(local_I, remote_I+i*fix_tiling_size_k*sizeof(DATA_TYPE), new_tiling_size_k*sizeof(DATA_TYPE));
                }

                for(int z=0;z<new_tiling_size_k;z++){
                    snrt_dma_start_1d(local_W + z*new_tiling_size_n*sizeof(DATA_TYPE),
                     remote_W + j*fix_tiling_size_n*sizeof(DATA_TYPE)+ i*N*fix_tiling_size_k*sizeof(DATA_TYPE) + z*N*sizeof(DATA_TYPE),
                     new_tiling_size_n*sizeof(DATA_TYPE));
                }

                if(i==0){
                snrt_dma_start_1d(local_B, remote_B+j*fix_tiling_size_n*sizeof(DATA_TYPE), new_tiling_size_n*sizeof(DATA_TYPE));
                }

                snrt_dma_start_1d(local_E, remote_E+j*fix_tiling_size_n*sizeof(DATA_TYPE), new_tiling_size_n*sizeof(DATA_TYPE));

                snrt_dma_wait_all();
            }
        
            snrt_cluster_hw_barrier();


            if(!snrt_is_dm_core()){
        //            snrt_mcycle();
                backpropagation_multicore((double*)local_I,(double*)local_W,(double*)local_B,(double*)local_E,e,M,new_tiling_size_n,new_tiling_size_k,i==0);
                snrt_fpu_fence();
        //            snrt_mcycle();
            }
         
            snrt_cluster_hw_barrier();
            

            if (snrt_is_dm_core()) {

                for(int z=0;z<new_tiling_size_k;z++){
                    snrt_dma_start_1d(remote_W + j*fix_tiling_size_n*sizeof(DATA_TYPE) + i*N*fix_tiling_size_k*sizeof(DATA_TYPE)+ z*N*sizeof(DATA_TYPE),
                     local_W + z*new_tiling_size_n*sizeof(DATA_TYPE),
                     new_tiling_size_n*sizeof(DATA_TYPE));
                }

                if(i==0){
                    snrt_dma_start_1d(remote_B+j*fix_tiling_size_n*sizeof(DATA_TYPE), local_B, new_tiling_size_n*sizeof(DATA_TYPE));
                }

                snrt_dma_wait_all();
            }

            snrt_cluster_hw_barrier();
        }
    }
    #endif
}   



        // while(1){
        //     asm volatile(
        //     "add a0, a0, %[zero]\n"
        //     :
        //     :[ zero ] "r"(0)
        //     : "a0");
        // }
