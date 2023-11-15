#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "snrt.h"
#include "gradient.h"
#include "data.h"

//#define SINGLE_CORE

//TODO: modify such that dim of tiling is not fixed. 
//corner cases. At the moment it works only with square matrixes (k==n)
//it is still missing the check on dimension to decide oh much to bring on TCDM

int main(int argc, char *argv[]) {
    #ifdef SINGLE_CORE
    if (snrt_cluster_core_idx() == 0){
        backpropagation_one_core(I,W,B,E,e,M,N,K);
        snrt_fpu_fence();

    }
    #else
    // Allocate space in TCDM
    // uint32_t size_i = M * K * sizeof(DATA_TYPE);
    // uint32_t size_w = K * N * sizeof(DATA_TYPE);
    // uint32_t size_b = M * N * sizeof(DATA_TYPE);
    // uint32_t size_e = M * N * sizeof(DATA_TYPE);

    uint32_t fix_tiling_size = K/2;//TODO: udjust 

    uint32_t size_i = fix_tiling_size * sizeof(DATA_TYPE);//shoud use sqrt, but it is 8*8 at the moment
    uint32_t size_w = fix_tiling_size*fix_tiling_size * sizeof(DATA_TYPE);
    uint32_t size_b = fix_tiling_size * sizeof(DATA_TYPE);
    uint32_t size_e = fix_tiling_size * sizeof(DATA_TYPE);


    void *local_I, *local_W, *local_B, *local_E;
    void *remote_B,*remote_I,*remote_E,*remote_W;
    local_I = (void *)snrt_l1_next();
    local_W = local_I + size_i;
    local_B = local_W + size_w;
    local_E = local_B + size_b;


    remote_B = B;
    remote_I= I;
    remote_W = W;
    remote_E = E;

    for(int i=0;i<K/fix_tiling_size;i++){
        for(int j=0;j<N/fix_tiling_size;j++){
            // Copy data in TCDM
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(local_I, remote_I+i*size_i, size_i);
                for(int z=0;z<fix_tiling_size;z++){
                    snrt_dma_start_1d(local_W + size_w/fix_tiling_size*z,
                     remote_W + (z*N)*sizeof(DATA_TYPE)+j*size_w/fix_tiling_size + i*fix_tiling_size*N*sizeof(DATA_TYPE),
                     size_w/fix_tiling_size);
                }
                snrt_dma_start_1d(local_B, remote_B+j*size_b, size_b);
                snrt_dma_start_1d(local_E, remote_E+j*size_e, size_e);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();


            if(!snrt_is_dm_core()){
        //            snrt_mcycle();
                backpropagation_multicore((double*)local_I,(double*)local_W,(double*)local_B,(double*)local_E,e,M,fix_tiling_size,fix_tiling_size,i,j);
                snrt_fpu_fence();
        //            snrt_mcycle();
            }else{
                snrt_cluster_hw_barrier();//there is a barrier inside the function of the cluster. DMA core needs to adjust to that too
            }

            snrt_cluster_hw_barrier();


            if (snrt_is_dm_core()) {
                for(int z=0;z<fix_tiling_size;z++){
                    snrt_dma_start_1d(remote_W + (z*N)*sizeof(DATA_TYPE)+j*size_w/fix_tiling_size+ i*fix_tiling_size*N*sizeof(DATA_TYPE),
                     local_W + size_w/fix_tiling_size*z,
                     size_w/fix_tiling_size);
                }
                if(i==j)
                    snrt_dma_start_1d(remote_B+j*size_b, local_B, size_b);
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