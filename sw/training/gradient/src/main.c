#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "snrt.h"
#include "gradient.h"
#include "data.h"

//#define SINGLE_CORE

int main(int argc, char *argv[]) {
    #ifdef SINGLE_CORE
    if (snrt_cluster_core_idx() == 0){
        backpropagation_one_core(I,W,B,E,e,M,N,K);
        snrt_fpu_fence();

    }
    #else
    // Allocate space in TCDM
    uint32_t size_i = M * K * sizeof(DATA_TYPE);
    uint32_t size_w = K * N * sizeof(DATA_TYPE);
    uint32_t size_b = M * N * sizeof(DATA_TYPE);
    uint32_t size_e = M * N * sizeof(DATA_TYPE);


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

    // Copy data in TCDM
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_I, remote_I, size_i);
        snrt_dma_start_1d(local_W, remote_W, size_w);
        snrt_dma_start_1d(local_B, remote_B, size_b);
        snrt_dma_start_1d(local_E, remote_E, size_e);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();


    if(!snrt_is_dm_core()){
        backpropagation_baseline_multicore((double*)local_I,(double*)local_W,(double*)local_B,(double*)local_E,e,M,N,K);
        //backpropagation_multicore((double*)local_I,(double*)local_W,(double*)local_B,(double*)local_E,e,M,N,K);
        snrt_fpu_fence();
    }

    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(remote_W, local_W, size_w);
        snrt_dma_start_1d(remote_B, local_B, size_b);
        snrt_dma_wait_all();
    }
    #endif
}   
