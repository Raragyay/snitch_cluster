#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "snrt.h"
#include "gradient.h"
#include "data.h"

int main(int argc, char *argv[]) {
    if (snrt_cluster_core_idx() == 0){
        backpropagation(I,W,B,E,e,M,N,K);

        // for(int i=0;i<K;i++){
        //     for(int j=0;j<N;j++){
        //         printf("%f,",W[i*N+j]);
        //     }
        // }
        // printf("\n\n");
        // for(int i=0;i<M;i++){
        //     for(int j=0;j<N;j++){
        //         printf("%f,",B[i*N+j]);
        //     }
        // }
        // printf("\n\n");
    }
}   
