#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "inc/gradient.h"
#include "inc/main.h"

int main(int argc, char *argv[]) {
    //already initialize the space for the gradients, we can do it inside the function since it is not needed outside
    DATA_TYPE grad_B[M*N];
    DATA_TYPE grad_W[N*K];
    backpropagation(I,W,B,grad_W,grad_B,E,e,M,N,K);

    for(int i=0;i<N;i++){
        for(int j=0;j<K;j++){
            printf("%.2f\t",W[i*N+j]);
        }
        printf("\n");
    }
    printf("\n");
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            printf("%.2f\t",B[i*M+j]);
        }
        printf("\n");
    }
}   
