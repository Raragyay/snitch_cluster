#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "gradient.h"
#include "main.h"

int main(int argc, char *argv[]) {
    //already initialize the space for the gradients, we can do it inside the function since it is not needed outside
    backpropagation(I,W,B,E,e,M,N,K);

    for(int i=0;i<K;i++){
        for(int j=0;j<N;j++){
            printf("%f,",W[i*N+j]);
        }
    }
    printf("\n\n");
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            printf("%f,",B[i*N+j]);
        }
    }
    printf("\n\n");
}   
