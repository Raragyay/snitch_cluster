#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "gemm.h"
#include "main.h"


int main(int argc, char *argv[]) {
    
    gemm(alpha,beta,transA,transB,A,B,C,Y,M,K,N);
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if(i==M-1 && j==N-1)
                printf("%f\n",Y[i*N+j]);
            else
                printf("%f,",Y[i*N+j]);
        }
        printf("\n");
    }
}   
