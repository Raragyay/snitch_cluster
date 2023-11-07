#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "../gradient/inc/gradient.h"
#include "inc/main.h"
#include "../gemm/inc/gemm.h"

//quadratic error,example, we could use different cost functions. this is the easiest
void compute_error(){
    for(int i=0;i<M*N;i++){
        E[i]= Y[i]- S[i];
    }
}

//avoids weight/biases explosion
void adjust_W_B(){
    for(int i=0;i<K*N;i++){
        if(W[i]<-1) W[i]=-1;
        if(W[i]>1) W[i]=1;
    }
    for(int i=0;i<M*N;i++){
        if(B[i]<-1) B[i]=-1;
        if(B[i]>1) B[i]=1;
    }
}

int main(int argc, char *argv[]) {
    int err=0,right_class;
    DATA_TYPE grad_B[M*N];
    DATA_TYPE grad_W[N*K];
    for(int i=0;i<1000;i++){
        gemm(1,1,0,0,I,W,B,Y,M,K,N);
        compute_error();
        err=0;
        right_class=0;
        if(i%100==0){
            for(int j=0;j<M*N;j++){
                err+=E[j]*E[j];
                if(Y[j]>=0.5 && S[j]==1 || Y[j]<0.5 && S[j]==0)
                    right_class++;
            }
            printf("Quad error it %d: %d, Right class: %d / %d\n",i, err/2, right_class,M*N);

        }
        backpropagation(I,W,B,grad_W,grad_W,E,e,M,N,K);
        adjust_W_B();
    }

    gemm(1,1,0,0,I,W,B,Y,M,K,N);
    err=0;
    right_class=0;
    for(int j=0;j<M*N;j++){
        err+=E[j]*E[j];
        if(Y[j]>=0.5 && S[j]==1 || Y[j]<0.5 && S[j]==0)
            right_class++;
    }
        printf("Quad error: %.6f, Right classification: %d/%d\n",(float)err/2,right_class,M*N);

}   
