#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "gemm.h"

#define A(i,j) A[(i)*k + (j)]
#define A_t(i,j) A[(i)*m + (j)]
#define B(i,j) B[(i)*n + (j)]
#define B_t(i,j) B[(i)*k + (j)]
#define C(i,j) C[(i)*n + (j)]
#define Y(i,j) Y[(i)*n + (j)]

// alpha*A[m][k]*B[k][n] + beta*C[m][n] = Y[m][n]
void gemm(float alpha, float beta, int transA, int transB, 
            DATA_TYPE *A, DATA_TYPE *B,DATA_TYPE *C, DATA_TYPE *Y,
            int m,int k, int n){
    DATA_TYPE sum;
    int i,j,z;
    DATA_TYPE val_A,val_B;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            sum=0;
            for(z=0;z<k;z++){
                val_A =(transA) ? A_t(z,i) : A(i,z);
                val_B = (transB) ? B_t(j,z) : B(z,j);
                sum =sum+ val_A*val_B;
            }
            //printf("Sum: %f, C: %f\t", sum,C(i,j));
            if(C!=NULL)
                Y(i,j)=alpha*sum + beta*C(i,j);
            else
                Y(i,j) = alpha*sum;
        }
    }
}
