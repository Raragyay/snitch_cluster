#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include "gradient.h"

#define I(i,j) I[(i)*k + (j)]
#define I_t(i,j) I[(i)*m + (j)]
#define W(i,j) W[(i)*n + (j)]
#define B(i,j) B[(i)*n + (j)]
#define E(i,j) E[(i)*n + (j)]
#define grad_W(i,j) grad_W[(i)*n + (j)]
#define grad_B(i,j) grad_B[(i)*n + (j)]

// I[m][k] inputs
// W[k][n] weights
// B[m][n] biases
// E[m][n] error
// e learning rate
void backpropagation(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             float e,int m, int n, int k){

    DATA_TYPE grad_B[m*n];//I dont like dynamic
    DATA_TYPE grad_W[n*k];//I dont like dynamic
    __find_grads__(I, W, B, E,grad_W,grad_B, m, n, k);
    __update_W_B__(W, B, e, grad_W, grad_B, m, n, k);
}


//W[k][n] = old(W[k][n]) - e*grad[dC/dW]
//B[m][n] = old(B[m][n])- e*grad[dC/dB]
void __update_W_B__(DATA_TYPE *W,DATA_TYPE *B, float e,DATA_TYPE *grad_W,DATA_TYPE *grad_B,int m, int n, int k){
    int i;
    for(i=0;i<n*k;i++){
        W[i] = W[i] - e*grad_W[i];  

    }
    for(i=0;i<n*m;i++){
        B[i] = B[i]- e*grad_B[i];
    }
}


//dC/db = E = grad_E
//dC/dW = I_t * E =grad_W
void __find_grads__( DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E, DATA_TYPE *grad_W,DATA_TYPE *grad_B, int m, int n, int k){
    int i,j,z;
    DATA_TYPE sum;

    for(i=0;i<m;i++)
        for(j=0;j<n;j++)
            grad_B(i,j) = E(i,j);
    

    //It does not make sense to use the gemm, because it wastes time compared to what we need (ex. alpha,beta, bias,transA,transB)
    for(i=0;i<k;i++){
        for(j=0;j<n;j++){
        sum=0;
        for(z=0;z<m;z++){
            sum += I_t(z,i)*E(z,j);
        }
        grad_W(i,j)=sum;
        }
    }
}
