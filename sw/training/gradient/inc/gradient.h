#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif 


void backpropagation(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B,DATA_TYPE*grad_W, DATA_TYPE*grad_B, DATA_TYPE *E,
             float e,int m, int n, int k);
void __update_W_B__(DATA_TYPE *W,DATA_TYPE *B, float e,DATA_TYPE *grad_W,DATA_TYPE *grad_B,int m, int n, int k);
void __find_grads__( DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E, DATA_TYPE *grad_W,DATA_TYPE *grad_B, int m, int n, int k);