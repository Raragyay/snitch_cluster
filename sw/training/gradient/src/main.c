#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "snrt.h"
#include "gradient.h"
#include "data.h"

#define SINGLE_CORE
//#define BASELINE

int main(int argc, char *argv[]) {
    #ifdef SINGLE_CORE
        #ifdef BASELINE
            backpropagation_baseline_one_core(I,E,grad_W,M,N,K);
        #else
            backpropagation_one_core(I,E,grad_W,M,N,K);
        #endif
    #else

        #ifdef BASELINE 
            backpropagation_baseline_multicore(I,E,grad_W,M,N,K);
        #else
       //     backpropagation_multicore(W,B,W_grad,B_grad,N,K,dtype_size);
        #endif
    #endif
}   



        // while(1){
        //     asm volatile(
        //     "add a0, a0, %[zero]\n"
        //     :
        //     :[ zero ] "r"(0)
        //     : "a0");
        // }
