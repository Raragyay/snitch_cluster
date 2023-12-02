#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "snrt.h"
#include "gradient.h"
#include "data.h"

//#define SINGLE_CORE
//#define BASELINE

int main(int argc, char *argv[]) {
    #ifdef SINGLE_CORE
        #ifdef BASELINE
            backpropagation_baseline_one_core(alpha,A,B,GRAD_C,GRAD_A,GRAD_B,M,N,K);
        #else
            backpropagation_one_core(alpha,A,B,GRAD_C,GRAD_A,GRAD_B,M,N,K);
        #endif
    #else

        #ifdef BASELINE 
            backpropagation_baseline_multicore(alpha,A,B,GRAD_C,GRAD_A,GRAD_B,M,N,K);
        #else
            backpropagation_multicore(alpha,A,B,GRAD_C,GRAD_A,GRAD_B,M,N,K,1,1,1,0,1);
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
