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
            backpropagation_baseline_one_core(I,W,B,E,e,N,K);
        #else
            backpropagation_one_core(I,W,B,E,e,N,K);
        #endif
    #else

        #ifdef BASELINE 
            backpropagation_baseline_multicore(I,W,B,E,e,N,K,dtype_size);
        #else
            backpropagation_multicore(I,W,B,E,e,N,K,dtype_size);
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
