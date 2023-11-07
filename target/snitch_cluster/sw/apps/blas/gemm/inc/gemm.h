#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif 

// #define NUM_OF_TILES_M 2
// #define NUM_OF_TILES_N 2
// #define TILE_SIZE_M 2
// #define TILE_SIZE_N 2

void gemm(float alpha, float beta, int transA, int transB, 
            DATA_TYPE *A,DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *Y,
            int m,int k, int n);

// void __tiles_gemm__(float alpha, float beta, DATA_TYPE *A, DATA_TYPE *B,DATA_TYPE *C, DATA_TYPE *Y,
//             int m,int k, int n);

// void __reset_c_block__(DATA_TYPE *c_block);