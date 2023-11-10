#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include "snrt.h"

#define A(i,j)   A[(i)*m + (j)]
#define A_t(i,j) A[(i)*k + (j)]
#define B(i,j)   B[(i)*k + (j)]
#define B_t(i,j) B[(i)*n + (j)]
#define C(i,j)   C[(i)*m + (j)]
#define Y(i,j)   Y[(i)*m + (j)]

#define DATA_TYPE double
//#define SSRFREP

// alpha*A[m][k]*B[k][n] + beta*C[m][n] = Y[m][n]
void gemm(uint32_t M, uint32_t N, uint32_t K, uint32_t sM, uint32_t sN, uint32_t sK, double* A,
                        uint32_t ta, double* B, 
                        uint32_t tb, double* C, double BETA){

    DATA_TYPE res;


    if (!ta && !tb) {


        for (uint32_t m = 0; m < sM; m++) {
            for (uint32_t n = 0; n < sN; n++) {
                res = BETA * C[m * N + n];

                #ifdef SSRFREP
                snrt_ssr_loop_1d(SNRT_SSR_DM0, sK, 8);
                snrt_ssr_loop_1d(SNRT_SSR_DM1, sK, 8*N);
                snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, A + m*K);
                snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, B + n);

                asm volatile
                ("frep.o %[n_frep], 1, 0, 0          \n"
                 "fmadd.d %[res], ft1, ft2, %[res]   \n"
                : [res] "+f"(res)
                : [n_frep] "r"(sK-1)
                : "ft0", "ft1"
                );

                #else
                for (uint32_t k = 0; k < sK; k++)
                    res += A[k + m * K] * B[k * N + n];
                #endif

                C[m * N + n] = res;
            }
        }
    } else if (ta && !tb) {
        for (uint32_t m = 0; m < sM; m++) {
            for (uint32_t n = 0; n < sN; n++) {
                res = BETA * C[m * N + n];
                for (uint32_t k = 0; k < sK; k++) {
                    res += A[k * M + m] * B[k * N + n];
                }
                C[m * N + n] = res;
            }
        }
    } else if (!ta && tb) {
        for (uint32_t m = 0; m < sM; m++) {
            for (uint32_t n = 0; n < sN; n++) {
                res = BETA * C[m * N + n];
                for (uint32_t k = 0; k < sK; k++) {
                    res += A[k + m * K] * B[k + n * K];
                }
                C[m * N + n] = res;
            }
        }
    } else {
        for (uint32_t m = 0; m < sM; m++) {
            for (uint32_t n = 0; n < sN; n++) {
                res = BETA * C[m * N + n];
                for (uint32_t k = 0; k < sK; k++) {
                    res += A[k * M + m] * B[k + n * K];
                }
                C[m * N + n] = res;
            }
        }
    }
}


// void __tiles_gemm__(float alpha, float beta, DATA_TYPE *A, DATA_TYPE *B,DATA_TYPE *C, DATA_TYPE *Y,
//             int m,int k, int n){
//     int i,j,z;
//     int ib,jb;
//     DATA_TYPE c_block[TILE_SIZE_M*TILE_SIZE_N];
//     for (i=0;i<NUM_OF_TILES_M;i++){
//         for(j=0;j<NUM_OF_TILES_N;j++){
//             __reset_c_block__(c_block);
//             for(z=0;z<k;z++){
//                 for(ib =0;ib< TILE_SIZE_M;ib++){
//                     for(jb=0;jb<TILE_SIZE_N;jb++){
//                         c_block[ib*TILE_SIZE_M+jb] =c_block[ib*TILE_SIZE_M+jb]+A[z*m+i*TILE_SIZE_M+ib]*B[(j*TILE_SIZE_N+jb)*k+z];
//                     }
//                 }
//             }
//             for(ib=0;ib<TILE_SIZE_M;ib++){
//                 for(jb=0;jb<TILE_SIZE_N;jb++){
//                     Y[(j*TILE_SIZE_M+jb)*m+i*TILE_SIZE_N+ib] = c_block[jb*TILE_SIZE_M+ib];     
//                 }
//             }
//         }
//     }
// }


// void __reset_c_block__(DATA_TYPE *c_block){
//     for(int i=0;i<TILE_SIZE_M*TILE_SIZE_N;i++)
//         c_block[i]=0;
// }
