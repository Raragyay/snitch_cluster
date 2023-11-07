#include <math.h>
#include <stdint.h>

#include "data.h"
#include "gemm.h"
#include "snrt.h"


int main(int argc, char *argv[]) {

    DATA_TYPE t [M*N];

    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++)
            t[i*N +j] = 0;
    snrt_cluster_hw_barrier();


    DATA_TYPE *a11 = a,          *a12 = a + K/2;
    DATA_TYPE *a21 = a + M/2 *K, *a22 = a + M/2 *K + K/2;

    DATA_TYPE *b11 = b,          *b12 = b + N/2;
    DATA_TYPE *b21 = b + K/2 *N, *b22 = a + K/2 *N + N/2;

    DATA_TYPE *c11 = c,          *c12 = c + N/2;
    DATA_TYPE *c21 = c + M/2 *N, *c22 = c + M/2 *N + N/2;

    DATA_TYPE *t11 = t,          *t12 = t + N/2;
    DATA_TYPE *t21 = t + M/2 *N, *t22 = t + M/2 *N + N/2;
    
    #ifdef SINGLE_CORE
    if (snrt_cluster_core_idx() == 0)
        gemm(M, N, K, M, N, K, a, TA, b, TB, c, BETA);
    #else
    switch (snrt_cluster_core_idx()) {
        case 0:
            gemm (M, N, K, M/2, N/2, K/2, a11, TA, b11, TB, c11, BETA);
            break;
        case 1:
            gemm (M, N, K, M/2, N/2, K/2, a11, TA, b12, TB, c12, BETA);
            break;
        case 2:
            gemm (M, N, K, M/2, N/2, K/2, a21, TA, b11, TB, c21, BETA);
            break;
        case 3:
            gemm (M, N, K, M/2, N/2, K/2, a21, TA, b12, TB, c22, BETA);
            break;
        case 4:
            gemm (M, N, K, M/2, N/2, K/2, a12, TA, b21, TB, t11, BETA);
            break;
        case 5:
            gemm (M, N, K, M/2, N/2, K/2, a12, TA, b22, TB, t12, BETA);
            break;
        case 6:
            gemm (M, N, K, M/2, N/2, K/2, a22, TA, b21, TB, t21, BETA);
            break;
        case 7:
            gemm (M, N, K, M/2, N/2, K/2, a22, TA, b22, TB, t22, BETA);
            break;
    }


    snrt_cluster_hw_barrier();
    if (snrt_cluster_core_idx() == 0)
        for (uint32_t i = 0; i < M; i++)
            for (uint32_t j = 0; j < N; j++)
                c[i*N +j] += t[i*N +j];
    #endif
}   