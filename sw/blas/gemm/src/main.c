#include <math.h>
#include <stdint.h>

#include "data.h"
#include "gemm.h"
#include "snrt.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))


int main(int argc, char *argv[]) {

    // Allocate space in TCDM
    uint32_t size_a = M * K * sizeof(DATA_TYPE);
    uint32_t size_b = K * N * sizeof(DATA_TYPE);
    uint32_t size_c = M * N * sizeof(DATA_TYPE);

    DATA_TYPE *local_a, *local_b, *local_c;
    local_a = (DATA_TYPE *)snrt_l1_next();
    local_b = local_a + size_a; //maybe multiplying by sizeof(datatype) isn't needed
    local_c = local_b + size_b;
    DATA_TYPE* t = local_c + size_c;

    // Copy data in TCDM
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, a, size_a);
        snrt_dma_start_1d(local_b, b, size_b);
        snrt_dma_start_1d(local_c, c, size_c);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Compute
    if (!snrt_is_dm_core()) {

    #ifdef SINGLE_CORE
    if (snrt_cluster_core_idx() == 0)
        gemm(M, N, K, M, N, K, local_a, TA, local_b, TB, local_c, BETA);
    #else

    DATA_TYPE *a11 = local_a,          *a12 = local_a + K/2;
    DATA_TYPE *a21 = local_a + M/2 *K, *a22 = local_a + M/2 *K + K/2;

    DATA_TYPE *b11 = local_b,          *b12 = local_b + N/2;
    DATA_TYPE *b21 = local_b + K/2 *N, *b22 = local_b + K/2 *N + N/2;

    DATA_TYPE *c11 = local_c,          *c12 = local_c + N/2;
    DATA_TYPE *c21 = local_c + M/2 *N, *c22 = local_c + M/2 *N + N/2;

    DATA_TYPE *t11 = t,                *t12 = t + N/2;
    DATA_TYPE *t21 = t + M/2 *N,       *t22 = t + M/2 *N + N/2;


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

        snrt_fpu_fence();
    }

    snrt_cluster_hw_barrier();

    if (!snrt_is_dm_core()) { ////////////////////////Call add function
        uint32_t c, lb, ub, core_idx = snrt_cluster_core_idx();
        c = CEIL(M, snrt_cluster_core_num());
        lb = c * core_idx;
        ub = MIN((c * (core_idx + 1)), M);

        for (uint32_t i = lb; i < ub; i++) {
            for (uint32_t j = 0; j < N; j++)
                local_c[i*N +j] += t[i*N +j];
        }
        snrt_fpu_fence();
    }
    #endif
    snrt_cluster_hw_barrier();

    // Copy data out of TCDM
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(c, local_c, size_c);
        snrt_dma_wait_all();
    }
    
    snrt_cluster_hw_barrier(); 
  
}   