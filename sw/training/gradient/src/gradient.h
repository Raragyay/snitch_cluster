
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif 


#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))

#define MAX_DIM 100000 / sizeof(DATA_TYPE) 

uint32_t num_iter;

uint32_t stalls_grad_B[8];
uint32_t stalls_grad_A[8];

void backpropagation_baseline_one_core(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *grad_C,DATA_TYPE *grad_A,DATA_TYPE *grad_B,
                    uint32_t M, uint32_t N, uint32_t K);

void backpropagation_baseline_multicore(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K);

void backpropagation_one_core(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K);

void backpropagation_multicore(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K);

void __backpropagation_multicore_computation_grad_B_fp64__(DATA_TYPE alpha, DATA_TYPE *local_A, DATA_TYPE *local_GRAD_C, DATA_TYPE *local_GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub);

void __backpropagation_multicore_computation_grad_A_fp64__(DATA_TYPE alpha, DATA_TYPE *local_GRAD_C, DATA_TYPE *local_B, DATA_TYPE *local_GRAD_A,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub);
                    

static inline uint64_t asuint(float f);
static inline float asfloat(uint32_t i);


// A[M][K]
// B[K][N]
// grad_C[M][N]
void backpropagation_baseline_one_core(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K){

    int i,j,z;
    DATA_TYPE sum;
    uint32_t size_A, size_B,size_GRAD_C;
    DATA_TYPE *local_A, *local_B, *local_GRAD_C,*local_GRAD_RES; //*local_GRAD_RES used for both computations

    size_A = M*K;
    size_B = K*N;
    size_GRAD_C = M*N;

    local_A = (DATA_TYPE *)snrt_l1_next();
    local_B = local_A + size_A;
    local_GRAD_C = local_B + size_B;
    local_GRAD_RES = local_GRAD_C + size_GRAD_C;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_A, A, M*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_B, B, K*N*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_GRAD_C, GRAD_C, M*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();


    if(snrt_cluster_core_idx()==0){
        snrt_mcycle();

        for(i=0;i<K;i++){
            for(j=0;j<N;j++){
                sum = 0;
                for(z=0;z<M;z++){
                    sum += local_A[z*K+i] * local_GRAD_C[z*N+j];
                };
                local_GRAD_RES [i*N+j] = alpha*sum;
            }
        }        
        snrt_mcycle();
    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(GRAD_B, local_GRAD_RES, K*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if(snrt_cluster_core_idx()==0){
        snrt_mcycle();

        for(i=0;i<M;i++){
            for(j=0;j<K;j++){
                sum = 0;
                for(z=0;z<N;z++){
                    sum += local_GRAD_C[i*N+z] * local_B[z+j*N];
                }
                local_GRAD_RES [i*K+j] = alpha*sum;
            }
        }        
        snrt_mcycle();
    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(GRAD_A, local_GRAD_RES, M*K*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
}


//works only without tiling
void backpropagation_baseline_multicore(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K){

    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    int32_t i,c,lb,ub,j,z;
    DATA_TYPE sum;
    uint32_t size_A, size_B,size_GRAD_C;
    DATA_TYPE *local_A, *local_B, *local_GRAD_C,*local_GRAD_RES; //*local_GRAD_RES used for both computations

    size_A = M*K;
    size_B = K*N;
    size_GRAD_C = M*N;

    local_A = (DATA_TYPE *)snrt_l1_next();
    local_B = local_A + size_A;
    local_GRAD_C = local_B + size_B;
    local_GRAD_RES = local_GRAD_C + size_GRAD_C;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_A, A, M*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_B, B, K*N*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_GRAD_C, GRAD_C, M*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();


    if(!snrt_is_dm_core()){
        snrt_mcycle();

        c = CEIL(K, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), K);

        for(i=lb;i<ub;i++){
            for(j=0;j<N;j++){
                sum = 0;
                for(z=0;z<M;z++){
                    sum += local_A[z*K+i] * local_GRAD_C[z*N+j];
                };
                local_GRAD_RES [i*N+j] = alpha*sum;
            }
        }        
        snrt_mcycle();
    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(GRAD_B, local_GRAD_RES, K*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if(!snrt_is_dm_core()){
        snrt_mcycle();

        c = CEIL(M, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), M);

        for(i=lb;i<ub;i++){
            for(j=0;j<K;j++){
                sum = 0;
                for(z=0;z<N;z++){
                    sum += local_GRAD_C[i*N+z] * local_B[z+j*N];
                }
                local_GRAD_RES [i*K+j] = alpha*sum;
            }
        }        
        snrt_mcycle();
    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(GRAD_A, local_GRAD_RES, M*K*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();        
}


//works only without tiling
void backpropagation_one_core(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K){
    
    int i,j,z,unroll=4;
    DATA_TYPE sum;
    uint32_t size_A, size_B,size_GRAD_C;
    DATA_TYPE *local_A, *local_B, *local_GRAD_C,*local_GRAD_RES; //*local_GRAD_RES used for both computations

    size_A = M*K;
    size_B = K*N;
    size_GRAD_C = M*N;

    local_A = (DATA_TYPE *)snrt_l1_next();
    local_B = local_A + size_A;
    local_GRAD_C = local_B + size_B;
    local_GRAD_RES = local_GRAD_C + size_GRAD_C;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_A, A, M*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_B, B, K*N*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_GRAD_C, GRAD_C, M*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    if(snrt_cluster_core_idx()==0){

        //prepare loop A transposed    
        const uint32_t ssr0_b[4] = {unroll, M, N / unroll, K};
        const uint32_t ssr0_i[4] = {0, 8 * K, 0, 8 };

        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                            ssr0_i[1], ssr0_i[2], ssr0_i[3]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);

        //prepare loop GRAD_C
        const uint32_t ssr1_b[4] = {unroll, M, N / unroll, K}; 
        const uint32_t ssr1_i[4] = {8, 8 * N, 8 * unroll, 0};


        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                            ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                            ssr1_i[3]);

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_A);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_GRAD_C);

        
        for (uint32_t k=0; k<K; k++) { 
            uint32_t n = 0;

            snrt_ssr_enable();

            for (uint32_t n0=0; n0<N/unroll; n0++) { 

                register DATA_TYPE sum[]={0.0f,0.0f,0.0f,0.0f};
                asm volatile(
                    "frep.o %[n_frep], 4, 0, 0 \n"
                    "fmadd.d %[sum0], ft0, ft1, %[sum0] \n"
                    "fmadd.d %[sum1], ft0, ft1, %[sum1] \n"
                    "fmadd.d %[sum2], ft0, ft1, %[sum2] \n"
                    "fmadd.d %[sum3], ft0, ft1, %[sum3] \n"

                    "fmul.d %[sum0], %[alpha],%[sum0] \n"
                    "fmul.d %[sum1], %[alpha],%[sum1] \n"
                    "fmul.d %[sum2], %[alpha],%[sum2] \n"
                    "fmul.d %[sum3], %[alpha],%[sum3] \n"

                    
                    :[ sum0 ] "+f"(sum[0]), [ sum1 ] "+f"(sum[1]), [ sum2 ] "+f"(sum[2]),
                        [ sum3 ] "+f"(sum[3])
                    : [ n_frep ] "r"(M - 1), [alpha] "f"(alpha)
                    : "ft0", "ft1", "ft2");

                // Store results back
                local_GRAD_RES[k*N + n + 0] = sum[0];
                local_GRAD_RES[k*N + n + 1] = sum[1];
                local_GRAD_RES[k*N + n + 2] = sum[2];
                local_GRAD_RES[k*N + n + 3] = sum[3];
                n += unroll;         
            }

            snrt_ssr_disable();

            for (; n<N; n++) {
                double sum=0;
                for (uint32_t m=0; m<M; m++) {
                    sum += local_A[k + m*K] * local_GRAD_C[m*N + n];
                }
                local_GRAD_RES[k*N + n] = alpha*sum;
            }
        }

        snrt_fpu_fence();
    }

    snrt_mcycle();
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(GRAD_B, local_GRAD_RES, K*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();   
    snrt_mcycle();

    if(snrt_cluster_core_idx()==0){
        //prepare loop GRAD_C    
        const uint32_t ssr0_b[4] = {unroll, N, K/unroll, M};
        const uint32_t ssr0_i[4] = {0, 8, 0, 8 * N};

        // A[k + unroll * m * ldA]
        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                            ssr0_i[1], ssr0_i[2], ssr0_i[3]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
        
        //prepare loop B TRANSPOSED 
        const uint32_t ssr1_b[4] = {unroll, N, K / unroll, M};
        const uint32_t ssr1_i[4] = {8*N, 8, N*unroll*8, 0};

        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                            ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                            ssr1_i[3]);

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_GRAD_C);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_B);

        
        for (uint32_t m=0; m<M; m++) { 
            uint32_t k = 0;

            snrt_ssr_enable();

            for (uint32_t k0=0; k0<K/unroll; k0++) { 

                register DATA_TYPE sum[]={0.0f,0.0f,0.0f,0.0f};
                asm volatile(
                    "frep.o %[n_frep], 4, 0, 0 \n"
                    "fmadd.d %[sum0], ft0, ft1, %[sum0] \n"
                    "fmadd.d %[sum1], ft0, ft1, %[sum1] \n"
                    "fmadd.d %[sum2], ft0, ft1, %[sum2] \n"
                    "fmadd.d %[sum3], ft0, ft1, %[sum3] \n"

                    "fmul.d %[sum0], %[alpha],%[sum0] \n"
                    "fmul.d %[sum1], %[alpha],%[sum1] \n"
                    "fmul.d %[sum2], %[alpha],%[sum2] \n"
                    "fmul.d %[sum3], %[alpha],%[sum3] \n"

                    
                    :[ sum0 ] "+f"(sum[0]), [ sum1 ] "+f"(sum[1]), [ sum2 ] "+f"(sum[2]),
                        [ sum3 ] "+f"(sum[3])
                    : [ n_frep ] "r"(N - 1), [alpha] "f"(alpha)
                    : "ft0", "ft1", "ft2");

                // Store results back
                local_GRAD_RES[m*K + k + 0] = sum[0];
                local_GRAD_RES[m*K + k + 1] = sum[1];
                local_GRAD_RES[m*K + k + 2] = sum[2];
                local_GRAD_RES[m*K + k + 3] = sum[3];
                k += unroll;         
            }

            snrt_ssr_disable();

            for (; k<K; k++) {
                double sum=0;
                for (uint32_t n=0; n<N; n++) {
                    sum +=  local_GRAD_C[m*N+n]*local_B[n+k*N];
                }
                local_GRAD_RES[m*K + k] = alpha*sum;
            }
        }

        snrt_fpu_fence();
    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();
    snrt_mcycle();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(GRAD_A, local_GRAD_RES, M*K*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

}


void backpropagation_multicore(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K){
                        
    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    int32_t c,lb,ub;
    DATA_TYPE sum;
    uint32_t size_A, size_B, size_GRAD_C;
    DATA_TYPE *local_A, *local_B, *local_GRAD_C, *local_GRAD_RES; //*local_GRAD_RES used for both computations

    size_A = M*K;
    size_B = K*N;
    size_GRAD_C = M*N;

    local_A = (DATA_TYPE *)snrt_l1_next();
    local_B = local_A + size_A;
    local_GRAD_C = local_B + size_B;
    local_GRAD_RES = local_GRAD_C + size_GRAD_C;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_A, A, M*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_B, B, K*N*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_GRAD_C, GRAD_C, M*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();


    if(!snrt_is_dm_core()){
        snrt_mcycle();

        c = CEIL(K, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), K);

        __backpropagation_multicore_computation_grad_B_fp64__(alpha,local_A,local_GRAD_C,local_GRAD_RES,M,N,K,lb,ub);

        snrt_mcycle();
    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(GRAD_B, local_GRAD_RES, K*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if(!snrt_is_dm_core()){
        snrt_mcycle();

        c = CEIL(M, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), M);

        __backpropagation_multicore_computation_grad_A_fp64__(alpha,local_GRAD_C,local_B,local_GRAD_RES,M,N,K,lb,ub);

        snrt_mcycle();
    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(GRAD_A, local_GRAD_RES, M*K*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

}



void __backpropagation_multicore_computation_grad_B_fp64__(DATA_TYPE alpha, DATA_TYPE *local_A, DATA_TYPE *local_GRAD_C, DATA_TYPE *local_GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub){

    int32_t dim = ub-lb;
    int32_t unroll=8;//high contention due to the transposed first matrix.
    if(dim>0){

        //prepare loop A transposed    
        const uint32_t ssr0_b[4] = {unroll, M, N / unroll, dim};
        const uint32_t ssr0_i[4] = {0, 8 * K, 0, 8 };

        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                            ssr0_i[1], ssr0_i[2], ssr0_i[3]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);

        //prepare loop GRAD_C
        const uint32_t ssr1_b[4] = {unroll, M, N / unroll, dim}; 
        const uint32_t ssr1_i[4] = {8, 8 * N, 8 * unroll, 0};


        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                            ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                            ssr1_i[3]);

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_A+lb);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_GRAD_C);

     //   snrt_start_perf_counter(SNRT_PERF_CNT0,SNRT_PERF_CNT_TCDM_CONGESTED,snrt_cluster_core_idx());
        for (uint32_t k=lb; k<ub; k++) { 
            uint32_t n = 0;

            snrt_ssr_enable();

            for (uint32_t n0=0; n0<N/unroll; n0++) { 
            
                register DATA_TYPE sum[]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
                asm volatile(
                    "frep.o %[n_frep], 8, 0, 0 \n"
                    "fmadd.d %[sum0], ft0, ft1, %[sum0] \n"
                    "fmadd.d %[sum1], ft0, ft1, %[sum1] \n"
                    "fmadd.d %[sum2], ft0, ft1, %[sum2] \n"
                    "fmadd.d %[sum3], ft0, ft1, %[sum3] \n"
                    "fmadd.d %[sum4], ft0, ft1, %[sum4] \n"
                    "fmadd.d %[sum5], ft0, ft1, %[sum5] \n"
                    "fmadd.d %[sum6], ft0, ft1, %[sum6] \n"
                    "fmadd.d %[sum7], ft0, ft1, %[sum7] \n"

                    "fmul.d %[sum0], %[alpha],%[sum0] \n"
                    "fmul.d %[sum1], %[alpha],%[sum1] \n"
                    "fmul.d %[sum2], %[alpha],%[sum2] \n"
                    "fmul.d %[sum3], %[alpha],%[sum3] \n"
                    "fmul.d %[sum4], %[alpha],%[sum4] \n"
                    "fmul.d %[sum5], %[alpha],%[sum5] \n"
                    "fmul.d %[sum6], %[alpha],%[sum6] \n"
                    "fmul.d %[sum7], %[alpha],%[sum7] \n"
                    
                    :[ sum0 ] "+f"(sum[0]), [ sum1 ] "+f"(sum[1]), [ sum2 ] "+f"(sum[2]),
                        [ sum3 ] "+f"(sum[3]), [ sum4 ] "+f"(sum[4]), [ sum5 ] "+f"(sum[5]),
                        [ sum6 ] "+f"(sum[6]), [ sum7 ] "+f"(sum[7])
                    : [ n_frep ] "r"(M - 1), [alpha] "f"(alpha)
                    : "ft0", "ft1", "ft2");

                // Store results back
                local_GRAD_B[k*N + n + 0] = sum[0];
                local_GRAD_B[k*N + n + 1] = sum[1];
                local_GRAD_B[k*N + n + 2] = sum[2];
                local_GRAD_B[k*N + n + 3] = sum[3];
                local_GRAD_B[k*N + n + 4] = sum[4];
                local_GRAD_B[k*N + n + 5] = sum[5];
                local_GRAD_B[k*N + n + 6] = sum[6];
                local_GRAD_B[k*N + n + 7] = sum[7];

                n += unroll;         
            }

            snrt_ssr_disable();

            for (; n<N; n++) {
                double sum=0;
                for (uint32_t m=0; m<M; m++) {
                    sum += local_A[k + m*K] * local_GRAD_C[m*N + n];
                }
                local_GRAD_B[k*N + n] = alpha*sum;
            }
        }
     //   snrt_stop_perf_counter(SNRT_PERF_CNT0);
     //   stalls_grad_B[snrt_cluster_core_idx()] = snrt_get_perf_counter(SNRT_PERF_CNT0);

        snrt_fpu_fence();

    }

}


void __backpropagation_multicore_computation_grad_A_fp64__(DATA_TYPE alpha, DATA_TYPE *local_GRAD_C, DATA_TYPE *local_B, DATA_TYPE *local_GRAD_A,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub){
                        
    int32_t dim = ub-lb,unroll=4;
    if(dim>0){

        //prepare loop GRAD_C    
        const uint32_t ssr0_b[4] = {unroll, N, K/unroll, dim};
        const uint32_t ssr0_i[4] = {0, 8, 0, 8 * N};

        // A[k + unroll * m * ldA]
        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                            ssr0_i[1], ssr0_i[2], ssr0_i[3]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
        
        //prepare loop B TRANSPOSED 
        const uint32_t ssr1_b[4] = {unroll, N, K / unroll, dim};
        const uint32_t ssr1_i[4] = {8*N, 8, N*unroll*8, 0};

        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                            ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                            ssr1_i[3]);

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_GRAD_C + lb*N);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_B);

        //snrt_start_perf_counter(SNRT_PERF_CNT1,SNRT_PERF_CNT_TCDM_CONGESTED,snrt_cluster_core_idx());

        for (uint32_t m=lb; m<ub; m++) { 
            uint32_t k = 0;

            snrt_ssr_enable();

            for (uint32_t k0=0; k0<K/unroll; k0++) { 

                register DATA_TYPE sum[]={0.0f,0.0f,0.0f,0.0f};
                asm volatile(
                    "frep.o %[n_frep], 4, 0, 0 \n"
                    "fmadd.d %[sum0], ft0, ft1, %[sum0] \n"
                    "fmadd.d %[sum1], ft0, ft1, %[sum1] \n"
                    "fmadd.d %[sum2], ft0, ft1, %[sum2] \n"
                    "fmadd.d %[sum3], ft0, ft1, %[sum3] \n"

                    "fmul.d %[sum0], %[alpha],%[sum0] \n"
                    "fmul.d %[sum1], %[alpha],%[sum1] \n"
                    "fmul.d %[sum2], %[alpha],%[sum2] \n"
                    "fmul.d %[sum3], %[alpha],%[sum3] \n"

                    
                    :[ sum0 ] "+f"(sum[0]), [ sum1 ] "+f"(sum[1]), [ sum2 ] "+f"(sum[2]),
                        [ sum3 ] "+f"(sum[3])
                    : [ n_frep ] "r"(N - 1), [alpha] "f"(alpha)
                    : "ft0", "ft1", "ft2");

                // Store results back
                local_GRAD_A[m*K + k + 0] = sum[0];
                local_GRAD_A[m*K + k + 1] = sum[1];
                local_GRAD_A[m*K + k + 2] = sum[2];
                local_GRAD_A[m*K + k + 3] = sum[3];
                k += unroll;         
            }

            snrt_ssr_disable();

            for (; k<K; k++) {
                double sum=0;
                for (uint32_t n=0; n<N; n++) {
                    sum +=  local_GRAD_C[m*N+n]*local_B[n+k*N];
                }
                local_GRAD_A[m*K + k] = alpha*sum;
            }
        }
        //snrt_stop_perf_counter(SNRT_PERF_CNT1);
        //stalls_grad_A[snrt_cluster_core_idx()] = snrt_get_perf_counter(SNRT_PERF_CNT1);


        snrt_fpu_fence();    
    }
}






static inline uint64_t asuint(float f) {
    uint32_t result;
    snrt_fpu_fence();
    result = *(uint32_t *)&f;
    return result;
}

static inline float asfloat(uint32_t i) {
	float result;
	snrt_fpu_fence();
	result = *(float *)&i;
	return result;
}
