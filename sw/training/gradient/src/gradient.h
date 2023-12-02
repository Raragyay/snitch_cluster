
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
                    uint32_t M, uint32_t N, uint32_t K, uint32_t M_tiles, uint32_t N_tiles, uint32_t K_tiles,
                    uint32_t compute_grad_a,uint32_t compute_grad_b);

void __backpropagation_multicore_computation_grad_B_fp64__(DATA_TYPE alpha, DATA_TYPE *local_A, DATA_TYPE *local_GRAD_C, DATA_TYPE *local_GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize);

void __backpropagation_multicore_computation_grad_A_fp64__(DATA_TYPE alpha, DATA_TYPE *local_GRAD_C, DATA_TYPE *local_B, DATA_TYPE *local_GRAD_A,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize);
                    

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


/*requires M%M_tiles==0 && N%N_tiles==0 && K%K_tiles==0 */
void backpropagation_multicore(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K, uint32_t M_tiles, uint32_t N_tiles, uint32_t K_tiles,
                    uint32_t compute_grad_a,uint32_t compute_grad_b){

                        
    const uint32_t frac_M = M / M_tiles;
    const uint32_t frac_N = N / N_tiles;
    const uint32_t frac_K = K / K_tiles;

    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    int32_t c,lb,ub;
    DATA_TYPE sum;
    uint32_t k,m,n;
    uint32_t size_A, size_B, size_GRAD_C,size_GRAD_B;
    DATA_TYPE *local_A, *local_B, *local_GRAD_C, *local_GRAD_RES; //*local_GRAD_RES used for both computations
    uint32_t mult_alpha,initialize;
    int iter=0;

    size_A = frac_M * frac_K;
    size_B = frac_N * frac_K;
    size_GRAD_C = frac_M * frac_N;
    size_GRAD_B = frac_N * frac_K;

    local_A = (DATA_TYPE *)snrt_l1_next();
    local_GRAD_RES = local_A + size_A;
    local_GRAD_C = local_GRAD_RES + size_GRAD_B;
    

    if(compute_id==0){
        snrt_start_perf_counter(SNRT_PERF_CNT0,SNRT_PERF_CNT_TCDM_CONGESTED,0);
        snrt_start_perf_counter(SNRT_PERF_CNT1,SNRT_PERF_CNT_ICACHE_MISS,0);
        snrt_start_perf_counter(SNRT_PERF_CNT2,SNRT_PERF_CNT_ICACHE_STALL,0);
    }
    snrt_cluster_hw_barrier();


    if(compute_grad_b){
        for(k=0;k<K_tiles;k++){        
            for(n=0;n<N_tiles;n++){
                for(m=0;m<M_tiles;m++){

                    if(compute_id==0)printf("%d\n",iter++);

                    if(snrt_is_dm_core()){
                        snrt_dma_load_2d_tile(local_A,A,m,k,frac_M,frac_K,K,sizeof(DATA_TYPE));
                        snrt_dma_load_2d_tile(local_GRAD_C,GRAD_C,m,n,frac_M,frac_N,N,sizeof(DATA_TYPE));
                        
                        snrt_dma_wait_all();
                    }      
                
                    snrt_cluster_hw_barrier();

                
                    if(!snrt_is_dm_core()){
                        snrt_mcycle();

                        c = CEIL(frac_K, compute_num);
                        lb = c * compute_id;
                        ub = MIN((c * (compute_id + 1)), frac_K);

                        mult_alpha = (m<M_tiles-1) ? 0 : 1;
                        initialize = (m==0) ? 1 : 0;

                        __backpropagation_multicore_computation_grad_B_fp64__(alpha,local_A,local_GRAD_C,local_GRAD_RES,
                                                                                frac_M,frac_N,frac_K,lb,ub,mult_alpha,initialize);

                        snrt_mcycle();
                    }
                    
                    snrt_fpu_fence();
                    snrt_cluster_hw_barrier();

                }

                if (snrt_is_dm_core()) {
                    snrt_dma_store_2d_tile(GRAD_B, local_GRAD_RES,k,n,frac_K,frac_N,N,sizeof(DATA_TYPE));

                    snrt_dma_wait_all();
                }

                snrt_cluster_hw_barrier();
            }
        }
    }

    if(compute_id==0){
        snrt_stop_perf_counter(SNRT_PERF_CNT0);
        snrt_stop_perf_counter(SNRT_PERF_CNT1);
        snrt_stop_perf_counter(SNRT_PERF_CNT2);
        printf("TCDM CONGESTED %d\n", snrt_get_perf_counter(SNRT_PERF_CNT0));
        printf("ICHACHE MISSES %d\n", snrt_get_perf_counter(SNRT_PERF_CNT1));
        printf("ICAHCE STALLS %d\n", snrt_get_perf_counter(SNRT_PERF_CNT2));
    }
    snrt_cluster_hw_barrier();


    if(compute_grad_a){
        local_GRAD_C = (DATA_TYPE *)snrt_l1_next();
        local_B = local_GRAD_C + size_GRAD_C;
        local_GRAD_RES = local_B + size_B;

        iter=0;
        for(m=0;m<M_tiles;m++){
            for(k=0;k<K_tiles;k++){
                for(n=0;n<N_tiles;n++){

                    if(compute_id==0)printf("%d\n",iter++);

                        if(snrt_is_dm_core()){
                            snrt_dma_load_2d_tile(local_GRAD_C,GRAD_C,m,n,frac_M,frac_N,N,sizeof(DATA_TYPE));
                            snrt_dma_load_2d_tile(local_B,B,k,n,frac_K,frac_N,N,sizeof(DATA_TYPE));

                            snrt_dma_wait_all();

                        }

                        snrt_cluster_hw_barrier();
                        
                        if(!snrt_is_dm_core()){
                            snrt_mcycle();

                            c = CEIL(frac_M, compute_num);
                            lb = c * compute_id;
                            ub = MIN((c * (compute_id + 1)), frac_M);

                            mult_alpha = (n<N_tiles-1) ? 0 : 1;
                            initialize = (n==0) ? 1 : 0;

                            __backpropagation_multicore_computation_grad_A_fp64__(alpha,local_GRAD_C,local_B,local_GRAD_RES,
                                                                                    frac_M,frac_N,frac_K,lb,ub,mult_alpha,initialize);

                            snrt_mcycle();
                        }

                        snrt_fpu_fence();
                        snrt_cluster_hw_barrier();
                }

                if (snrt_is_dm_core()) {
                    snrt_dma_store_2d_tile(GRAD_A, local_GRAD_RES,m,k,frac_M,frac_K,K,sizeof(DATA_TYPE));

                    snrt_dma_wait_all();
                }

                snrt_cluster_hw_barrier();

            }
        }
    }
}



void __backpropagation_multicore_computation_grad_B_fp64__(DATA_TYPE alpha, DATA_TYPE *local_A, DATA_TYPE *local_GRAD_C, DATA_TYPE *local_GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize){



    int32_t dim = ub-lb;
    int32_t unroll=8;//high contention due to the transposed first matrix.
    uint32_t i;
    if(dim>0){

        //high tcdm concurrency, I dont know how to fix it
        if(N/unroll > 0){
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
        }

        for (uint32_t k=lb; k<ub; k++) { 
            uint32_t n = 0;

            snrt_ssr_enable();

            for (uint32_t n0=0; n0<N/unroll; n0++) { 
            
                if(initialize){
                    for(i=0;i<unroll;i++)
                        local_GRAD_B[k*N + n + i]=0;
                }

                if(mult_alpha){
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
                        
                        :[ sum0 ] "+f"(local_GRAD_B[k*N + n + 0]), [ sum1 ] "+f"(local_GRAD_B[k*N + n + 1]), [ sum2 ] "+f"(local_GRAD_B[k*N + n + 2]),
                            [ sum3 ] "+f"(local_GRAD_B[k*N + n + 3]), [ sum4 ] "+f"(local_GRAD_B[k*N + n + 4]), [ sum5 ] "+f"(local_GRAD_B[k*N + n + 5]),
                            [ sum6 ] "+f"(local_GRAD_B[k*N + n + 6]), [ sum7 ] "+f"(local_GRAD_B[k*N + n + 7])
                        : [ n_frep ] "r"(M - 1), [alpha] "f"(alpha)
                        : "ft0", "ft1", "ft2");
                }else{
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
 
                        :[ sum0 ] "+f"(local_GRAD_B[k*N + n + 0]), [ sum1 ] "+f"(local_GRAD_B[k*N + n + 1]), [ sum2 ] "+f"(local_GRAD_B[k*N + n + 2]),
                            [ sum3 ] "+f"(local_GRAD_B[k*N + n + 3]), [ sum4 ] "+f"(local_GRAD_B[k*N + n + 4]), [ sum5 ] "+f"(local_GRAD_B[k*N + n + 5]),
                            [ sum6 ] "+f"(local_GRAD_B[k*N + n + 6]), [ sum7 ] "+f"(local_GRAD_B[k*N + n + 7])
                        : [ n_frep ] "r"(M - 1)
                        : "ft0", "ft1", "ft2");                    
                }
                n += unroll;         
            }

            snrt_ssr_disable();

            for (; n<N; n++) {
                double sum=(initialize) ? 0 : local_GRAD_B[k*N + n] ;
                for (uint32_t m=0; m<M; m++) {
                    sum += local_A[k + m*K] * local_GRAD_C[m*N + n];
                }
                local_GRAD_B[k*N + n] = (mult_alpha) ? alpha*sum : sum;
            }
        }

        snrt_fpu_fence();

    }

}


void __backpropagation_multicore_computation_grad_A_fp64__(DATA_TYPE alpha, DATA_TYPE *local_GRAD_C, DATA_TYPE *local_B, DATA_TYPE *local_GRAD_A,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize){
                        
    int32_t dim = ub-lb,unroll=4;
    int32_t i;
    if(dim>0){

        if(K/unroll>0){
            //prepare loop GRAD_C    
            const uint32_t ssr0_b[4] = {unroll, N, K/unroll, dim};
            const uint32_t ssr0_i[4] = {0, 8, 0, 8 * N};

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
        }

        //snrt_start_perf_counter(SNRT_PERF_CNT1,SNRT_PERF_CNT_TCDM_CONGESTED,snrt_cluster_core_idx());

        for (uint32_t m=lb; m<ub; m++) { 
            uint32_t k = 0;

            snrt_ssr_enable();

            for (uint32_t k0=0; k0<K/unroll; k0++) {

                if(initialize){
                    for(i=0;i<unroll;i++)
                        local_GRAD_A[m*K + k + i]=0;
                } 

                if(mult_alpha){
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

                        
                        :[ sum0 ] "+f"(local_GRAD_A[m*K + k + 0]), [ sum1 ] "+f"(local_GRAD_A[m*K + k + 1]), [ sum2 ] "+f"(local_GRAD_A[m*K + k + 2]),
                            [ sum3 ] "+f"(local_GRAD_A[m*K + k + 3])
                        : [ n_frep ] "r"(N - 1), [alpha] "f"(alpha)
                        : "ft0", "ft1", "ft2");

                    // Store results back
                    // local_GRAD_A[m*K + k + 0] = sum[0];
                    // local_GRAD_A[m*K + k + 1] = sum[1];
                    // local_GRAD_A[m*K + k + 2] = sum[2];
                    // local_GRAD_A[m*K + k + 3] = sum[3];
                    k += unroll;         
                }else{
                    asm volatile(
                        "frep.o %[n_frep], 4, 0, 0 \n"
                        "fmadd.d %[sum0], ft0, ft1, %[sum0] \n"
                        "fmadd.d %[sum1], ft0, ft1, %[sum1] \n"
                        "fmadd.d %[sum2], ft0, ft1, %[sum2] \n"
                        "fmadd.d %[sum3], ft0, ft1, %[sum3] \n"

                        :[ sum0 ] "+f"(local_GRAD_A[m*K + k + 0]), [ sum1 ] "+f"(local_GRAD_A[m*K + k + 1]), [ sum2 ] "+f"(local_GRAD_A[m*K + k + 2]),
                            [ sum3 ] "+f"(local_GRAD_A[m*K + k + 3])
                        : [ n_frep ] "r"(N - 1), [alpha] "f"(alpha)
                        : "ft0", "ft1", "ft2");

                    k += unroll;         
                }
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


    }
    snrt_fpu_fence();    

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
