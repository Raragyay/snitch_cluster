
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef DATA_TYPE
#define DATA_TYPE float
#endif 

dump_uint(iter, 0);


#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))


void backpropagation_baseline_one_core(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *grad_C,DATA_TYPE *grad_A,DATA_TYPE *grad_B,
                    uint32_t M, uint32_t N, uint32_t K);

void backpropagation_baseline_multicore(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K);

void backpropagation_one_core(DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *GRAD_C,DATA_TYPE *GRAD_A, DATA_TYPE *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K);

void backpropagation_multicore(void *alpha, void *A, void *B, void *GRAD_C,void *GRAD_A, void *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K, uint32_t M_tiles, uint32_t N_tiles, uint32_t K_tiles,
                    uint32_t compute_grad_a,uint32_t compute_grad_b, uint32_t dtype_size);

void __main_loop_fp64__(double* alpha_ptr, double *A, double *B, double *GRAD_C,double *GRAD_A, double *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K, uint32_t M_tiles, uint32_t N_tiles, uint32_t K_tiles,
                    uint32_t compute_grad_a,uint32_t compute_grad_b);
void __main_loop_fp32__(float* alpha_ptr, float *A, float *B, float *GRAD_C,float *GRAD_A, float *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K, uint32_t M_tiles, uint32_t N_tiles, uint32_t K_tiles,
                    uint32_t compute_grad_a,uint32_t compute_grad_b);

void __backpropagation_multicore_computation_grad_B_fp64__(double* alpha_ptr, double *local_A, double *local_GRAD_C, double *local_GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize,uint32_t setup_SSR);

void __backpropagation_multicore_computation_grad_A_fp64__(double* alpha_ptr, double *local_GRAD_C, double *local_B, double *local_GRAD_A,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize,uint32_t setup_SSR);

void __backpropagation_multicore_computation_grad_B_fp32__(float* alpha_ptr, float *local_A, float *local_GRAD_C, float *local_GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize,uint32_t setup_SSR);

void __backpropagation_multicore_computation_grad_A_fp32__(float *alpha_ptr, float *local_GRAD_C, float *local_B, float *local_GRAD_A,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize, uint32_t setup_SSR);

static inline void __attribute__((always_inline))
__clean_up_grad_A_fp64__(double *local_GRAD_A,double* local_B, double* local_GRAD_C, double alpha,
                    int32_t k, int32_t m, uint32_t K, uint32_t N, uint32_t mult_alpha, uint32_t initialize);

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
void backpropagation_multicore(void  *alpha_ptr, void *A, void *B, void *GRAD_C,void *GRAD_A, void *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K, uint32_t M_tiles, uint32_t N_tiles, uint32_t K_tiles,
                    uint32_t compute_grad_a,uint32_t compute_grad_b, uint32_t dtype_size){
    
    switch (dtype_size)
    {
    case 4:
        __main_loop_fp32__(alpha_ptr, A, B, GRAD_C, GRAD_A, GRAD_B, M, N, K, M_tiles, N_tiles, K_tiles, compute_grad_a, compute_grad_b);
        break;
    
    case 8:
        __main_loop_fp64__(alpha_ptr, A, B, GRAD_C, GRAD_A, GRAD_B, M, N, K, M_tiles, N_tiles, K_tiles, compute_grad_a, compute_grad_b);
        break;
    }

}

void __main_loop_fp64__(double* alpha_ptr, double *A, double *B, double *GRAD_C,double *GRAD_A, double *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K, uint32_t M_tiles, uint32_t N_tiles, uint32_t K_tiles,
                    uint32_t compute_grad_a,uint32_t compute_grad_b){


    uint32_t frac_M = M / M_tiles;
    uint32_t frac_N = N / N_tiles;
    uint32_t frac_K = K / K_tiles;

    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    int32_t c,lb,ub;
    uint32_t k,m,n;
    uint32_t size_A, size_B, size_GRAD_C;
    double *local_A, *local_B, *local_GRAD_C, *local_GRAD_RES; //*local_GRAD_RES used for both computations
    register uint32_t mult_alpha,initialize,setup_SSR;

    size_A = frac_M * frac_K;
    size_B = frac_N * frac_K;
    size_GRAD_C = frac_M * frac_N;
    
    snrt_cluster_hw_barrier();

    int iter=0;

    if(compute_grad_b){

        c = CEIL(frac_K, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), frac_K);

        local_A = (double *)snrt_l1_next();
        local_GRAD_C = local_A + size_A;
        local_GRAD_RES = local_GRAD_C + size_GRAD_C;

        setup_SSR = 1;
        snrt_mcycle();
        for(k=0;k<K_tiles;k++){   
            for(n=0;n<N_tiles;n++){
                for(m=0;m<M_tiles;m++){
                    if(compute_id==0) printf("iter: %d\n",iter++);
                    if(snrt_is_dm_core()){
                        snrt_dma_load_2d_tile(local_A,A,m,k,frac_M,frac_K,K, 8);
                        snrt_dma_load_2d_tile(local_GRAD_C,GRAD_C,m,n,frac_M,frac_N,N, 8);
                        snrt_dma_wait_all();
                    }      

                    snrt_cluster_hw_barrier();
                    if(!snrt_is_dm_core()){

                        //C code
                        //mult_alpha = (m<M_tiles-1) ? 0 : 1;
                        //initialize = (m==0) ? 1 : 0;
                        asm volatile(
                            "mv %[initialize], zero \n"
                            "mv %[mult_alpha], zero \n"

                            "bnez %[m], 1f \n"
                            "addi %[initialize], %[initialize], 1 \n"

                            "1:\n"
                            "bne %[m], %[M_tiles_m_1], 2f \n"
                            "addi %[mult_alpha], %[mult_alpha], 1 \n"

                            "2:\n"
                            :[mult_alpha] "=r"(mult_alpha),[initialize] "=r"(initialize)
                            :[m] "r"(m),[M_tiles_m_1] "r"(M_tiles-1)
                        :);
                        __backpropagation_multicore_computation_grad_B_fp64__(alpha_ptr,local_A,local_GRAD_C,local_GRAD_RES,
                                                        frac_M,frac_N,frac_K,lb,ub,mult_alpha,initialize,setup_SSR);                                  

                        //C code
                        //if(setup_SSR==1) setup_SSR=0;
                        asm volatile(
                            "beqz %[setup_SSR], 1f \n"
                            "mv %[setup_SSR], zero \n"
                            "1: \n"
                            :[setup_SSR] "+r"(setup_SSR)
                            :
                        :);                    
                    }
                    
                    snrt_fpu_fence();
                    snrt_cluster_hw_barrier();

                }
                if (snrt_is_dm_core()) {
                    snrt_dma_store_2d_tile(GRAD_B, local_GRAD_RES,k,n,frac_K,frac_N,N,8);
                    snrt_dma_wait_all();
                }

                snrt_cluster_hw_barrier();
            }
        }
        snrt_mcycle();

    }


    snrt_cluster_hw_barrier();

    if(compute_grad_a){

        local_GRAD_C = (double *)snrt_l1_next();
        local_B =(frac_N%32!=0) ? local_GRAD_C + size_GRAD_C : local_GRAD_C + size_GRAD_C + 4;//to make the access on different banks  
        //local_B = local_GRAD_C + size_GRAD_C;
        local_GRAD_RES = local_B + size_B;

        c = CEIL(frac_M, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), frac_M);
        setup_SSR = 1;

        snrt_mcycle();
        for(m=0;m<M_tiles;m++){
            for(k=0;k<K_tiles;k++){
                for(n=0;n<N_tiles;n++){
                    if(compute_id==0) dump_iter(iter);
                    iter++;

                    if(snrt_is_dm_core()){

                        snrt_dma_load_2d_tile(local_GRAD_C,GRAD_C,m,n,frac_M,frac_N,N,8);
                        snrt_dma_load_2d_tile(local_B,B,k,n,frac_K,frac_N,N,8);
                        snrt_dma_wait_all();

                    }

                    snrt_cluster_hw_barrier();
                    
                    if(!snrt_is_dm_core()){
                        
                        //C code
                        //mult_alpha = (n==N_tiles-1) ? 1 : 0;
                        //initialize = (n==0) ? 1 : 0;
                        asm volatile(
                            "mv %[initialize], zero \n"
                            "mv %[mult_alpha], zero \n"

                            "bnez %[n], 1f \n"
                            "addi %[initialize], %[initialize], 1 \n"

                            "1:\n"
                            "bne %[n], %[N_tiles_m_1], 2f \n"
                            "addi %[mult_alpha], %[mult_alpha], 1 \n"

                            "2:\n"
                            :[mult_alpha] "=r"(mult_alpha),[initialize] "=r"(initialize)
                            :[n] "r"(n),[N_tiles_m_1] "r"(N_tiles-1)
                        :);

                        __backpropagation_multicore_computation_grad_A_fp64__(alpha_ptr,local_GRAD_C,local_B,local_GRAD_RES,
                                                                        frac_M,frac_N,frac_K,lb,ub,mult_alpha,initialize,setup_SSR);                              
                        //C code
                        //if(setup_SSR==1) setup_SSR=0;
                        asm volatile(
                            "beqz %[setup_SSR], 1f \n"
                            "mv %[setup_SSR], zero \n"
                            "1: \n"
                        :[setup_SSR] "+r"(setup_SSR)
                        :
                        :);

                    }

                    snrt_fpu_fence();
                    snrt_cluster_hw_barrier();
                }

                if (snrt_is_dm_core()) {
                    //tiling doesn't bother write-back.
                    snrt_dma_store_2d_tile(GRAD_A, local_GRAD_RES,m,k,frac_M,frac_K,K,8);
                    snrt_dma_wait_all();
                }

                snrt_cluster_hw_barrier();

            }
        }
        snrt_mcycle();

    }

}


void __main_loop_fp32__(float* alpha_ptr, float *A, float *B, float *GRAD_C,float *GRAD_A, float *GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K, uint32_t M_tiles, uint32_t N_tiles, uint32_t K_tiles,
                    uint32_t compute_grad_a,uint32_t compute_grad_b){

    uint32_t frac_M = M / M_tiles;
    uint32_t frac_N = N / N_tiles;
    uint32_t frac_K = K / K_tiles;

    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    int32_t c,lb,ub;
    uint32_t k,m,n;
    uint32_t size_A, size_B, size_GRAD_C;
    float *local_A, *local_B, *local_GRAD_C, *local_GRAD_RES; //*local_GRAD_RES used for both computations
    register uint32_t mult_alpha,initialize,setup_SSR;

    size_A = frac_M * frac_K;
    size_B = frac_N * frac_K;
    size_GRAD_C = frac_M * frac_N;
    
    const uint32_t padding_K_req = (frac_K%2!=0) ? 1: 0;
    const uint32_t padding_N_req = (frac_N%2!=0) ? 1: 0;

    if(compute_grad_b){
        
        //If padding is required on K we just increment frac_K to be allocated for the tcdm. The last iteration
        //will iterate over one meaningful data and one useless. The useless one will not be saved back into main memory  
        if(padding_K_req){
            size_A = frac_M *(frac_K+1);
            c = CEIL(frac_K+1, compute_num);
            lb = c * compute_id;
            ub = MIN((c * (compute_id + 1)), frac_K+1);

        }else{
            c = CEIL(frac_K, compute_num);
            lb = c * compute_id;
            ub = MIN((c * (compute_id + 1)), frac_K);
        }
        if((ub-lb)%2!=0){
            lb = (lb%2==0)? lb :lb+1;
            ub = (ub%2==0)? ub :ub+1;
        }

        if(padding_N_req){
            size_GRAD_C = frac_M * (frac_N+1);

        }

        local_A = (float *)snrt_l1_next();
        local_GRAD_C = local_A + size_A;
        local_GRAD_RES = local_GRAD_C + size_GRAD_C;

        setup_SSR = 1;

        snrt_mcycle();
        for(k=0;k<K_tiles;k++){   
            for(n=0;n<N_tiles;n++){
                for(m=0;m<M_tiles;m++){

                    if(snrt_is_dm_core()){
                        if(padding_K_req)
                            snrt_dma_start_2d(local_A,A+k*frac_K+m*frac_M*K,(frac_K+1)*4,(frac_K+1)*4,K*4,frac_M);
                        else
                            snrt_dma_load_2d_tile(local_A,A,m,k,frac_M,frac_K,K,4);
                        if(padding_N_req)
                            snrt_dma_start_2d(local_GRAD_C,GRAD_C+n*frac_N+m*frac_M*N,(frac_N+1)*4,(frac_N+1)*4,N*4,frac_M);
                        else
                            snrt_dma_load_2d_tile(local_GRAD_C,GRAD_C,m,n,frac_M,frac_N,N,4);

                        snrt_dma_wait_all();
                    }      

                    snrt_cluster_hw_barrier();
                    if(!snrt_is_dm_core()){

                        //C code
                        //mult_alpha = (m<M_tiles-1) ? 0 : 1;
                        //initialize = (m==0) ? 1 : 0;
                        asm volatile(
                            "mv %[initialize], zero \n"
                            "mv %[mult_alpha], zero \n"

                            "bnez %[m], 1f \n"
                            "addi %[initialize], %[initialize], 1 \n"

                            "1:\n"
                            "bne %[m], %[M_tiles_m_1], 2f \n"
                            "addi %[mult_alpha], %[mult_alpha], 1 \n"

                            "2:\n"
                            :[mult_alpha] "=r"(mult_alpha),[initialize] "=r"(initialize)
                            :[m] "r"(m),[M_tiles_m_1] "r"(M_tiles-1)
                        :);

                        if(padding_K_req && padding_N_req){
                            __backpropagation_multicore_computation_grad_B_fp32__(alpha_ptr,local_A,local_GRAD_C,local_GRAD_RES,
                                                        frac_M,frac_N+1,frac_K+1,lb,ub,mult_alpha,initialize,setup_SSR);                           
                        }else if(padding_K_req){
                            __backpropagation_multicore_computation_grad_B_fp32__(alpha_ptr,local_A,local_GRAD_C,local_GRAD_RES,
                                                        frac_M,frac_N,frac_K+1,lb,ub,mult_alpha,initialize,setup_SSR);
                        }else if(padding_N_req){
                            __backpropagation_multicore_computation_grad_B_fp32__(alpha_ptr,local_A,local_GRAD_C,local_GRAD_RES,
                                                        frac_M,frac_N+1,frac_K,lb,ub,mult_alpha,initialize,setup_SSR);
                        }else{
                            __backpropagation_multicore_computation_grad_B_fp32__(alpha_ptr,local_A,local_GRAD_C,local_GRAD_RES,
                                                        frac_M,frac_N,frac_K,lb,ub,mult_alpha,initialize,setup_SSR);                                                       
                        }

                        //C code
                        //if(setup_SSR==1) setup_SSR=0;
                        asm volatile(
                            "beqz %[setup_SSR], 1f \n"
                            "mv %[setup_SSR], zero \n"
                            "1: \n"
                            :[setup_SSR] "+r"(setup_SSR)
                            :
                        :);                    
                    }
                    
                    snrt_fpu_fence();
                    snrt_cluster_hw_barrier();

                }
                if (snrt_is_dm_core()) {
                    if(padding_N_req){
                        snrt_dma_start_2d(GRAD_B+n*frac_N+k*frac_K*N,local_GRAD_RES,frac_N*4,N*4,(frac_N+1)*4,frac_K);
                    }else{
                        snrt_dma_store_2d_tile(GRAD_B, local_GRAD_RES,k,n,frac_K,frac_N,N,4);
                    }
                    snrt_dma_wait_all();
                }

                snrt_cluster_hw_barrier();
            }
        }
        snrt_mcycle();

    }

    snrt_cluster_hw_barrier();

    if(compute_grad_a){

        if(padding_N_req){
            frac_N +=1;
            size_GRAD_C = frac_N * frac_M;
            size_B = frac_N * frac_K;
        }
        local_GRAD_C = (float *)snrt_l1_next();
        local_B =local_GRAD_C + size_GRAD_C;
        local_GRAD_RES = local_B + size_B;

        c = CEIL(frac_M, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), frac_M);
        setup_SSR = 1;

        snrt_mcycle();
        for(m=0;m<M_tiles;m++){
            for(k=0;k<K_tiles;k++){
                for(n=0;n<N_tiles;n++){

                        if(snrt_is_dm_core()){

                            if(padding_N_req){
                                const float zero=0.0f;
                                snrt_dma_start_2d(local_GRAD_C,GRAD_C+n*(frac_N-1)+m*frac_M*N,frac_N*4,frac_N*4,N*4,frac_M);
                                snrt_dma_start_2d(local_B,B+n*(frac_N-1)+k*frac_K*N,frac_N*4,frac_N*4,N*4,frac_K);

                                snrt_dma_start_2d(local_GRAD_C+frac_N-1,&zero,4,frac_N*4,0,frac_M);
                                snrt_dma_start_2d(local_B+frac_N-1,&zero,4,frac_N*4,0,frac_K);                            
                            }else{
                                snrt_dma_load_2d_tile(local_GRAD_C,GRAD_C,m,n,frac_M,frac_N,N,4);
                                snrt_dma_load_2d_tile(local_B,B,k,n,frac_K,frac_N,N,4);
                            }
                            snrt_dma_wait_all();

                        }

                        snrt_cluster_hw_barrier();

                        if(!snrt_is_dm_core()){
                            
                            //C code
                            //mult_alpha = (n==N_tiles-1) ? 1 : 0;
                            //initialize = (n==0) ? 1 : 0;
                            asm volatile(
                                "mv %[initialize], zero \n"
                                "mv %[mult_alpha], zero \n"

                                "bnez %[n], 1f \n"
                                "addi %[initialize], %[initialize], 1 \n"

                                "1:\n"
                                "bne %[n], %[N_tiles_m_1], 2f \n"
                                "addi %[mult_alpha], %[mult_alpha], 1 \n"

                                "2:\n"
                                :[mult_alpha] "=r"(mult_alpha),[initialize] "=r"(initialize)
                                :[n] "r"(n),[N_tiles_m_1] "r"(N_tiles-1)
                            :);

                            __backpropagation_multicore_computation_grad_A_fp32__(alpha_ptr,local_GRAD_C,local_B,local_GRAD_RES,
                                                                        frac_M,frac_N,frac_K,lb,ub,mult_alpha,initialize,setup_SSR);
                
                            //C code
                            //if(setup_SSR==1) setup_SSR=0;
                            asm volatile(
                                "beqz %[setup_SSR], 1f \n"
                                "mv %[setup_SSR], zero \n"
                                "1: \n"
                            :[setup_SSR] "+r"(setup_SSR)
                            :
                            :);

                        }

                        snrt_fpu_fence();
                        snrt_cluster_hw_barrier();
                }

                if (snrt_is_dm_core()) {
                    //tiling doesn't bother write-back.
                    snrt_dma_store_2d_tile(GRAD_A, local_GRAD_RES,m,k,frac_M,frac_K,K,4);
                    snrt_dma_wait_all();
                }

                snrt_cluster_hw_barrier();

            }
        }
        snrt_mcycle();

    }

}


void __backpropagation_multicore_computation_grad_B_fp64__(double* alpha_ptr, double *local_A, double *local_GRAD_C, double *local_GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize,uint32_t setup_SSR){

    double alpha=*alpha_ptr;
    register const uint32_t compute_num = snrt_cluster_compute_core_num();
    register const uint32_t compute_id = snrt_cluster_core_idx();
    register double ZERO=0.0f;
    int32_t dim = ub-lb;
    int32_t unroll=8;//high contention due to the transposed first matrix.
    uint32_t i,index_addr;
    register uint32_t n;
    register int32_t n0; 
    const register int32_t loops= N/unroll;
    register int32_t k=compute_id;
    double sum;
    if(dim>0){

        if(loops > 0){
            if(setup_SSR==1){
                //prepare loop A transposed    
                const uint32_t ssr0_b[4] = {unroll, M, loops, dim};
                const uint32_t ssr0_i[4] = {0, 8 * K, 0, 8*compute_num };

                snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                                    ssr0_i[1], ssr0_i[2], ssr0_i[3]);
                snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
                

                //prepare loop GRAD_C
                const uint32_t ssr1_b[4] = {unroll, M, loops, dim}; 
                const uint32_t ssr1_i[4] = {8, 8 * N, 8 * unroll, 0};


                snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                                    ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                                    ssr1_i[3]);
            }
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_A+compute_id);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_GRAD_C);
        }

        //check lower bounds
        while(k<K){ 
            n=0;
            n0=0;
            snrt_ssr_enable();

            while(n0<loops){

                asm volatile(
                    "beqz %[initialize], 1f\n"
                    "fcvt.d.w %[ZERO], zero\n"
                    "fsgnj.d ft10,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft3,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft4,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft5,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft6,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft7,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft8,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft9,%[ZERO],%[ZERO]\n"
                    "j 2f\n"            

                    "1:\n"
                    "fld ft10, 0(%[sum_addr]) \n"
                    "fld ft3, 8(%[sum_addr]) \n"
                    "fld ft4, 16(%[sum_addr]) \n"
                    "fld ft5, 24(%[sum_addr]) \n"
                    "fld ft6, 32(%[sum_addr]) \n"
                    "fld ft7, 40(%[sum_addr]) \n"
                    "fld ft8, 48(%[sum_addr]) \n"
                    "fld ft9, 56(%[sum_addr]) \n"

                    "2:\n"
                    "frep.o %[n_frep], 8, 0, 0 \n"
                    "fmadd.d ft10, ft0, ft1, ft10 \n"
                    "fmadd.d ft3, ft0, ft1, ft3 \n"
                    "fmadd.d ft4, ft0, ft1, ft4 \n"
                    "fmadd.d ft5, ft0, ft1, ft5 \n"
                    "fmadd.d ft6, ft0, ft1, ft6 \n"
                    "fmadd.d ft7, ft0, ft1, ft7 \n"
                    "fmadd.d ft8, ft0, ft1, ft8 \n"
                    "fmadd.d ft9, ft0, ft1, ft9 \n"


                    "beqz %[mult_alpha], 3f \n"
                    "fmul.d ft10, %[alpha],ft10 \n"
                    "fmul.d ft3, %[alpha],ft3 \n"
                    "fmul.d ft4, %[alpha],ft4 \n"
                    "fmul.d ft5, %[alpha],ft5 \n"
                    "fmul.d ft6, %[alpha],ft6 \n"
                    "fmul.d ft7, %[alpha],ft7 \n"
                    "fmul.d ft8, %[alpha],ft8 \n"
                    "fmul.d ft9, %[alpha],ft9 \n"

                    "3: \n"
                    "fsd ft10, 0(%[sum_addr]) \n"
                    "fsd ft3, 8(%[sum_addr]) \n"
                    "fsd ft4, 16(%[sum_addr]) \n"
                    "fsd ft5, 24(%[sum_addr]) \n"
                    "fsd ft6, 32(%[sum_addr]) \n"
                    "fsd ft7, 40(%[sum_addr]) \n"
                    "fsd ft8, 48(%[sum_addr]) \n"
                    "fsd ft9, 56(%[sum_addr]) \n"  

                    "addi %[n0],%[n0],1 \n"
                    "addi %[n], %[n],8 \n"//if unroll changes, change this

                    : [n0] "+r"(n0), [n] "+r"(n)
                    :[ sum_addr ] "r"(local_GRAD_B+ k*N +n), [ n_frep ] "r"(M - 1), [alpha] "f"(alpha), [initialize] "r"(initialize), [ZERO] "fr"(ZERO),
                    [mult_alpha] "r"(mult_alpha)
                    : "ft0", "ft1", "ft10","ft3","ft5","ft6","ft7","ft8","ft9","ft2");
                
            }

            snrt_ssr_disable();

           //cleanup sucks, make it better
            for (; n<N; n++) {
                sum=(initialize) ? 0 : local_GRAD_B[k*N + n] ;
                for (uint32_t m=0; m<M; m++) {
                    sum += local_A[k + m*K] * local_GRAD_C[m*N + n];
                }
                local_GRAD_B[k*N + n] = (mult_alpha) ? alpha*sum : sum;
            }

        k+=compute_num;
        }
    }
}

void __backpropagation_multicore_computation_grad_A_fp64__(double* alpha_ptr, double *local_GRAD_C, double *local_B, double *local_GRAD_A,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize,uint32_t setup_SSR){

    double alpha = *alpha_ptr;
    int32_t dim = ub-lb,unroll=4;
    int32_t i,k,m,k0;
    const register int32_t loops=K/unroll;
    register double ZERO=0.0f;

    if(dim>0){

        if(loops>0){
            if(setup_SSR==1){
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

            }
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_GRAD_C + lb*N);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_B);
    
        }


        m=lb;
        while(m<ub){
            k=0;
            k0=0;
            snrt_ssr_enable();

            while(k0<loops){

                asm volatile(
                    "beqz %[initialize], 1f\n"
                    "fcvt.d.w %[ZERO], zero\n"
                    "fsgnj.d ft3,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft4,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft5,%[ZERO],%[ZERO]\n"
                    "fsgnj.d ft6,%[ZERO],%[ZERO]\n"
                    "j 2f\n"            

                    "1:\n"
                    "fld ft3, 0(%[sum_addr]) \n"
                    "fld ft4, 0(%[sum_addr]) \n"
                    "fld ft5, 0(%[sum_addr]) \n"
                    "fld ft6, 0(%[sum_addr]) \n"

                    "2:\n"
                    "frep.o %[n_frep], 4, 0, 0 \n"
                    "fmadd.d ft3, ft0, ft1, ft3 \n"
                    "fmadd.d ft4, ft0, ft1, ft4 \n"
                    "fmadd.d ft5, ft0, ft1, ft5 \n"
                    "fmadd.d ft6, ft0, ft1, ft6 \n"
                    

                    "beqz %[mult_alpha], 3f \n"
                    "fmul.d ft3, %[alpha],ft3 \n"
                    "fmul.d ft4, %[alpha],ft4 \n"
                    "fmul.d ft5, %[alpha],ft5 \n"
                    "fmul.d ft6, %[alpha],ft6 \n"

                    "3: \n"
                    "fsd ft3, 0(%[sum_addr]) \n"
                    "fsd ft4, 8(%[sum_addr]) \n"
                    "fsd ft5, 16(%[sum_addr]) \n"
                    "fsd ft6, 24(%[sum_addr]) \n"
                    

                    "addi %[k0],%[k0],1 \n"
                    "addi %[k], %[k],4 \n"//if unroll changes, change this
                    
                    : [k0] "+r"(k0), [k] "+r"(k)
                    : [ n_frep ] "r"(N - 1), [alpha] "f"(alpha), [initialize] "r"(initialize), [mult_alpha] "r"(mult_alpha), [ZERO] "fr"(ZERO),
                    [ sum_addr ] "r"(local_GRAD_A + m*K + k)
                    : "ft0", "ft1", "ft2","ft3","ft4","ft5","ft6");

            }
            snrt_ssr_disable();
            snrt_fpu_fence();

            __clean_up_grad_A_fp64__(local_GRAD_A,local_B,local_GRAD_C,alpha,k,m,K,N,mult_alpha,initialize);
            
            m++;
        }

    }
    snrt_fpu_fence();    

}


void __backpropagation_multicore_computation_grad_B_fp32__(float* alpha_ptr, float *local_A, float *local_GRAD_C, float *local_GRAD_B,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize,uint32_t setup_SSR){

        float alpha= *alpha_ptr;
    register float ZERO=0.0f;
    int32_t dim = ub-lb;

    int32_t unroll=8;
    uint32_t i,index_addr;
    register uint32_t n;
    register int32_t n0; 
    const register int32_t loops= N/unroll;
    register int32_t k=lb;
    volatile float sum_1,sum_2;
    if(dim>0){

        if(loops > 0){
            if(setup_SSR==1){
                //prepare loop A transposed    
                const uint32_t ssr0_b[4] = {unroll/2, M, loops, dim/2};
                const uint32_t ssr0_i[4] = {0, 4 * K, 0, 4 * 2 };

                snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                                    ssr0_i[1], ssr0_i[2], ssr0_i[3]);
                snrt_ssr_repeat(SNRT_SSR_DM0, 2);
                

                //dont work
                //prepare loop GRAD_C
                const uint32_t ssr1_b[4] = {unroll/2, M, loops, dim/2}; 
                const uint32_t ssr1_i[4] = {8,4*N ,4 *  unroll, 0};


                snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                                    ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                                    ssr1_i[3]);
                snrt_ssr_repeat(SNRT_SSR_DM1, 2);

            }
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_A+lb);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_GRAD_C);
        }

        //check lower bounds
        while(k<ub){ 
            n=0;
            n0=0;
            snrt_ssr_enable();
            while(n0<loops){

                asm volatile(
                    //load first values of A and de-couple them
                    "vfcpka.s.s ft10,ft0,ft0\n"// a reg with the first two value repeated twice 
                    "vfcpka.s.s ft11,%[ZERO],%[ZERO] \n"
                    "vfsum.s ft11,ft0 \n"
                    "vfsub.r.s ft11, ft10, ft11 \n"

                    "beqz %[initialize], 1f \n"   
                    "vfmul.s ft3, ft10, ft1 \n"
                    "vfmul.s ft4, ft11, ft1 \n"
                    "vfmul.s ft5, ft10, ft1 \n"
                    "vfmul.s ft6, ft11, ft1 \n"
                    "vfmul.s ft7, ft10, ft1 \n"
                    "vfmul.s ft8, ft11, ft1 \n"
                    "vfmul.s ft9, ft10, ft1 \n"
                    "vfmul.s fa0, ft11, ft1 \n"
                    "j 2f\n"            
                    //load previous data and operate the mac on it
                    "1:\n"
                    "fld ft3,  0(%[sum_addr_0]) \n"
                    "fld ft4,  0(%[sum_addr_1]) \n"
                    "fld ft5,  8(%[sum_addr_0]) \n"
                    "fld ft6,  8(%[sum_addr_1]) \n"
                    "fld ft7, 16(%[sum_addr_0]) \n"
                    "fld ft8, 16(%[sum_addr_1]) \n"
                    "fld ft9, 24(%[sum_addr_0]) \n"
                    "fld fa0, 24(%[sum_addr_1]) \n"

                    "vfmac.s ft3, ft10, ft1 \n"
                    "vfmac.s ft4, ft11, ft1 \n"
                    "vfmac.s ft5, ft10, ft1 \n"
                    "vfmac.s ft6, ft11, ft1 \n"
                    "vfmac.s ft7, ft10, ft1 \n"
                    "vfmac.s ft8, ft11, ft1 \n"
                    "vfmac.s ft9, ft10, ft1 \n"
                    "vfmac.s fa0, ft11, ft1 \n"
                  
                    "2:\n"                   
                    "frep.o %[n_frep],12 , 0, 0 \n"
                    "vfcpka.s.s ft11,%[ZERO],%[ZERO] \n"
                    "vfcpka.s.s ft10,ft0,ft0\n"
                    "vfsum.s ft11,ft0 \n"
                    "vfsub.r.s ft11, ft10, ft11 \n"
                    
                    "vfmac.s ft3, ft10, ft1 \n"
                    "vfmac.s ft4, ft11, ft1 \n"
                    "vfmac.s ft5, ft10, ft1 \n"
                    "vfmac.s ft6, ft11, ft1 \n"
                    "vfmac.s ft7, ft10, ft1 \n"
                    "vfmac.s ft8, ft11, ft1 \n"
                    "vfmac.s ft9, ft10, ft1 \n"
                    "vfmac.s fa0, ft11, ft1 \n"
                  
                    

                    "beqz %[mult_alpha], 3f \n"
                    "vfmul.r.s ft3, ft3, %[alpha] \n"
                    "vfmul.r.s ft4, ft4, %[neg_alpha] \n"
                    "vfmul.r.s ft5, ft5, %[alpha] \n"
                    "vfmul.r.s ft6, ft6, %[neg_alpha] \n"
                    "vfmul.r.s ft7, ft7, %[alpha] \n"
                    "vfmul.r.s ft8, ft8, %[neg_alpha] \n"
                    "vfmul.r.s ft9, ft9, %[alpha] \n"
                    "vfmul.r.s fa0, fa0, %[neg_alpha] \n"

                    "3: \n"
                    "fsd ft3, 0(%[sum_addr_0]) \n"
                    "fsd ft5, 8(%[sum_addr_0]) \n"
                    "fsd ft7, 16(%[sum_addr_0]) \n"
                    "fsd ft9, 24(%[sum_addr_0]) \n"

                    "fsd ft4,  0(%[sum_addr_1]) \n"
                    "fsd ft6,  8(%[sum_addr_1]) \n"
                    "fsd ft8, 16(%[sum_addr_1]) \n"
                    "fsd fa0, 24(%[sum_addr_1]) \n"

                   
                    "addi %[n0],%[n0],1 \n"
                    "addi %[n], %[n],8 \n"//if unroll changes, change this

                    : [n0] "+r"(n0), [n] "+r"(n)
                    :[ sum_addr_0 ] "r"(local_GRAD_B+ k*N +n),[ sum_addr_1 ] "r"(local_GRAD_B+ (k+1)*N +n), [ n_frep ] "r"(M - 2), [alpha] "f"(alpha),
                     [neg_alpha] "f"(-alpha), [initialize] "r"(initialize), [ZERO] "fr"(ZERO), [mult_alpha] "r"(mult_alpha)
                    : "ft0", "ft1", "ft2","ft3","ft4","ft5","ft6","ft7","ft8","ft9","ft10","ft11","fa0");

            }
            snrt_ssr_disable();

           //cleanup sucks, make it better
            for (; n<N; n++) {
                sum_1=(initialize) ? 0 : local_GRAD_B[k*N + n] ;
                sum_2=(initialize) ? 0 : local_GRAD_B[(k+1)*N + n] ;

                for (uint32_t m=0; m<M; m++) {
                    sum_1 += local_A[k + m*K] * local_GRAD_C[m*N + n];
                    sum_2 += local_A[(k+1)  + m*K] * local_GRAD_C[m*N + n];

                }
                local_GRAD_B[k*N + n] = (mult_alpha) ? alpha*sum_1 : sum_1;
                local_GRAD_B[(k+1)*N + n] = (mult_alpha) ? alpha*sum_2 : sum_2;

            }

            k+=2;
        }
    }

}


void __backpropagation_multicore_computation_grad_A_fp32__(float *alpha_ptr, float *local_GRAD_C, float *local_B, float *local_GRAD_A,
                    uint32_t M, uint32_t N, uint32_t K,int32_t lb,int32_t ub, uint32_t mult_alpha, uint32_t initialize, uint32_t setup_SSR){
    float alpha=*alpha_ptr;
    int32_t dim = ub-lb,unroll=4;
    int32_t i,k,m,k0;
    const register int32_t loops=K/unroll;
    register float ZERO=0.0f;
    register const uint32_t compute_id = snrt_cluster_core_idx();
    volatile float sum;
    uint32_t n;
    if(dim>0){

        if(loops>0){
            if(setup_SSR==1){

                //prepare loop GRAD_C    
                const uint32_t ssr0_b[4] = {unroll, N/2, K/unroll, dim};
                const uint32_t ssr0_i[4] = {0, 8, 0, 4 * N};

                snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                                    ssr0_i[1], ssr0_i[2], ssr0_i[3]);
                snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
                
                //prepare loop B TRANSPOSED 
                const uint32_t ssr1_b[4] = {unroll, N/2, K / unroll, dim};
                const uint32_t ssr1_i[4] = {4*N, 8, N*unroll*4, 0};

                snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                                    ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                                    ssr1_i[3]);
                snrt_ssr_repeat(SNRT_SSR_DM1, 1);


            }
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_GRAD_C + lb*N);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_B);
    
        }


        m=lb;
        while(m<ub){
            k=0;
            k0=0;
            snrt_ssr_enable();

            while(k0<loops){
                asm volatile(
                    "beqz %[initialize], 1f\n"
                    "fcvt.d.w %[ZERO], zero\n"
                    "vfcpka.s.s ft7, %[ZERO], %[ZERO]\n"
                    "vfcpka.s.s ft8, %[ZERO], %[ZERO]\n"
                    "vfcpka.s.s ft9, %[ZERO], %[ZERO]\n"
                    "vfcpka.s.s ft10, %[ZERO], %[ZERO]\n"
                    "j 2f\n"            

                    "1:\n"
                    "flw ft3, 0(%[sum_addr]) \n"
                    "flw ft4, 4(%[sum_addr]) \n"
                    "flw ft5, 8(%[sum_addr]) \n"
                    "flw ft6, 12(%[sum_addr]) \n"
                    "vfcpka.s.s ft7, ft3, %[ZERO]\n"
                    "vfcpka.s.s ft8, ft4, %[ZERO]\n"
                    "vfcpka.s.s ft9, ft5, %[ZERO]\n"
                    "vfcpka.s.s ft10,ft6, %[ZERO]\n"


                    "2:\n"
                    "vfmul.s ft3, ft0, ft1\n"
                    "vfmul.s ft4, ft0, ft1\n"
                    "vfmul.s ft5, ft0, ft1\n"
                    "vfmul.s ft6, ft0, ft1\n"
                    "frep.o %[n_frep], 4, 0, 0 \n"
                    "vfmac.s ft3, ft0, ft1 \n"
                    "vfmac.s ft4, ft0, ft1 \n"
                    "vfmac.s ft5, ft0, ft1 \n"
                    "vfmac.s ft6, ft0, ft1 \n"

                    "vfsum.s ft7,ft3\n"
                    "vfsum.s ft8,ft4\n"
                    "vfsum.s ft9,ft5\n"
                    "vfsum.s ft10,ft6\n"

                    "beqz %[mult_alpha], 3f \n"
                    "fmul.s ft7, %[alpha],ft7 \n"
                    "fmul.s ft8, %[alpha],ft8 \n"
                    "fmul.s ft9, %[alpha],ft9 \n"
                    "fmul.s ft10, %[alpha],ft10 \n"

                    "3: \n"
                    "fsw ft7, 0(%[sum_addr]) \n"
                    "fsw ft8, 4(%[sum_addr]) \n"
                    "fsw ft9, 8(%[sum_addr]) \n"
                    "fsw ft10, 12(%[sum_addr]) \n"
                    

                    "addi %[k0],%[k0],1 \n"
                    "addi %[k], %[k],4 \n"//if unroll changes, change this
                    
                    : [k0] "+r"(k0), [k] "+r"(k)
                    : [ n_frep ] "r"(N/2 - 2), [alpha] "f"(alpha), [initialize] "r"(initialize), [mult_alpha] "r"(mult_alpha), [ZERO] "fr"(ZERO),
                    [ sum_addr ] "r"(local_GRAD_A + m*K + k)
                    : "ft0", "ft1", "ft2","ft3","ft4","ft5","ft6","ft7","ft8","ft9","ft10");

            }

            snrt_ssr_disable();

            //cleanup sucks
            for (; k<K; k++) {
               
                sum=(initialize) ? 0 : local_GRAD_A[m*K + k];
                for (n=0; n<N; n++) {
                    sum +=  local_GRAD_C[m*N+n]*local_B[n+k*N];
                }
                local_GRAD_A[m*K + k] = (mult_alpha) ? alpha*sum : sum;
                snrt_fpu_fence();//otherwise not saved correctly
            }            
            m++;
        }

    }
    snrt_fpu_fence();    

}

static inline void __attribute__((always_inline))
__clean_up_grad_A_fp64__(double *local_GRAD_A,double* local_B, double* local_GRAD_C, double alpha,
                    int32_t k, int32_t m, uint32_t K, uint32_t N, uint32_t mult_alpha, uint32_t initialize){
    //C code
    double sum;
    for (; k<K; k++) {
        sum=(initialize) ? 0 : local_GRAD_A[m*K + k] ;
        for (uint32_t n=0; n<N; n++) {
            sum +=  local_GRAD_C[m*N+n]*local_B[n+k*N];
        }
        local_GRAD_A[m*K + k] =(mult_alpha)? alpha*sum :sum;
    }
    // register double ZERO=0.0f;
    // switch (K-k){
    // case 3:
    //     asm volatile(
    //         "beqz %[initialize], 1f\n"
    //         "fcvt.d.w %[ZERO], zero\n"
    //         "fsgnj.d ft3,%[ZERO],%[ZERO]\n"
    //         "fsgnj.d ft4,%[ZERO],%[ZERO]\n"
    //         "fsgnj.d ft5,%[ZERO],%[ZERO]\n"
    //         "mv t0, zero\n" //t0 is n
    //         "j 2f\n"            

    //         "1:\n"
    //         "fld ft3, 0(%[local_GRAD_A]) \n" //ft3,4,5 are sum
    //         "fld ft4, 8(%[local_GRAD_A]) \n"
    //         "fld ft5, 16(%[local_GRAD_A]) \n"
    //         "mv t0, zero\n" //t0 is n
            
    //         "2:\n"
    //         "fld ft6, 0(%[local_GRAD_C]) \n" //ft6 is the local_grad_c value                
    //         "mul t1, %[k], %[N] \n"
    //         "add t1, t1, t0 \n"
    //         "add t1, t1, %[local_B] \n"
    //         "fld ft7, 0(t1)\n" //ft7,8,9 contains local_B values 
    //         "add t1, t1, %[N] \n"
    //         "fld ft8, 0(t1)\n"
    //         "add t1, t1, %[N] \n"
    //         "fld ft9, 0(t1)\n"

    //         "fmadd.d ft3, ft6, ft7, ft3 \n"
    //         "fmadd.d ft4, ft6, ft8, ft4 \n"
    //         "fmadd.d ft5, ft6, ft9, ft5 \n"

    //         "addi t0, t0, 8 \n"
    //         "addi %[local_GRAD_C],%[local_GRAD_C],8 \n"
    //         "bne t0, %[N], 2b\n"
            
    //         "beqz %[mult_alpha], 3f \n"
    //         "fmul.d ft3, %[alpha],ft3 \n"
    //         "fmul.d ft4, %[alpha],ft4 \n"
    //         "fmul.d ft5, %[alpha],ft5 \n"

    //         "3:\n"
    //         "fsd ft3, 0(%[local_GRAD_A]) \n"
    //         "fsd ft4, 8(%[local_GRAD_A]) \n"
    //         "fsd ft5, 16(%[local_GRAD_A]) \n"
            

    //         "4:\n"
    //         :
    //         :[local_GRAD_A] "r"(local_GRAD_A + m*K + 0),[local_B] "r"(local_B), [local_GRAD_C] "r"(local_GRAD_C + m*N), [k] "r"(k), [K] "r"(K),
    //         [initialize] "r"(initialize), [mult_alpha] "r"(mult_alpha), [N] "r"(8*N), [ZERO] "fr"(ZERO), [alpha] "f"(alpha)
    //         :"t0","t1","ft3","ft4","ft5","ft6","ft7","ft8","ft9");
    //     break;
    // case 2:
    //     asm volatile(
    //         "beqz %[initialize], 1f\n"
    //         "fcvt.d.w %[ZERO], zero\n"
    //         "fsgnj.d ft3,%[ZERO],%[ZERO]\n"
    //         "fsgnj.d ft4,%[ZERO],%[ZERO]\n"
    //         "mv t0, zero\n" //t0 is n
    //         "j 2f\n"            

    //         "1:\n"
    //         "fld ft3, 0(%[local_GRAD_A]) \n" //ft3,4,5 are sum
    //         "fld ft4, 8(%[local_GRAD_A]) \n"
    //         "mv t0, zero\n" //t0 is n
            
    //         "2:\n"
    //         "fld ft6, 0(%[local_GRAD_C]) \n" //ft6 is the local_grad_c value                
    //         "mul t1, %[k], %[N] \n"
    //         "add t1, t1, t0 \n"
    //         "add t1, t1, %[local_B] \n"
    //         "fld ft7, 0(t1)\n" //ft7,8 contains local_B values 
    //         "add t1, t1, %[N] \n"
    //         "fld ft8, 0(t1)\n"
            
    //         "fmadd.d ft3, ft6, ft7, ft3 \n"
    //         "fmadd.d ft4, ft6, ft8, ft4 \n"

    //         "addi t0, t0, 8 \n"
    //         "addi %[local_GRAD_C],%[local_GRAD_C],8 \n"
    //         "bne t0, %[N], 2b\n"
            
    //         "beqz %[mult_alpha], 3f \n"
    //         "fmul.d ft3, %[alpha],ft3 \n"
    //         "fmul.d ft4, %[alpha],ft4 \n"

    //         "3:\n"
    //         "fsd ft3, 0(%[local_GRAD_A]) \n"
    //         "fsd ft4, 8(%[local_GRAD_A]) \n"
            
    //         "4:\n"
    //         :
    //         :[local_GRAD_A] "r"(local_GRAD_A + m*K + 0),[local_B] "r"(local_B), [local_GRAD_C] "r"(local_GRAD_C + m*N), [k] "r"(k), [K] "r"(K),
    //         [initialize] "r"(initialize), [mult_alpha] "r"(mult_alpha), [N] "r"(8*N), [ZERO] "fr"(ZERO), [alpha] "f"(alpha)
    //         :"t0","t1","ft3","ft4","ft6","ft7","ft8");
    //     break;
    // case 1:
    //     asm volatile(
    //         "beqz %[initialize], 1f\n"
    //         "fcvt.d.w %[ZERO], zero\n"
    //         "fsgnj.d ft3,%[ZERO],%[ZERO]\n"
    //         "mv t0, zero\n" //t0 is n
    //         "j 2f\n"            

    //         "1:\n"
    //         "fld ft3, 0(%[local_GRAD_A]) \n" //ft3,4,5 are sum
    //         "mv t0, zero\n" //t0 is n
            
    //         "2:\n"
    //         "fld ft6, 0(%[local_GRAD_C]) \n" //ft6 is the local_grad_c value                
    //         "mul t1, %[k], %[N] \n"
    //         "add t1, t1, t0 \n"
    //         "add t1, t1, %[local_B] \n"
    //         "fld ft7, 0(t1)\n" //ft7,8 contains local_B values 
            
    //         "fmadd.d ft3, ft6, ft7, ft3 \n"

    //         "addi t0, t0, 8 \n"
    //         "addi %[local_GRAD_C],%[local_GRAD_C],8 \n"
    //         "bne t0, %[N], 2b\n"
            
    //         "beqz %[mult_alpha], 3f \n"
    //         "fmul.d ft3, %[alpha],ft3 \n"

    //         "3:\n"
    //         "fsd ft3, 0(%[local_GRAD_A]) \n"
            
    //         "4:\n"
    //         :
    //         :[local_GRAD_A] "r"(local_GRAD_A + m*K + k),[local_B] "r"(local_B), [local_GRAD_C] "r"(local_GRAD_C + m*N), [k] "r"(k), [K] "r"(K),
    //         [initialize] "r"(initialize), [mult_alpha] "r"(mult_alpha), [N] "r"(8*N), [ZERO] "fr"(ZERO), [alpha] "f"(alpha)
    //         :"t0","t1","ft3","ft4","ft6","ft7","ft8");
    //     break;                        
    // }

}