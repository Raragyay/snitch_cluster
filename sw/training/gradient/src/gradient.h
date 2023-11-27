
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

void backpropagation_baseline_one_core(DATA_TYPE *I, DATA_TYPE *E, DATA_TYPE *grad_W, uint32_t M, uint32_t N, uint32_t K);

void backpropagation_baseline_multicore(DATA_TYPE *I, DATA_TYPE *E, DATA_TYPE *grad_W, uint32_t M, uint32_t N, uint32_t K);

void backpropagation_one_core(DATA_TYPE *I, DATA_TYPE *E, DATA_TYPE *grad_W, uint32_t M, uint32_t N, uint32_t K);

void __backpropagation_multicore_computation_fp64__(DATA_TYPE *local, DATA_TYPE *local_grad,uint32_t dim);

void __backpropagation_multicore_computation_fp32__(DATA_TYPE *local_i, DATA_TYPE *local_w, DATA_TYPE *local_b, DATA_TYPE *local_e,
             double e, int n, int k, uint32_t do_bias);

static inline uint64_t asuint(float f);
static inline float asfloat(uint32_t i);


// I[M][k]
// E[M][N]
// grad_W[K][N]
void backpropagation_baseline_one_core(DATA_TYPE *I, DATA_TYPE *E, DATA_TYPE *grad_W, uint32_t M, uint32_t N, uint32_t K){

    int i,j,z;
    DATA_TYPE sum;
    uint32_t size_I, size_E;
    DATA_TYPE *local_I, *local_E, *local_grad_W;

    size_I = M * K;
    size_E = M * N ;

    local_I = (DATA_TYPE *)snrt_l1_next();
    local_E = local_I + size_I;
    local_grad_W = local_E + size_E;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_I, I, M*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_E, E, M*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();


    if(snrt_cluster_core_idx()==0){
        snrt_mcycle();

        for(i=0;i<K;i++){
            for(j=0;j<N;j++){
                sum = 0;
                for(z=0;z<M;z++){
                    sum += local_I[z*K+i] * local_E[z*N+j];
                }
                local_grad_W [i*N+j] = sum;
            }
        }
        snrt_mcycle();

    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(grad_W, local_grad_W, K*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

}


//works only without tiling
void backpropagation_baseline_multicore(DATA_TYPE *I, DATA_TYPE *E, DATA_TYPE *grad_W, uint32_t M, uint32_t N, uint32_t K){

    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    int32_t i,c,lb,ub,j,z;
    uint32_t size_I, size_E;
    DATA_TYPE *local_I, *local_E, *local_grad_W;
    DATA_TYPE sum;

    size_I = M * K;
    size_E = M * N ;

    local_I = (DATA_TYPE *)snrt_l1_next();
    local_E = local_I + size_I;
    local_grad_W = local_E + size_E;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_I, I, M*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_E, E, M*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();


    if(!snrt_is_dm_core()){

        c = CEIL(K, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), K);

        for(i=lb;i<ub;i++){
            for(j=0;j<N;j++){
                sum = 0;
                for(z=0;z<M;z++){
                    sum += local_I[z*K+i] * local_E[z*N+j];
                }
                local_grad_W [i*N+j] = sum;
            }
        }
        
        snrt_fpu_fence();
    
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(grad_W, local_grad_W, K*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();        
}



void backpropagation_one_core(DATA_TYPE *I, DATA_TYPE *E, DATA_TYPE *grad_W, uint32_t M, uint32_t N, uint32_t K){
    int32_t i,c,lb,ub,j,z;
    uint32_t size_I, size_E;
    DATA_TYPE *local_I, *local_E, *local_grad_W;
    DATA_TYPE sum;

    size_I = M * K;
    size_E = M * N ;

    local_I = (DATA_TYPE *)snrt_l1_next();
    local_E = local_I + size_I;
    local_grad_W = local_E + size_E;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_I, I, M*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_E, E, M*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    uint32_t unroll=8;
    if(snrt_cluster_core_idx()==0){
        //prepare loop I transposed    
        const uint32_t ssr0_b[4] = {unroll, K, N / unroll, M};
        const uint32_t ssr0_i[4] = {0, 8 * M, 0, 8 * 8};

        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3],
                            ssr0_i[1], ssr0_i[2], ssr0_i[3]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);

        const uint32_t ssr1_b[4] = {unroll, K, N / unroll, M};
        const uint32_t ssr1_i[4] = {8, 8 * N, 8 * unroll, 0};

        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                            ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                            ssr1_i[3]);

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, local_I);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, local_E);

        
        for (uint32_t m = 0; m < M; m++) {
        uint32_t n = 0;
            for (uint32_t n0 = 0; n0 < N / unroll; n0++) {
                double c[]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
                snrt_ssr_enable();
                asm volatile(
                    "frep.o %[n_frep], 8, 0, 0 \n"
                    "fmadd.d %[c0], ft0, ft1, %[c0] \n"
                    "fmadd.d %[c1], ft0, ft1, %[c1] \n"
                    "fmadd.d %[c2], ft0, ft1, %[c2] \n"
                    "fmadd.d %[c3], ft0, ft1, %[c3] \n"
                    "fmadd.d %[c4], ft0, ft1, %[c4] \n"
                    "fmadd.d %[c5], ft0, ft1, %[c5] \n"
                    "fmadd.d %[c6], ft0, ft1, %[c6] \n"
                    "fmadd.d %[c7], ft0, ft1, %[c7] \n"
                    : [ c0 ] "+f"(c[0]), [ c1 ] "+f"(c[1]), [ c2 ] "+f"(c[2]),
                    [ c3 ] "+f"(c[3]), [ c4 ] "+f"(c[4]), [ c5 ] "+f"(c[5]),
                    [ c6 ] "+f"(c[6]), [ c7 ] "+f"(c[7])
                    : [ n_frep ] "r"(K - 1)
                    : "ft0", "ft1", "ft2");

                // Store results back
                local_grad_W[m * N + n + 0] = c[0];
                local_grad_W[m * N + n + 1] = c[1];
                local_grad_W[m * N + n + 2] = c[2];
                local_grad_W[m * N + n + 3] = c[3];
                local_grad_W[m * N + n + 4] = c[4];
                local_grad_W[m * N + n + 5] = c[5];
                local_grad_W[m * N + n + 6] = c[6];
                local_grad_W[m * N + n + 7] = c[7];
                n += unroll;
            }
        }
        snrt_fpu_fence();
        snrt_ssr_disable();

    }


    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(grad_W, local_grad_W, K*N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();   

}



void backpropagation_multicore(DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *W_grad, DATA_TYPE *B_grad,
             uint32_t N, uint32_t K,uint32_t dtype_size){
    int i;
    uint32_t  size_b, size_w;
    DATA_TYPE *local, *local_grad;
    uint32_t n_iter, size_iter, size_to_proc;

    //# tot di valori = 2*(N*1)*K
    n_iter = (2*(N+1)*K % MAX_DIM==0) ? 2*(N+1)*K/MAX_DIM : 2*(N+1)*K/MAX_DIM + 1;
    num_iter = n_iter;
    n_iter = 2;

    //in futuro cambia per double buffering
    size_iter = MAX_DIM/2;

    local = (DATA_TYPE *)snrt_l1_next();
    local_grad = local+ size_iter;

    //works but I dont like it. Use a different method on which there is an incrementer, and loads W until finished, then B
    for(i=0;i<n_iter;i++){  


        if((i+1)*size_iter<=N*K){
            if (snrt_is_dm_core()) {
                //move as much weight as it can
                snrt_dma_start_1d(local, W + size_iter*i, size_iter*sizeof(DATA_TYPE));
                snrt_dma_start_1d(local_grad ,W_grad + size_iter*i, size_iter*sizeof(DATA_TYPE));
                snrt_dma_wait_all();

            }
            size_to_proc = size_iter;
        }else{
            if(snrt_is_dm_core()){
                //otherwise move all remaining weight + move the bias. Only all if there is space
                snrt_dma_start_1d(local, W+size_iter*i, (N*K-size_iter*i)*sizeof(DATA_TYPE));
                snrt_dma_start_1d(local_grad, W_grad+size_iter*i, (N*K-size_iter*i)*sizeof(DATA_TYPE));
                //change that we need to check that B overfit the remaining dimension
                snrt_dma_start_1d((local + (N*K-size_iter*i)), B, N*sizeof(DATA_TYPE));
                snrt_dma_start_1d((local_grad + (N*K-size_iter*i)), B_grad, N*sizeof(DATA_TYPE));
                snrt_dma_wait_all();
            }
            size_to_proc = (N*K-size_iter*i) +N;
        }

            
        

        snrt_cluster_hw_barrier();
        snrt_mcycle();


        if(!snrt_is_dm_core()){
            if(dtype_size==8) __backpropagation_multicore_computation_fp64__(local,local_grad,size_to_proc);
        
        }

        snrt_cluster_hw_barrier();
     
        if (snrt_is_dm_core()) {

            if((i+1)*size_iter<=N*K){
                snrt_dma_start_1d( W + size_iter*i,local, size_to_proc*sizeof(DATA_TYPE));
                snrt_dma_wait_all();
            }else{
                snrt_dma_start_1d(W+size_iter*i, local, (N*K-size_iter*i)*sizeof(DATA_TYPE));
                //change that we need to check that B overfit the remaining dimension
                snrt_dma_start_1d(B,local + (N*K-size_iter*i), (size_to_proc-N*K+size_iter*i)*sizeof(DATA_TYPE));
            }

        snrt_dma_wait_all();
        
        }

        snrt_cluster_hw_barrier();
     
    }
}


// I[k] inputs
// W[k][n] weights
// B[n] biases
// E[n] error
// e learning rate
// void backpropagation_multicore(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
//              double e,int N, int K,int prec){
//     float div;
//     uint32_t fix_tiling_size_k,new_tiling_size_k;
//     uint32_t fix_tiling_size_n, new_tiling_size_n; 
//     uint32_t size_i, size_w, size_b, size_e;
//     void *local_I, *local_W, *local_B, *local_E;
//     void *remote_B,*remote_I,*remote_E,*remote_W;
//     uint32_t ub_i, ub_j,i ,j;
//     //find best values for fix sizes of tiling
//     if(K>=N){
//         div =(float) (K/N); //K=div*N
//         //K*N+k+n<=MAX_DIM ---> div*N*N+(div+1)*N-MAX_DIM == 0 ----> N =(-(div+1)+sqrt((div+1)*(div+1)-4*div*(-MAX_DIM))) / 2*div
//         fix_tiling_size_n = ((uint32_t)(-(div+1.0f)+(float)sqrtf((div+1.0f)*(div+1.0f)-4.0f*div*(-MAX_DIM)))/(uint32_t)(2*div));
//         fix_tiling_size_k = fix_tiling_size_n * (uint32_t)div;
//     }else{
//         div =(float) (N/K); 
//         fix_tiling_size_k = ((uint32_t)(-(div+1.0f)+(float)sqrtf((div+1.0f)*(div+1.0f)-4.0f*div*(-MAX_DIM)))/(uint32_t)(2*div));
//         fix_tiling_size_n = fix_tiling_size_k * (uint32_t)div;
//     }
//     size_i = fix_tiling_size_k * sizeof(DATA_TYPE);
//     size_w = fix_tiling_size_k * fix_tiling_size_n * sizeof(DATA_TYPE);
//     size_b = fix_tiling_size_n * sizeof(DATA_TYPE);
//     size_e = fix_tiling_size_n * sizeof(DATA_TYPE);
//     local_I = (void *)snrt_l1_next();
//     local_W = local_I + size_i;
//     local_B = local_W + size_w;
//     local_E = local_B + size_b;
//     remote_B = B;
//     remote_I = I;
//     remote_W = W;
//     remote_E = E;
//     ub_i =(K%fix_tiling_size_k==0)? K/fix_tiling_size_k : K/fix_tiling_size_k+1;
//     ub_j =(N%fix_tiling_size_n==0)? N/fix_tiling_size_n : N/fix_tiling_size_n+1;
//     if(snrt_cluster_core_idx()==0){
//         n_iter[0]=ub_i;
//         n_iter[1]= ub_j;
//         fix_size_k = fix_tiling_size_k;
//         fix_size_n = fix_tiling_size_n;
//    //     printf("Number of iterations is: %u\n",ub_i*ub_j);
//     }
//     for(i=0;i<ub_i;i++){
//         new_tiling_size_k = (i!=ub_i-1) ? fix_tiling_size_k : (K-i*fix_tiling_size_k);
//    //     if(snrt_cluster_core_idx()==0)
//     //        printf("Iteration number %u out of %u\n",i*ub_j,ub_i*ub_j);
//         for(j=0;j<ub_j;j++){
//             //all the cycles but the last one  with fixed size
//             new_tiling_size_n = (j!=ub_j-1) ? fix_tiling_size_n : (N-j*fix_tiling_size_n);
//             // Copy data in TCDM
//             if (snrt_is_dm_core()) {
//                 snrt_mcycle();
//                 if(j==0){
//                     snrt_dma_start_1d(local_I, remote_I+i*fix_tiling_size_k*sizeof(DATA_TYPE), new_tiling_size_k*sizeof(DATA_TYPE));
//                 }
//                 //it is not slower than 2d dma
//                 for(int z=0;z<new_tiling_size_k;z++){
//                     snrt_dma_start_1d(local_W + z*new_tiling_size_n*sizeof(DATA_TYPE),
//                      remote_W + j*fix_tiling_size_n*sizeof(DATA_TYPE)+ i*N*fix_tiling_size_k*sizeof(DATA_TYPE) + z*N*sizeof(DATA_TYPE),
//                      new_tiling_size_n*sizeof(DATA_TYPE));
//                 }
//                 if(i==0){
//                 snrt_dma_start_1d(local_B, remote_B+j*fix_tiling_size_n*sizeof(DATA_TYPE), new_tiling_size_n*sizeof(DATA_TYPE));
//                 }
//                 snrt_dma_start_1d(local_E, remote_E+j*fix_tiling_size_n*sizeof(DATA_TYPE), new_tiling_size_n*sizeof(DATA_TYPE));
//                 snrt_dma_wait_all();
//             }
//             snrt_cluster_hw_barrier();
//             if(!snrt_is_dm_core()){
//                 snrt_mcycle();
//                 if(prec==8){
//                     __backpropagation_multicore_computation_fp64__((DATA_TYPE*)local_I,(DATA_TYPE*)local_W,(DATA_TYPE*)local_B,(DATA_TYPE*)local_E,e,new_tiling_size_n,new_tiling_size_k,i==0);
//                 }else if(prec==4){
//                     __backpropagation_multicore_computation_fp32__((DATA_TYPE*)local_I,(DATA_TYPE*)local_W,(DATA_TYPE*)local_B,(DATA_TYPE*)local_E,e,new_tiling_size_n,new_tiling_size_k,i==0);
//                 }
//                 snrt_fpu_fence();
//             }
//             snrt_cluster_hw_barrier();
//             snrt_mcycle();
//             if (snrt_is_dm_core()) {
//                 for(int z=0;z<new_tiling_size_k;z++){
//                     snrt_dma_start_1d(remote_W + j*fix_tiling_size_n*sizeof(DATA_TYPE) + i*N*fix_tiling_size_k*sizeof(DATA_TYPE)+ z*N*sizeof(DATA_TYPE),
//                      local_W + z*new_tiling_size_n*sizeof(DATA_TYPE),
//                      new_tiling_size_n*sizeof(DATA_TYPE));
//                 }
//                 if(i==0){
//                     snrt_dma_start_1d(remote_B+j*fix_tiling_size_n*sizeof(DATA_TYPE), local_B, new_tiling_size_n*sizeof(DATA_TYPE));
//                 }
//                 snrt_dma_wait_all();
//             }
//             snrt_cluster_hw_barrier();
//         }
//     }
// }



void __backpropagation_multicore_computation_fp64__(DATA_TYPE *local, DATA_TYPE *local_grad,
             uint32_t max_dim){

    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    int32_t dim,c,lb,ub;

    //weights+biases update 
    //treat them as a singular matrix of (N+1)*K on which last line is the bias. The addresses of local and local_grad must be contiguous in TCDM
    c = CEIL(max_dim, compute_num);
    lb = c * compute_id;
    ub = MIN((c * (compute_id + 1)),max_dim);

    dim = ub-lb;
    if(dim>0){
        snrt_ssr_loop_1d(SNRT_SSR_DM0, dim , sizeof(DATA_TYPE));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, dim , sizeof(DATA_TYPE));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, dim , sizeof(DATA_TYPE));


        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, local+lb);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, local_grad+lb);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, local+lb);

        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fsub.d ft2, ft0, ft1\n"
            :
            : [ n_frep ] "r"(dim-1) //define variables used 
            : "ft0", "ft1", "ft2", "memory"); //registered touched

        snrt_ssr_disable();
    }

    snrt_fpu_fence();

}


void __backpropagation_multicore_computation_fp32__(DATA_TYPE *local_i, DATA_TYPE *local_w, DATA_TYPE *local_b, DATA_TYPE *local_e,
             double e, int n, int k, uint32_t do_bias){

                //TODO
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
