
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif 


#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))


#define MAX_DIM 100000.0f / sizeof(DATA_TYPE)

uint32_t n_iter[2]__attribute__ ((aligned (4096)));
uint32_t fix_size_k;
uint32_t fix_size_n;
float local_E_array[5];


void backpropagation_baseline_multicore(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e,int m, int n, int k);

void __backpropagation_multicore_computation_fp64__(DATA_TYPE *local_i, DATA_TYPE *local_w, DATA_TYPE *local_b, DATA_TYPE *local_e,
             double e, int n, int k, uint32_t do_bias);

void __backpropagation_multicore_computation_fp32__(DATA_TYPE *local_i, DATA_TYPE *local_w, DATA_TYPE *local_b, DATA_TYPE *local_e,
             double e, int n, int k, uint32_t do_bias);

// I[m][k] inputs
// W[k][n] weights
// B[m][n] biases
// E[m][n] error
// e learning rate
void backpropagation_baseline_one_core(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e, int N, int K){

    int i,j;
    uint32_t size_i, size_w, size_b, size_e;
    DATA_TYPE *local_I, *local_W, *local_B, *local_E;
    void *remote_B,*remote_I,*remote_E,*remote_W;

    size_i = K ;
    size_w = K * N;
    size_b = N ;
    size_e = N ;


    local_I = (DATA_TYPE *)snrt_l1_next();
    local_W = local_I + size_i;
    local_B = local_W + size_w;
    local_E = local_B + size_b;

    remote_B = B;
    remote_I = I;
    remote_W = W;
    remote_E = E;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_I, remote_I, K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_W, remote_W, N*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_B, remote_B, N*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_E, remote_E, N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();


    if(snrt_cluster_core_idx()==0){
        snrt_mcycle();

        //dC/dB = E 
        //B[m][n] = old(B[m][n])- e*dC/dB[m][n]
        for(i=0;i<N;i++){
            local_B[i] = local_B[i]- e*local_E[i];
        }

        //dC/dW = I_t * E
        //W[k][n] = old(W[k][n]) - e*dC/dW[k][n]
        for(i=0;i<K;i++){
            for(j=0;j<N;j++){
                local_W[(i)*N + (j)] = local_W[(i)*N + (j)] - e * local_I[i]*local_E[j];
            }
        }

        snrt_mcycle();

    }
    
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(remote_W, local_W, N*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(remote_B, local_B, N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

}

void backpropagation_one_core(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e, int N, int K){

    int i,j;
    uint32_t size_i, size_w, size_b, size_e;
    DATA_TYPE *local_I, *local_W, *local_B, *local_E;
    void *remote_B,*remote_I,*remote_E,*remote_W;
    const register double neg_e_reg = -e;
    register uint32_t n_frep_reg = N-1;
    register double I_reg;

    size_i = K ;
    size_w = K * N;
    size_b = N ;
    size_e = N ;


    local_I = (DATA_TYPE *)snrt_l1_next();
    local_W = local_I + size_i;
    local_B = local_W + size_w;
    local_E = local_B + size_b;

    remote_B = B;
    remote_I = I;
    remote_W = W;
    remote_E = E;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_I, remote_I, K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_W, remote_W, N*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_B, remote_B, N*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_E, remote_E, N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();
    snrt_mcycle();

    if(snrt_cluster_core_idx()==0){

        snrt_ssr_loop_1d(SNRT_SSR_DM0, N , sizeof(DATA_TYPE));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, N , sizeof(DATA_TYPE));
        snrt_ssr_loop_1d(SNRT_SSR_DM2, N , sizeof(DATA_TYPE));


        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, local_B);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, local_E);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, local_B);

        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fmadd.d ft2, %[neg_e], ft1, ft0\n"
            :
            : [ n_frep ] "r"(n_frep_reg), [ neg_e ] "f"(neg_e_reg) //define variables used 
            : "ft0", "ft1", "ft2", "memory"); //registered touched

        snrt_fpu_fence();
        snrt_ssr_disable();

        
        n_frep_reg = N/4-1;

        for(i=0;i<K;i++){
            I_reg = local_I[i];
            snrt_ssr_loop_1d(SNRT_SSR_DM0,N,sizeof(DATA_TYPE));//E
            snrt_ssr_loop_1d(SNRT_SSR_DM1,N,sizeof(DATA_TYPE));//W read
            snrt_ssr_loop_1d(SNRT_SSR_DM2,N,sizeof(DATA_TYPE));//W write

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, local_E);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, local_W+i*N);
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, local_W+i*N);

            snrt_ssr_enable();
            asm volatile(  "frep.o %[n_frep], 8, 0, 0\n"
            "fmul.d ft3, ft0, %[I] \n"
            "fmul.d ft4, ft0, %[I] \n"
            "fmul.d ft5, ft0, %[I] \n"
            "fmul.d ft6, ft0, %[I] \n"
            "fmadd.d ft2, %[neg_e], ft3, ft1 \n"
            "fmadd.d ft2, %[neg_e], ft4, ft1 \n"
            "fmadd.d ft2, %[neg_e], ft5, ft1 \n"
            "fmadd.d ft2, %[neg_e], ft6, ft1 \n"

            :
            : [ n_frep ] "r"(n_frep_reg), [ neg_e ] "fr"(neg_e_reg), [I] "fr"(I_reg)
            :"ft0", "ft1","ft2","ft3", "ft4", "ft5","ft6","memory");

            switch(N%4){
                case 3:
                    asm volatile(
                        "fmul.d ft3, ft0, %[I] \n"
                        "fmul.d ft4, ft0, %[I] \n"
                        "fmul.d ft5, ft0, %[I] \n"
                        "fmadd.d ft2, %[neg_e], ft3, ft1 \n"
                        "fmadd.d ft2, %[neg_e], ft4, ft1 \n"
                        "fmadd.d ft2, %[neg_e], ft5, ft1 \n"

                        :
                        : [ neg_e ] "fr"(neg_e_reg), [I] "fr"(I_reg)
                        :"ft0", "ft1","ft2","ft3","ft4","ft5","memory");
                        break;
                case 2:
                    asm volatile(
                        "fmul.d ft3, ft0, %[I] \n"
                        "fmul.d ft4, ft0, %[I] \n"
                        "fmadd.d ft2, %[neg_e], ft3, ft1 \n"
                        "fmadd.d ft2, %[neg_e], ft4, ft1 \n"
                        :
                        : [ neg_e ] "fr"(neg_e_reg), [I] "fr"(I_reg)
                        :"ft0", "ft1","ft2","ft3","ft4","memory");
                        break;
                case 1:
                    asm volatile(
                        "fmul.d ft3, ft0, %[I] \n"
                        "fmadd.d ft2, %[neg_e], ft3, ft1 \n"

                        :
                        : [ neg_e ] "fr"(neg_e_reg), [I] "fr"(I_reg)
                        :"ft0", "ft1","ft2","ft3","memory");
                    break;

            }
            snrt_fpu_fence();
            snrt_ssr_disable();
        }
    }

    snrt_mcycle();
    snrt_fpu_fence();
    snrt_cluster_hw_barrier();


    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(remote_W, local_W, N*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(remote_B, local_B, N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

}


//works only without tiling
void backpropagation_baseline_multicore(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e,int N, int K, int prec){


    uint32_t c, lb, ub,i,j;
    uint32_t size_i, size_w, size_b, size_e;
    DATA_TYPE *local_I, *local_W, *local_B, *local_E;
    void *remote_B,*remote_I,*remote_E,*remote_W;
    
    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();

    size_i = K ;
    size_w = K * N;
    size_b = N ;
    size_e = N ;


    local_I = (DATA_TYPE *)snrt_l1_next();
    local_W = local_I + size_i;
    local_B = local_W + size_w;
    local_E = local_B + size_b;

    remote_B = B;
    remote_I = I;
    remote_W = W;
    remote_E = E;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_I, remote_I, K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_W, remote_W, N*K*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_B, remote_B,N*sizeof(DATA_TYPE));
        snrt_dma_start_1d(local_E, remote_E,N*sizeof(DATA_TYPE));
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();


    if(!snrt_is_dm_core()){
        snrt_mcycle();
        c = CEIL(N, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), N);


        //dC/dB = E 
        //B[n] = old(B[n])- e*dC/dB[n]
        for(i=lb;i<ub;i++){
            local_B[i] = local_B[i]- e*local_E[i];
        }

        c = CEIL(K, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), K);

        //dC/dW = I_t * E
        //W[k][n] = old(W[k][n]) - e*dC/dW[k][n]
        for(i=lb;i<ub;i++){
            for(j=0;j<N;j++){
                local_W[(i)*N+ (j)] = local_W[(i)*N + (j)] - e * local_I[i]*local_E[j];
            }
        }
        snrt_mcycle();
        snrt_fpu_fence();
    
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {

                snrt_dma_start_1d(remote_W, local_W, N*K*sizeof(DATA_TYPE));

                snrt_dma_start_1d(remote_B, local_B, N*sizeof(DATA_TYPE));

                snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

        
}

void backpropagation_multicore(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e,int N, int K,int prec){

    float div;
    uint32_t fix_tiling_size_k,new_tiling_size_k;
    uint32_t fix_tiling_size_n, new_tiling_size_n; 
    uint32_t size_i, size_w, size_b, size_e;
    void *local_I, *local_W, *local_B, *local_E;
    void *remote_B,*remote_I,*remote_E,*remote_W;
    uint32_t ub_i, ub_j,i ,j;

    //find best values for fix sizes of tiling
    if(K>=N){
        div =(float) (K/N); //K=div*N

        //K*N+k+n<=MAX_DIM ---> div*N*N+(div+1)*N-MAX_DIM == 0 ----> N =(-(div+1)+sqrt((div+1)*(div+1)-4*div*(-MAX_DIM))) / 2*div
        fix_tiling_size_n = ((uint32_t)(-(div+1.0f)+(float)sqrtf((div+1.0f)*(div+1.0f)-4.0f*div*(-MAX_DIM)))/(uint32_t)(2*div));

        fix_tiling_size_k = fix_tiling_size_n * (uint32_t)div;
        
    }else{
        div =(float) (N/K); 

        fix_tiling_size_k = ((uint32_t)(-(div+1.0f)+(float)sqrtf((div+1.0f)*(div+1.0f)-4.0f*div*(-MAX_DIM)))/(uint32_t)(2*div));

        fix_tiling_size_n = fix_tiling_size_k * (uint32_t)div;

    }

    size_i = fix_tiling_size_k * sizeof(DATA_TYPE);
    size_w = fix_tiling_size_k * fix_tiling_size_n * sizeof(DATA_TYPE);
    size_b = fix_tiling_size_n * sizeof(DATA_TYPE);
    size_e = fix_tiling_size_n * sizeof(DATA_TYPE);

    local_I = (void *)snrt_l1_next();
    local_W = local_I + size_i;
    local_B = local_W + size_w;
    local_E = local_B + size_b;

    remote_B = B;
    remote_I = I;
    remote_W = W;
    remote_E = E;

    ub_i =(K%fix_tiling_size_k==0)? K/fix_tiling_size_k : K/fix_tiling_size_k+1;
    ub_j =(N%fix_tiling_size_n==0)? N/fix_tiling_size_n : N/fix_tiling_size_n+1;
  
    if(snrt_cluster_core_idx()==0){
        n_iter[0]=ub_i;
        n_iter[1]= ub_j;
        fix_size_k = fix_tiling_size_k;
        fix_size_n = fix_tiling_size_n;
   //     printf("Number of iterations is: %u\n",ub_i*ub_j);
    }

    for(i=0;i<ub_i;i++){
        new_tiling_size_k = (i!=ub_i-1) ? fix_tiling_size_k : (K-i*fix_tiling_size_k);
   //     if(snrt_cluster_core_idx()==0)
    //        printf("Iteration number %u out of %u\n",i*ub_j,ub_i*ub_j);
        for(j=0;j<ub_j;j++){

            //all the cycles but the last one  with fixed size
            new_tiling_size_n = (j!=ub_j-1) ? fix_tiling_size_n : (N-j*fix_tiling_size_n);
        

            // Copy data in TCDM
            if (snrt_is_dm_core()) {

                if(j==0){
                    snrt_dma_start_1d(local_I, remote_I+i*fix_tiling_size_k*sizeof(DATA_TYPE), new_tiling_size_k*sizeof(DATA_TYPE));
                }
                //it is not slower than 2d dma
                for(int z=0;z<new_tiling_size_k;z++){
                    snrt_dma_start_1d(local_W + z*new_tiling_size_n*sizeof(DATA_TYPE),
                     remote_W + j*fix_tiling_size_n*sizeof(DATA_TYPE)+ i*N*fix_tiling_size_k*sizeof(DATA_TYPE) + z*N*sizeof(DATA_TYPE),
                     new_tiling_size_n*sizeof(DATA_TYPE));
                }

                if(i==0){
                snrt_dma_start_1d(local_B, remote_B+j*fix_tiling_size_n*sizeof(DATA_TYPE), new_tiling_size_n*sizeof(DATA_TYPE));
                }

                snrt_dma_start_1d(local_E, remote_E+j*fix_tiling_size_n*sizeof(DATA_TYPE), new_tiling_size_n*sizeof(DATA_TYPE));

                snrt_dma_wait_all();
            }
        
            snrt_cluster_hw_barrier();


            if(!snrt_is_dm_core()){
                snrt_mcycle();
                
                if(prec==8){
                    __backpropagation_multicore_computation_fp64__((DATA_TYPE*)local_I,(DATA_TYPE*)local_W,(DATA_TYPE*)local_B,(DATA_TYPE*)local_E,e,new_tiling_size_n,new_tiling_size_k,i==0);
                }else if(prec==4){
                    __backpropagation_multicore_computation_fp32__((DATA_TYPE*)local_I,(DATA_TYPE*)local_W,(DATA_TYPE*)local_B,(DATA_TYPE*)local_E,e,new_tiling_size_n,new_tiling_size_k,i==0);
                }

                snrt_mcycle();
                snrt_fpu_fence();
            }
         
            snrt_cluster_hw_barrier();
            

            if (snrt_is_dm_core()) {

                for(int z=0;z<new_tiling_size_k;z++){
                    snrt_dma_start_1d(remote_W + j*fix_tiling_size_n*sizeof(DATA_TYPE) + i*N*fix_tiling_size_k*sizeof(DATA_TYPE)+ z*N*sizeof(DATA_TYPE),
                     local_W + z*new_tiling_size_n*sizeof(DATA_TYPE),
                     new_tiling_size_n*sizeof(DATA_TYPE));
                }

                if(i==0){
                    snrt_dma_start_1d(remote_B+j*fix_tiling_size_n*sizeof(DATA_TYPE), local_B, new_tiling_size_n*sizeof(DATA_TYPE));
                }

                snrt_dma_wait_all();
            }

            snrt_cluster_hw_barrier();
        }
    }
 
}



void __backpropagation_multicore_computation_fp64__(DATA_TYPE *local_i, DATA_TYPE *local_w, DATA_TYPE *local_b, DATA_TYPE *local_e,
             double e, int n, int k, uint32_t do_bias){

    uint32_t c, lb, ub,i,j,z;
    int32_t dim; 
    const register double neg_e_reg = -e;
    register uint32_t n_frep_reg;
    register double I_reg;

    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
 

    if(do_bias){      
        c = CEIL(n, compute_num);
        lb = c * compute_id;
        ub = MIN((c * (compute_id + 1)), n);
        dim = ub-lb;  

        //dC/dB = E 
        //B[n] = old(B[n])- e*dC/dB[n]
        if(dim>0){
            n_frep_reg = dim-1;
            snrt_ssr_loop_1d(SNRT_SSR_DM0, dim, sizeof(DATA_TYPE));
            snrt_ssr_loop_1d(SNRT_SSR_DM1,dim, sizeof(DATA_TYPE));
            snrt_ssr_loop_1d(SNRT_SSR_DM2, dim, sizeof(DATA_TYPE));


            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, local_b+lb);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, local_e+lb);
            snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, local_b+lb);

            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fmadd.d ft2, %[neg_e], ft1, ft0\n"
                :
                : [ n_frep ] "r"(n_frep_reg), [ neg_e ] "f"(neg_e_reg) //define variables used 
                : "ft0", "ft1", "ft2", "memory"); //registered touched

            snrt_fpu_fence();
            snrt_ssr_disable();
        }
    }

    c = CEIL(k, compute_num);
    lb = c * compute_id;
    ub = MIN((c * (compute_id + 1)), k);
    n_frep_reg = n/4-1;
    //dC/dW = I_t * E
    //W[k][n] = old(W[k][n]) - e*dC/dW[k][n]
    for(i =lb;i<ub;i++){
        I_reg = local_i[i];
        snrt_ssr_loop_1d(SNRT_SSR_DM0,n,sizeof(DATA_TYPE));//E
        snrt_ssr_loop_1d(SNRT_SSR_DM1,n,sizeof(DATA_TYPE));//W read
        snrt_ssr_loop_1d(SNRT_SSR_DM2,n,sizeof(DATA_TYPE));//W write

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, local_e);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, local_w+i*n);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, local_w+i*n);

        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 8, 0, 0\n"
            "fmul.d ft3, ft0, %[I] \n"
            "fmul.d ft4, ft0, %[I] \n"
            "fmul.d ft5, ft0, %[I] \n"
            "fmul.d ft6, ft0, %[I] \n"
            "fmadd.d ft2, %[neg_e], ft3, ft1 \n"
            "fmadd.d ft2, %[neg_e], ft4, ft1 \n"
            "fmadd.d ft2, %[neg_e], ft5, ft1 \n"
            "fmadd.d ft2, %[neg_e], ft6, ft1 \n"

            :
            : [ n_frep ] "r"(n_frep_reg), [ neg_e ] "f"(neg_e_reg), [I] "f"(I_reg)
            :"ft0", "ft1","ft2","ft3", "ft4", "ft5","ft6","memory");

    
        switch(n%4){
            case 3:
                asm volatile(
                    "fmul.d ft3, ft0, %[I] \n"
                    "fmul.d ft4, ft0, %[I] \n"
                    "fmul.d ft5, ft0, %[I] \n"
                    "fmadd.d ft2, %[neg_e], ft3, ft1 \n"
                    "fmadd.d ft2, %[neg_e], ft4, ft1 \n"
                    "fmadd.d ft2, %[neg_e], ft5, ft1 \n"

                    :
                    : [ neg_e ] "f"(neg_e_reg), [I] "f"(I_reg)
                    :"ft0", "ft1","ft2","ft3","ft4","ft5","memory");
                    break;
            case 2:
                asm volatile(
                    "fmul.d ft3, ft0, %[I] \n"
                    "fmul.d ft4, ft0, %[I] \n"
                    "fmadd.d ft2, %[neg_e], ft3, ft1 \n"
                    "fmadd.d ft2, %[neg_e], ft4, ft1 \n"
                    :
                    : [ neg_e ] "f"(neg_e_reg), [I] "f"(I_reg)
                    :"ft0", "ft1","ft2","ft3","ft4","memory");
                    break;
            case 1:
                asm volatile(
                    "fmul.d ft3, ft0, %[I] \n"
                    "fmadd.d ft2, %[neg_e], ft3, ft1 \n"

                    :
                    : [ neg_e ] "f"(neg_e_reg), [I] "f"(I_reg)
                    :"ft0", "ft1","ft2","ft3","memory");
                break;

        }
        snrt_fpu_fence();
        snrt_ssr_disable();
    
    }


        //I liked this one more
        // snrt_ssr_loop_2d(SNRT_SSR_DM1,n,k,0,sizeof(DATA_TYPE));//I_t
        // snrt_ssr_loop_2d(SNRT_SSR_DM2,n,k,sizeof(DATA_TYPE),0);//E
        // snrt_ssr_loop_2d(SNRT_SSR_DM0,2,n*k,0,sizeof(DATA_TYPE));//W       
        // snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, local_w); 
        // snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, local_i);            
        // snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, local_e); 
        // snrt_ssr_write(SNRT_SSR_DM0, SNRT_SSR_2D, local_w);      
        // snrt_ssr_enable();
        // asm volatile(
        //     "frep.o %[n_frep], 2, 0, 0 \n" //rep_loop, length_loop, stagger_count, dtagger [rd|rs1|rs2|rs3]
        //     "fmul.d fa0, ft2, ft1 \n"
        //     "fmadd.d ft0, %[neg_e], fa0, ft0 \n"
        //     :  
        //     : [ n_frep ] "r"(k*n-1), [ neg_e ] "f"(-e), [ addr ] "r"(local_w)//define variables used 
        //     : "ft0", "ft1","fa0","fa1", "memory"); //registered touched
        // snrt_fpu_fence();
        // snrt_ssr_disable();
}


void __backpropagation_multicore_computation_fp32__(DATA_TYPE *local_i, DATA_TYPE *local_w, DATA_TYPE *local_b, DATA_TYPE *local_e,
             double e, int n, int k, uint32_t do_bias){

                //TODO
}
