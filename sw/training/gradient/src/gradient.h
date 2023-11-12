
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif 

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))

double ubs[8]__attribute__ ((aligned (4096)));
double lbs[8]__attribute__ ((aligned (4096))); 

void backpropagation_one_core(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e,int m, int n, int k);
void backpropagation_baseline_multicore(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e,int m, int n, int k);
void backpropagation_multicore(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e,int m, int n, int k);

// I[m][k] inputs
// W[k][n] weights
// B[m][n] biases
// E[m][n] error
// e learning rate
void backpropagation_one_core(DATA_TYPE *I, DATA_TYPE *W, DATA_TYPE *B, DATA_TYPE *E,
             double e,int m, int n, int k){

    int i,j,z;
    DATA_TYPE sum;

    //dC/dB = E 
    //B[m][n] = old(B[m][n])- e*dC/dB[m][n]
    for(i=0;i<n*m;i++){
        B[i] = B[i]- e*E[i];
    }

    //dC/dW = I_t * E
    //W[k][n] = old(W[k][n]) - e*dC/dW[k][n]
    for(i=0;i<k;i++){
        for(j=0;j<n;j++){
            sum=0;
            for(z=0;z<m;z++){                
                sum += I[(z)*m+(i)]*E[(z)*n+(j)];
            }
            W[(i)*n + (j)] = W[(i)*n + (j)] - e * sum;
        }
    }

}

void backpropagation_baseline_multicore(DATA_TYPE *local_i, DATA_TYPE *local_w, DATA_TYPE *local_b, DATA_TYPE *local_e,
             double e,int m, int n, int k){

    uint32_t c, lb, ub,i,j,z;
    DATA_TYPE sum;
    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    
    c = CEIL(n*m, compute_num);
    lb = c * compute_id;
    ub = MIN((c * (compute_id + 1)), n*m);

    lbs[compute_id]=lb;
    ubs[compute_id]=ub;

    //dC/dB = E 
    //B[m][n] = old(B[m][n])- e*dC/dB[m][n]
    for(i=lb;i<ub;i++){
        local_b[i] = local_b[i]- e*local_e[i];
    }

    c = CEIL(k, compute_num);
    lb = c * compute_id;
    ub = MIN((c * (compute_id + 1)), k);

    //dC/dW = I_t * E
    //W[k][n] = old(W[k][n]) - e*dC/dW[k][n]
    for(i=lb;i<ub;i++){
        for(j=0;j<n;j++){
            sum=0;
            for(z=0;z<m;z++){                
                sum += local_i[(z)*m+(i)]*local_e[(z)*n+(j)];
            }
            local_w[(i)*n + (j)] = local_w[(i)*n + (j)] - e * sum;
        }
    }
    
}

void backpropagation_multicore(DATA_TYPE *local_i, DATA_TYPE *local_w, DATA_TYPE *local_b, DATA_TYPE *local_e,
             double e,int m, int n, int k){

    uint32_t c, lb, ub,i,j,z;
    DATA_TYPE sum;
    const uint32_t compute_num = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();
    
    c = CEIL(n*m, compute_num);
    lb = c * compute_id;
    ub = MIN((c * (compute_id + 1)), n*m);

    lbs[compute_id]=lb;
    ubs[compute_id]=ub;

    uint32_t dim = ub-lb;  
    //dC/dB = E 
    //B[m][n] = old(B[m][n])- e*dC/dB[m][n]
    if(dim>0){
        snrt_ssr_loop_1d(SNRT_SSR_DM0, dim, sizeof(DATA_TYPE));
        snrt_ssr_loop_1d(SNRT_SSR_DM1,dim, sizeof(DATA_TYPE));

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, local_b+lb);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, local_e+lb);
        snrt_ssr_write(SNRT_SSR_DM0, SNRT_SSR_1D, local_b+lb);

        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fmadd.d ft0, %[neg_e], ft1, ft0\n"
            :
            : [ n_frep ] "r"(dim-1), [ neg_e ] "f"(-e) //define variables used 
            : "ft0", "ft1", "ft2", "memory"); //registered touched

        snrt_fpu_fence();
        snrt_ssr_disable();
    }

    c = CEIL(k, compute_num);
    lb = c * compute_id;
    ub = MIN((c * (compute_id + 1)), k);

    //dC/dW = I_t * E
    //W[k][n] = old(W[k][n]) - e*dC/dW[k][n]
    for(i =lb;i<ub;i++){
        snrt_ssr_loop_1d(SNRT_SSR_DM0,n,sizeof(DATA_TYPE));//E
        snrt_ssr_loop_1d(SNRT_SSR_DM1,n,sizeof(DATA_TYPE));//W read
        snrt_ssr_loop_1d(SNRT_SSR_DM2,n,sizeof(DATA_TYPE));//W write

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, local_e);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, local_w+i*n);
        snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, local_w+i*n);

        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 2, 0, 0\n"
            "fmul.d fa0, ft0, %[I] \n"
            "fmadd.d ft2, %[neg_e], fa0, ft1 \n"
            :
            : [ n_frep ] "r"(n-1), [ neg_e ] "f"(-e), [I] "f"(local_i[i])
            :"ft0", "ft1","ft2", "memory");
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