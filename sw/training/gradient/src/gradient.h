
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
