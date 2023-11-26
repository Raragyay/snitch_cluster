#pragma once

#include <math.h>
#include "batchnorm_data_structures.h"
#include "batchnorm_utils.h"
#include "snrt.h"

// No DMA, SSR, or FREP. Still uses TCDM.
static inline void batchnorm_backward_single_core(
    batchnorm_backward_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in HWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster
    // Calculate output dimensions
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    double eps = l->eps;
    uint32_t num_points = N * H * W;

    uint32_t buffer_len = l->CI;
    uint32_t invstd_scratch_len = l->CI;
    uint32_t weight_times_invstd_len = l->CI,
             running_mean_times_invstd_len = l->CI;

    double *ptr = (double *)snrt_l1_start_addr();
    double *invstd_scratch = ptr;
    ptr += invstd_scratch_len;
    double *weight_times_invstd_scratch = ptr;
    ptr += weight_times_invstd_len;
    double *running_mean_times_invstd_scratch = ptr;
    ptr += running_mean_times_invstd_len;
    uint32_t start_dma_load = snrt_mcycle();
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    uint32_t start_invstd_computations = SNRT_SECTIONED_MCYCLE();
    if (compute_id == 0) {
        for (uint32_t channel = 0; channel < C; ++channel) {
            double invstd = 1 / sqrt(l->running_var[channel] + eps);
            invstd_scratch[channel] = invstd;
            weight_times_invstd_scratch[channel] = invstd * l->weight[channel];
            running_mean_times_invstd_scratch[channel] =
                invstd * l->running_mean[channel];
        }
    }
    uint32_t end_invstd_computations = SNRT_SECTIONED_MCYCLE();
    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();

    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    if (compute_id == 0) {
        for (uint32_t i = 0; i < num_points; i += 1) {
            for (uint32_t channel = 0; channel < C; ++channel) {
                double dy = l->grad_ofmap[i * C + channel];
                double x = l->ifmap[i * C + channel];
                l->grad_bias[channel] += dy;
                l->grad_weight[channel] +=
                    dy * (x * invstd_scratch[channel] -
                          running_mean_times_invstd_scratch[channel]);
                l->grad_ifmap[i * C + channel] =
                    dy * weight_times_invstd_scratch[channel];
            }
        }
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    uint32_t start_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    uint32_t done = snrt_mcycle();
}

// uses DMA, SSR, FREP
static inline void batchnorm_backward_single_core_opt(
    batchnorm_backward_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster

    // keep the dma core and one compute core

    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    // ignore tiling for now
    uint32_t num_points = N * H * W;
    uint32_t num_doubles = num_points * C;
    uint32_t num_bytes = num_doubles * sizeof(double);
    uint32_t point_size_in_bytes = C * sizeof(double);

    ptrdiff_t grad_bias_scratch_len = C, grad_weight_scratch_len = C;

    // dataflow:
    double *ptr = (double *)snrt_l1_start_addr();
    double *weight_scratch = ptr;
    ptr += C;
    double *invstd_scratch = ptr;
    ptr += C;
    double *running_mean_scratch = ptr;
    ptr += C;
    double *grad_bias_scratch = ptr;
    ptr += grad_bias_scratch_len;
    double *grad_weight_scratch = ptr;
    ptr += grad_weight_scratch_len;
    ptrdiff_t grad_ofmap_len = num_points * C, grad_ifmap_len = grad_ofmap_len,
              ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;
    double *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    snrt_dma_txid_t running_var_load, weight_load, running_mean_load,
        grad_ofmap_load, ifmap_load, grad_ifmap_write;

    uint32_t start_dma_load = snrt_mcycle();
    // load running_var, initiate the rest
    if (snrt_is_dm_core()) {
        // Initiate loads for everything but only wait for the running var load.
        // Q: is it better to wait then initiate the rest? we'll see
        running_var_load = snrt_dma_start_1d(invstd_scratch, l->running_var,
                                             C * sizeof(double));
        snrt_dma_wait(running_var_load);
    } else if (compute_id == 0) {
        // PRECONFIGURE: operations on arrays of size C, split by core.
        snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, C, sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_invstd_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        weight_load =
            snrt_dma_start_1d(weight_scratch, l->weight, point_size_in_bytes);
        running_mean_load = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, point_size_in_bytes);
        // load first tile in. We can do this here because sqrt/div are really
        // slow.
        grad_ofmap_load =
            snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap, num_bytes);
        ifmap_load = snrt_dma_start_1d(ifmap_scratch, l->ifmap, num_bytes);

        snrt_dma_wait_all();
    } else if (compute_id == 0) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, invstd_scratch);
        register double eps = l->eps;  // any value in dma'ing this? idk
        const register double ONE = 1;
        snrt_ssr_enable();
        // might be worth unrolling to avoid dependencies? not sure
        asm volatile(
            "frep.o %[n_frep], 3, 0, 0 \n"
            "fadd.d ft3, ft0, %[eps]\n"
            "fsqrt.d ft3, ft3\n"
            "fdiv.d ft1, %[ONE], ft3\n"
            :
            : [eps] "fr"(eps), [ONE] "fr"(ONE),
              [n_frep] "r"(C - 1)  // we repeat n_frep+1 times
            : "ft0", "ft1", "ft2", "ft3");

        snrt_fpu_fence();                     // thought: do we need this?
        __builtin_ssr_barrier(SNRT_SSR_DM1);  // thought: do we need this?
        snrt_ssr_disable();
    }
    uint32_t end_invstd_calc = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute weight*invstd and running_mean*invstd

    // computing invstd scratch and using it for weight: can we do it in 1 frep?
    // load running var: 1 ssr
    // write running var: 1 ssr
    // load weight: 1 ssr
    // write weight: 1 ssr
    // answer: not really. Still worth precomputing I think
    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    // compute grad_weight, grad_bias, grad_ifmap
    if (snrt_is_dm_core()) {
    } else if (compute_id == 0) {
        batchnorm_backward_tile_fp64(grad_ofmap_scratch, grad_ifmap_scratch,
                                     ifmap_scratch, running_mean_scratch,
                                     weight_scratch, invstd_scratch,
                                     grad_bias_scratch, grad_weight_scratch, C,
                                     num_points, C, 1, true, true);
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    // don't need second reduction
    snrt_cluster_hw_barrier();
    uint32_t start_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    // write back grad_bias and grad_weight. then wait for all transactions to
    // complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, C * sizeof(double));
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          C * sizeof(double));
        snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch, num_bytes);
        snrt_dma_wait_all();
    } else if (compute_id == 0) {
    }
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
}

// No DMA, SSR, or FREP. Still uses TCDM.
static inline void batchnorm_backward_training_single_core(
    batchnorm_backward_training_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in HWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster
    // Calculate output dimensions
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    double eps = l->eps;
    uint32_t num_points = N * H * W;

    uint32_t invstd_scratch_len = l->CI, k_scratch_len = l->CI,
             grad_mean_scratch_len = l->CI, dx_scratch_len = l->CI;

    double *ptr = (double *)snrt_l1_start_addr();
    double *invstd_scratch = ptr;
    ptr += invstd_scratch_len;
    double *k_scratch = ptr;
    ptr += k_scratch_len;
    double *grad_mean_scratch = ptr;
    ptr += grad_mean_scratch_len;
    uint32_t start_dma_load = snrt_mcycle();
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_invstd_load = SNRT_SECTIONED_MCYCLE();
    if (compute_id == 0) {
        for (uint32_t channel = 0; channel < C; ++channel) {
            double invstd = 1 / sqrt(l->current_var[channel] + eps);
            double sum = 0;
            double dotp = 0;
            invstd_scratch[channel] = invstd;
            for (uint32_t i = 0; i < num_points; i++) {
                double dy = l->grad_ofmap[i * C + channel];
                double x = l->ifmap[i * C + channel];
                double curr_mean = l->current_mean[channel];
                sum += dy;
                dotp += dy * (x - curr_mean);
            }
            k_scratch[channel] = dotp * invstd * invstd / num_points;
            grad_mean_scratch[channel] = sum / num_points;
            l->grad_bias[channel] = sum;
            l->grad_weight[channel] = dotp * invstd;
        }
    }
    uint32_t end_compute_invstd_load = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_sum_dotp_reduction_1 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_sum_dotp_reduction_1 = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    uint32_t start_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();

    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
    if (compute_id == 0) {
        for (uint32_t channel = 0; channel < C; ++channel) {
            double invstd = invstd_scratch[channel];
            double k = k_scratch[channel];
            double curr_mean = l->current_mean[channel];
            double grad_mean = grad_mean_scratch[channel];
            double weight = l->weight[channel];
            for (uint32_t i = 0; i < num_points; i += 1) {
                double dy = l->grad_ofmap[i * C + channel];
                double x = l->ifmap[i * C + channel];
                double dx = (x - curr_mean) * k;
                l->grad_ifmap[i * C + channel] =
                    (dy - grad_mean - dx) * invstd * weight;
            }
        }
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    uint32_t start_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction = SNRT_SECTIONED_MCYCLE();
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    uint32_t done = snrt_mcycle();
}

// uses DMA, SSR, FREP
static inline void batchnorm_backward_training_single_core_opt(
    batchnorm_backward_training_layer_t *l) {
    uint32_t kernel_start = snrt_mcycle();
    // data is in HWC format
    const uint32_t compute_id = snrt_cluster_core_idx();

    // keep the dma core and one compute core
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    // ignore tiling for now
    uint32_t num_points = N * H * W;
    uint32_t num_doubles = num_points * C;
    uint32_t num_bytes = num_doubles * sizeof(double);
    uint32_t point_size_in_bytes = C * sizeof(double);

    double *ptr = (double *)snrt_l1_start_addr();

    double *invstd_scratch = ptr;
    ptr += C;
    double *dotp_scratch = ptr;
    ptr += C;
    double *k_scratch = ptr;
    ptr += C;
    double *grad_mean_scratch = ptr;
    ptr += C;
    double *dx_scratch = ptr;
    ptr += C * num_points;

    double *weight_scratch = ptr;
    ptr += C;
    double *curr_mean_scratch = ptr;
    ptr += C;
    double *curr_var_scratch = ptr;
    ptr += C;

    ptrdiff_t grad_weight_scratch_len = C, grad_bias_scratch_len = C;

    double *grad_weight_scratch = ptr;
    ptr += grad_weight_scratch_len;
    double *grad_bias_scratch = ptr;
    ptr += grad_bias_scratch_len;

    ptrdiff_t grad_ofmap_len = num_points * C, grad_ifmap_len = grad_ofmap_len,
              ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;
    double *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer
    ptr += grad_ifmap_len;

    // Load data
    snrt_dma_txid_t invstd_load, curr_var_load, weight_load, curr_mean_load,
        grad_ofmap_load, ifmap_load;

    uint32_t start_dma_load = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        curr_var_load = snrt_dma_start_1d(invstd_scratch, l->current_var,
                                          point_size_in_bytes);
        grad_ofmap_load =
            snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap, num_bytes);
        ifmap_load = snrt_dma_start_1d(ifmap_scratch, l->ifmap, num_bytes);
        curr_mean_load = snrt_dma_start_1d(curr_mean_scratch, l->current_mean,
                                           point_size_in_bytes);
        weight_load =
            snrt_dma_start_1d(weight_scratch, l->weight, point_size_in_bytes);
        snrt_dma_wait(curr_var_load);
    } else if (compute_id == 0) {
        snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, C, sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_invstd_computations = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_wait(grad_ofmap_load);
        snrt_dma_wait(ifmap_load);
        snrt_dma_wait(curr_mean_load);
    } else if (compute_id == 0) {
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, invstd_scratch);
        register double eps = l->eps;
        const register double ONE = 1;
        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 3, 0, 0 \n"
            "fadd.d ft3, ft0, %[eps]\n"
            "fsqrt.d ft3, ft3\n"
            "fdiv.d ft1, %[ONE], ft3\n"
            :
            : [eps] "fr"(eps), [ONE] "fr"(ONE), [n_frep] "r"(C - 1)
            : "ft0", "ft1", "ft2", "ft3");

        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();
    }
    uint32_t end_invstd_computations = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_sum_dotp_reduction = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
    } else if (compute_id == 0) {
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, num_points, C, point_size_in_bytes,
                         sizeof(double));
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, ifmap_scratch);
        for (uint32_t channel = 0; channel < C; ++channel) {
            register volatile double sum = 0;
            register volatile double dotp = 0;
            register double curr_mean = curr_mean_scratch[channel];
            const register double ZERO = 0;
            snrt_ssr_enable();
            // If split sum and dotp compute, need 4 ssr intotal
            // Is the overhead worth?
            asm volatile(
                "frep.o %[n_frep], 5, 0, 0 \n"
                "fadd.d ft3, ft0, %[zero] \n"
                "fadd.d %[sum], ft3, %[sum] \n"
                "fsub.d ft4, ft1, %[curr_mean]\n"
                "fmul.d ft4, ft4, ft3\n"
                "fadd.d %[dotp], ft4, %[dotp]\n"
                : [sum] "+fr"(sum), [dotp] "+fr"(dotp)
                : [curr_mean] "fr"(curr_mean), [zero] "fr"(ZERO),
                  [n_frep] "r"(num_points - 1)
                : "ft0", "ft1", "ft2", "ft3", "ft4");
            snrt_fpu_fence();
            snrt_ssr_disable();

            grad_bias_scratch[channel] = sum;
            dotp_scratch[channel] = dotp;
        }
        __builtin_ssr_barrier(SNRT_SSR_DM1);
    }
    uint32_t end_compute_sum_dotp_reduction = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, point_size_in_bytes);
        snrt_dma_wait(weight_load);
    } else if (compute_id == 0) {
        register double num_points_reg = num_points;
        const register double ZERO = 0;
        snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, C, sizeof(double));

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, grad_weight_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, dotp_scratch);
        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fmul.d ft1, ft0, ft2 \n"
            :
            : [n_frep] "r"(C - 1)
            : "ft0", "ft1", "ft2");
        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, invstd_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, k_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, grad_weight_scratch);
        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 2, 0, 0 \n"
            "fmul.d ft3, ft0, ft2 \n"
            "fdiv.d ft1, ft3, %[num_points] \n"
            :
            : [n_frep] "r"(C - 1), [num_points] "fr"(num_points_reg),
              [zero] "fr"(ZERO)
            : "ft0", "ft1", "ft2", "ft3", "ft4");
        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, grad_bias_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, grad_mean_scratch);
        snrt_ssr_enable();
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fdiv.d ft1, ft0, %[num_points] \n"
            :
            : [n_frep] "r"(C - 1), [num_points] "fr"(num_points_reg)
            : "ft0", "ft1", "ft2");
        snrt_fpu_fence();
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();
    }
    uint32_t end_compute_k_grad_mean = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          point_size_in_bytes);
    } else if (compute_id == 0) {
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, num_points, C, point_size_in_bytes,
                         sizeof(double));
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, grad_ofmap_scratch);
        for (uint32_t channel = 0; channel < C; ++channel) {
            register double curr_mean = curr_mean_scratch[channel];
            register double k = k_scratch[channel];
            register double grad_mean = grad_mean_scratch[channel];
            register double invstd = invstd_scratch[channel];
            register double weight = weight_scratch[channel];
            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 6, 0, 0 \n"
                "fsub.d ft3, ft0, %[curr_mean] \n"
                "fmul.d ft4, ft3, %[k] \n"
                "fsub.d ft4, ft2, ft4 \n"
                "fsub.d ft4, ft4, %[grad_mean] \n"
                "fmul.d ft4, ft4, %[invstd] \n"
                "fmul.d ft1, ft4, %[weight] \n"
                :
                : [curr_mean] "fr"(curr_mean), [k] "fr"(k),
                  [grad_mean] "fr"(grad_mean), [invstd] "fr"(invstd),
                  [weight] "fr"(weight), [n_frep] "r"(num_points - 1)
                : "ft0", "ft1", "ft2", "ft3", "ft4");
            snrt_fpu_fence();
            snrt_ssr_disable();
        }
        __builtin_ssr_barrier(SNRT_SSR_DM1);
    }
    uint32_t end_compute_grad_ifmap = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch, num_bytes);
    }
    uint32_t end_dma_writeback = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
}