// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include <stdbool.h>
#include "printf.h"
#include "snrt.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    uint32_t TILE_CI;
    double *ifmap;
    double *ofmap;
    double *gamma;
    double *beta;

    float eps;
    precision_t dtype;
} batchnorm_layer_t;

typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    // uint32_t TILE_CI;
    double const *ifmap;

    double *ofmap;

    double *running_mean;
    double *running_var;

    double *weight;
    double *bias;

    float eps;
    float momentum;
    precision_t dtype;
} batchnorm_training_layer_t;

typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    // uint32_t TILE_CI;

    double const *ifmap;
    double const *grad_ofmap;
    double const *running_mean;
    double const *running_var;
    double const *weight;

    double *grad_ifmap;
    double *grad_weight;
    double *grad_bias;

    float eps;
    precision_t dtype;
} batchnorm_backward_layer_t;

static inline uint32_t get_core_num_work_items(uint32_t num_work_items,
                                               uint32_t num_compute_cores,
                                               uint32_t compute_id) {
    return num_work_items / num_compute_cores +
           (compute_id < (num_work_items % num_compute_cores));
}

/**
 * @brief implementation of a FP64 batchnorm as a linear combination
 * y = gamma * x + beta
 *
 * @param ifmap pointer to input feature map
 * @param gamma pointer to gamma
 * @param beta pointer to beta
 * @param ofmap pointer to output feature map
 * @param OW width of output feature map
 * @param CI number of input channels
 * @param num_compute_cores number of compute units
 * @param setup_SSR setup SSR strides and bounds
 */
static inline void batchnorm_fp64(
    double *ifmap, double *gamma, double *beta, double *ofmap,
    uint32_t OW,  // number of pixels in a row
    uint32_t CI,  // number of channels per pixel in ifmap
    uint32_t num_compute_cores, uint32_t setup_SSR) {
    // initial SSR setup
    // Dimension 1 is pixel
    // Dimension 2 is channel
    if (setup_SSR) {
        uint32_t ssr_bounds[2] = {
            OW,  // number of pixels each core will have to deal with
            CI / num_compute_cores  // number of channels each core will have to
                                    // deal with per pixel. ASSUMES divisible
        };
        uint32_t ssr_strides[2] = {
            CI * sizeof(double),                // Each pixel is this far apart
            num_compute_cores * sizeof(double)  // Each channel a specific core
                                                // deals with is this far apart
        };

        snrt_ssr_loop_2d(SNRT_SSR_DM0, ssr_bounds[0], ssr_bounds[1],
                         ssr_strides[0], ssr_strides[1]);
        snrt_ssr_loop_2d(SNRT_SSR_DM1, ssr_bounds[0], ssr_bounds[1],
                         ssr_strides[0], ssr_strides[1]);
    }

    // SSR address setup. reads/writes to these registers will now be memory
    // accesses to these pointers in the pattern specified above
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, ofmap);
    snrt_ssr_enable();

    for (uint32_t ci = 0; ci < CI; ci += num_compute_cores) {
        register double g = gamma[ci];
        register double b = beta[ci];

        // frep over OW dimension
        asm volatile(
            "frep.o %[n_frep], 1, 0, 0 \n"
            "fmadd.d ft1, ft0, %[gamma], %[beta] \n"
            :
            : [gamma] "f"(g), [beta] "f"(b),
              [n_frep] "r"(OW - 1)  // we repeat n_frep+1 times
            : "ft0", "ft1", "ft2");
    }
    snrt_fpu_fence();
    // wait for writes to the ofmap to finish?
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void batchnorm_layer(const batchnorm_layer_t *l) {
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
    uint32_t OH = l->IH;
    uint32_t OW = l->IW;
    uint32_t CO = l->CI;

    // Each cluster loads one tile of a row
    // the tile being W x C?
    uint32_t ifmap_size =
        2 * l->IW * l->TILE_CI;     // one for read one for write or smth?
    uint32_t weights_size = l->CI;  // we need C * sizeof(double) space
    uint32_t ofmap_size = 2 * l->IW * l->TILE_CI;

    // here i think we're assuming it's already in dram or smth?
    // no, i think we are just making our own little scratchpad

    // dataflow:
    double *ptr = (double *)snrt_l1_start_addr();
    double *ifmap = ptr;
    ptr += ifmap_size;
    double *gamma = ptr;
    ptr += weights_size;
    double *beta = ptr;
    ptr += weights_size;
    double *ofmap = ptr;
    ptr += ofmap_size;

    uint32_t read_buf = 0;
    uint32_t write_buf = 0;

    uint32_t prev_oh;
    uint32_t prev_ow;
    uint32_t prev_ci;

    // iterate over the rows of the output, skipping by num_clusters each time
    // clusters are responsible for a set of rows
    for (uint32_t row_idx = cluster_id; row_idx < OH; row_idx += num_clusters) {
        // perform transformation for each pixel, but only for channels
        // [ci,ci+l->tile_ci)
        for (uint32_t ci = 0; ci < l->CI; ci += l->TILE_CI) {
            if (snrt_is_dm_core()) {
                // Load weights once in the beginning
                if (row_idx == cluster_id && ci == 0) {
                    snrt_dma_start_1d(gamma, l->gamma, sizeof(double) * l->CI);
                    snrt_dma_start_1d(beta, l->beta, sizeof(double) * l->CI);
                    // q: why do we need to block here?
                    snrt_dma_wait_all();
                }

                // Load a row of pixels, but only channels [ci, ci+l->tile_ci)
                // art: i hate this comment
                if (l->TILE_CI == l->CI) {
                    // data layout is consecutively in memory, so we can just
                    // load the whole row
                    snrt_dma_start_1d(
                        &ifmap[write_buf * ifmap_size /
                               2],  // use half the write buffer
                        &l->ifmap[row_idx * l->IW *
                                  l->CI],  // IW*CI is W * C, so we are starting
                                           // from the base of a row
                        sizeof(double) * l->IW *
                            l->CI);  // copy the whole thing W * C
                } else {
                    // data is interleaved
                    // Guess: Suppose we have CI=3 with RGBRGBRGBRGB
                    // Then if TILE_CI=1, we are loading all RRRRRR
                    // contiguously?
                    snrt_dma_start_2d(
                        &ifmap[write_buf * ifmap_size / 2],      /* dst */
                        &l->ifmap[row_idx * l->IW * l->CI + ci], /* src */
                        sizeof(double) * l->TILE_CI,             /* size */
                        sizeof(double) * l->TILE_CI, /* dst_stride */
                        sizeof(double) * l->CI,      /* src_stride */
                        l->IW);                      /* repetitions */
                }

                snrt_dma_wait_all();

                // Notify compute cores that data has been written into
                // scratchpad
                snrt_cluster_hw_barrier();

                // If this is not the first execution, write out the computed
                // values of previous iteration from scratchpad back out to
                // memory
                if (!(row_idx == cluster_id && ci == 0)) {
                    if (l->TILE_CI == l->CI) {
                        // data is stored consecutively
                        snrt_dma_start_1d(&l->ofmap[prev_oh * OW * l->CI],
                                          &ofmap[!read_buf * (ofmap_size / 2)],
                                          sizeof(double) * l->IW * l->CI);
                    } else {
                        // data is stored in interleaved layout
                        snrt_dma_start_2d(
                            &l->ofmap[prev_oh * OW * l->CI + prev_ci], /* dst */
                            &ofmap[!read_buf * (ofmap_size / 2)],      /* src */
                            sizeof(double) * l->TILE_CI, /* size */
                            sizeof(double) * l->CI,      /* dst_stride */
                            sizeof(double) * l->TILE_CI, /* src_stride */
                            l->IW);                      /* repetitions */
                    }
                }

                snrt_dma_wait_all();
                // Swap read/write buffers for next DMA iteration
                write_buf = !write_buf;
                read_buf = !read_buf;
                // Record previous tile to write out on next iteration
                prev_ci = ci;
                prev_oh = row_idx;
                /* prev_ow = ow; */
            }

            if (snrt_is_compute_core()) {
                // Wait for data
                snrt_cluster_hw_barrier();
                // initially setup SSRs
                // q: can't we put this into a different function?
                uint32_t setup_SSR = (row_idx == cluster_id && ci == 0);

                // Start kernel
                batchnorm_fp64(&ifmap[read_buf * ofmap_size / 2 + compute_id],
                               &gamma[ci + compute_id], &beta[ci + compute_id],
                               &ofmap[write_buf * ofmap_size / 2 + compute_id],
                               OW, l->TILE_CI, num_compute_cores, setup_SSR);

                write_buf = !write_buf;
                read_buf = !read_buf;
            }
        }
    }

    snrt_cluster_hw_barrier();

    // Store last tile back
    if (snrt_is_dm_core()) {
        if (l->TILE_CI == l->CI) {
            // data is stored consecutively
            snrt_dma_start_1d(&l->ofmap[prev_oh * OW * l->CI],
                              &ofmap[!read_buf * (ofmap_size / 2)],
                              sizeof(double) * l->IW * l->CI);
        } else {
            // data is stored in interleaved layout
            snrt_dma_start_2d(
                &l->ofmap[prev_oh * OW * l->CI + prev_ci], /* dst */
                &ofmap[!read_buf * (ofmap_size / 2)],      /* src */
                sizeof(double) * l->TILE_CI,               /* size */
                sizeof(double) * l->CI,                    /* dst_stride */
                sizeof(double) * l->TILE_CI,               /* src_stride */
                l->IW);                                    /* repetitions */
        }

        snrt_dma_wait_all();
    }
}

static inline void batchnorm_training(batchnorm_training_layer_t *layer) {
    // collect stats
    // update running mean and running var
    // pass << batch mean and batch var >> as parameters to compute beta/gamma
    // call batchnorm_layer
}
static inline void batchnorm_backward_tile(
    const double *grad_ofmap_scratch, double *grad_ifmap_scratch,
    const double *ifmap_scratch,
    const double *running_mean_times_invstd_scratch,
    const double *weight_times_invstd_scratch, const double *invstd_scratch,
    double *grad_bias_scratch, double *grad_weight_scratch, uint32_t C,
    uint32_t num_compute_cores, uint32_t num_points_work_in_tile,
    bool is_first_iteration, bool is_last_iteration) {
    // access pattern: iterate over the different channels, then over
    // the different points split over points
    // outside loop: channels
    // inside loop: points
    if (is_first_iteration || is_last_iteration) {
        snrt_ssr_loop_2d(
            SNRT_SSR_DM_ALL,
            num_points_work_in_tile,                 // dimension of inner loop
            C,                                       // dimension of outer loop
            num_compute_cores * C * sizeof(double),  // stride per inner loop
                                                     // iteration
            sizeof(double));  // stride per outer loop iteration
    }
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
    for (uint32_t channel = 0; channel < C; ++channel) {
        register double running_mean_times_invstd =
            running_mean_times_invstd_scratch[channel];
        register double weight_times_invstd =
            weight_times_invstd_scratch[channel];
        register double invstd = invstd_scratch[channel];
        register double grad_weight_0 = 0;
        register double grad_weight_1 = 0;
        register double grad_weight_2 = 0;
        register double grad_bias_0 = 0;
        register double grad_bias_1 = 0;
        register double grad_bias_2 = 0;
        register double ZERO = 0;  // can consider fcvt instead
        snrt_ssr_enable();
        // Can only manual unroll 3 times since the max for frep is 16
        if (num_points_work_in_tile >= 3) {
            asm volatile(
                // manual unroll for
                "frep.o %[n_frep], 15, 0, 0 \n"
                "fadd.d ft3, ft0, %[zero] \n"
                "fadd.d ft5, ft0, %[zero] \n"
                "fadd.d ft7, ft0, %[zero] \n"
                "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fmsub.d ft6, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fmsub.d ft8, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fadd.d %[grad_bias_0], ft3, %[grad_bias_0]\n"
                "fadd.d %[grad_bias_1], ft5, %[grad_bias_1]\n"
                "fadd.d %[grad_bias_2], ft7, %[grad_bias_2]\n"
                "fmadd.d %[grad_weight_0], ft4, ft3, %[grad_weight_0]\n"
                "fmadd.d %[grad_weight_1], ft6, ft5, %[grad_weight_1]\n"
                "fmadd.d %[grad_weight_2], ft8, ft7, %[grad_weight_2]\n"
                "fmul.d ft1, ft3, %[weight_times_invstd]\n"
                "fmul.d ft1, ft5, %[weight_times_invstd]\n"
                "fmul.d ft1, ft7, %[weight_times_invstd]\n"
                : [grad_weight_0] "+fr"(grad_weight_0),
                  [grad_weight_1] "+fr"(grad_weight_1),
                  [grad_weight_2] "+fr"(grad_weight_2),
                  [grad_bias_0] "+fr"(grad_bias_0),
                  [grad_bias_1] "+fr"(grad_bias_1),
                  [grad_bias_2] "+fr"(grad_bias_2)
                : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
                  [weight_times_invstd] "fr"(weight_times_invstd),
                  [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                  [n_frep] "r"(num_points_work_in_tile / 3 -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
                  "ft8");
        }
        switch (num_points_work_in_tile % 3) {
            case 2:
                asm volatile(
                    "fadd.d ft3, ft0, %[zero] \n"
                    "fmsub.d ft4, ft2, %[invstd], "
                    "%[running_mean_times_invstd]\n"
                    "fadd.d ft5, ft0, %[zero] \n"
                    "fmsub.d ft6, ft2, %[invstd], "
                    "%[running_mean_times_invstd]\n"
                    "fadd.d %[grad_bias_0], ft3, %[grad_bias_0]\n"
                    "fadd.d %[grad_bias_1], ft5, %[grad_bias_1]\n"
                    "fmul.d ft1, ft3, %[weight_times_invstd]\n"
                    "fmul.d ft1, ft5, %[weight_times_invstd]\n"
                    "fmadd.d %[grad_weight_0], ft4, ft3, %[grad_weight_0]\n"
                    "fmadd.d %[grad_weight_1], ft6, ft5, %[grad_weight_1]\n"
                    : [grad_weight_0] "+fr"(grad_weight_0),
                      [grad_weight_1] "+fr"(grad_weight_1),
                      [grad_bias_0] "+fr"(grad_bias_0),
                      [grad_bias_1] "+fr"(grad_bias_1)
                    : [running_mean_times_invstd] "fr"(
                          running_mean_times_invstd),
                      [weight_times_invstd] "fr"(weight_times_invstd),
                      [invstd] "fr"(invstd), [zero] "fr"(ZERO)
                    : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
                break;

            case 1:
                asm volatile(
                    "fadd.d ft3, ft0, %[zero] \n"
                    "fmsub.d ft4, ft2, %[invstd], "
                    "%[running_mean_times_invstd]\n"
                    "fadd.d %[grad_bias_0], ft3, %[grad_bias_0]\n"
                    "fmul.d ft1, ft3, %[weight_times_invstd]\n"
                    "fmadd.d %[grad_weight_0], ft4, ft3, %[grad_weight_0]\n"
                    : [grad_weight_0] "+fr"(grad_weight_0), [grad_bias_0] "+fr"(
                                                                grad_bias_0)
                    : [running_mean_times_invstd] "fr"(
                          running_mean_times_invstd),
                      [weight_times_invstd] "fr"(weight_times_invstd),
                      [invstd] "fr"(invstd), [zero] "fr"(ZERO)
                    : "ft0", "ft1", "ft2", "ft3", "ft4");
                break;
        }

        snrt_fpu_fence();
        // wait for writes to the ofmap to finish?
        snrt_ssr_disable();

        if (is_first_iteration) {
            grad_bias_scratch[channel] =
                grad_bias_0 + grad_bias_1 + grad_bias_2;
            grad_weight_scratch[channel] =
                grad_weight_0 + grad_weight_1 + grad_weight_2;

        } else {
            grad_bias_scratch[channel] +=
                grad_bias_0 + grad_bias_1 + grad_bias_2;
            grad_weight_scratch[channel] +=
                grad_weight_0 + grad_weight_1 + grad_weight_2;
        }
    }
    __builtin_ssr_barrier(SNRT_SSR_DM1);
}

static inline void batchnorm_backward(batchnorm_backward_layer_t *l) {
    uint32_t start = snrt_mcycle();
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
    uint32_t num_points = N * H * W;

    uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);
    uint32_t num_points_work_per_channel_for_core =
        get_core_num_work_items(num_points, num_compute_cores, compute_id);

    // thought: how could I minimize the # of reads to grad_ofmap?
    // dy is used for: grad_bias (addition)
    //                 grad_weight (dy * (x[i,C]-running_mean[C]) * invstd[C])
    //                             (can it become a fused somehow? not really..
    //                             can precompute invstd * running_mean though)
    //                             then you get an fmsub(x[i,C], invstd[C],
    //                             invstd[C]*-running_mean[C])
    //                 grad_ifmap (dy * invstd[C] * weight[C])
    // from this I think that the best result is to tile dy and x.
    // need to also tile the write out to grad_ifmap. This fills up all 3 ssrs.

    ptrdiff_t grad_bias_scratch_len = C * num_compute_cores,
              grad_weight_scratch_len = C * num_compute_cores;

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

    // here, we have the more unbounded ones.
    // How much? util/trace/stack.py stack pointer shows that currently core
    // 8 uses up to 0x1001df58. So probably fine to use up to 0x18000 or
    // 64KB. can tune might change if this is called from external, of
    // course. But currently no way to inspect the stack Ideally the len is
    // a multiple of C so ALIGN_DOWN, C but also with this, we need to
    // consider what happens with the final modulo as well (if not
    // multiple).. hmm probably have to recompute the point_work
    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   (snrt_l1_end_addr() - snrt_l1_start_addr()) /
                       4);  // use 3/4 for now
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;
    // C doubles per point (assume fp64)
    // We want two halves to work with
    // We need three buffers, one for grad_ofmap, one for grad_ifmap, one for
    // ifmap
    ptrdiff_t tile_size_in_points = min((space_left / C) / 2 / 3, num_points);
    DUMP(tile_size_in_points);

    ptrdiff_t grad_ofmap_len = tile_size_in_points * C * 2,
              grad_ifmap_len = grad_ofmap_len, ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *grad_ifmap_scratch = ptr;
    ptr += grad_ifmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;

    bool buf_flag = 0;

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
    } else {
        // PRECONFIGURE: operations on arrays of size C, split by core.
        snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, num_channels_work_for_core,
                         num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = snrt_mcycle();
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_invstd_calc = snrt_mcycle();
    if (snrt_is_dm_core()) {
        weight_load =
            snrt_dma_start_1d(weight_scratch, l->weight, C * sizeof(double));
        running_mean_load = snrt_dma_start_1d(
            running_mean_scratch, l->running_mean, C * sizeof(double));
        // load first tile in. We can do this here because sqrt/div are really
        // slow.
        grad_ofmap_load =
            snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap,
                              tile_size_in_points * C * sizeof(double));
        ifmap_load = snrt_dma_start_1d(
            ifmap_scratch, l->ifmap, tile_size_in_points * C * sizeof(double));
        buf_flag = !buf_flag;
        snrt_dma_wait(weight_load);
        snrt_dma_wait(running_mean_load);
    } else {
        if (num_channels_work_for_core > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D,
                          &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D,
                           &invstd_scratch[compute_id]);
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
                  [n_frep] "r"(num_channels_work_for_core -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3");

            snrt_fpu_fence();                     // thought: do we need this?
            __builtin_ssr_barrier(SNRT_SSR_DM1);  // thought: do we need this?
            snrt_ssr_disable();
        }
    }
    uint32_t end_invstd_calc = snrt_mcycle();
    snrt_cluster_hw_barrier();

    // compute weight*invstd and running_mean*invstd

    // computing invstd scratch and using it for weight: can we do it in 1 frep?
    // load running var: 1 ssr
    // write running var: 1 ssr
    // load weight: 1 ssr
    // write weight: 1 ssr
    // answer: not really. Still worth precomputing I think
    uint32_t start_running_var_weight_inplace_mul = snrt_mcycle();
    if (snrt_is_dm_core()) {
    } else {
        if (num_channels_work_for_core > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D,
                          &weight_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D,
                           &weight_scratch[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D,
                          &invstd_scratch[compute_id]);

            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fmul.d ft1, ft0, ft2 \n"
                :
                : [n_frep] "r"(num_channels_work_for_core -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2");

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D,
                          &running_mean_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D,
                           &running_mean_scratch[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D,
                          &invstd_scratch[compute_id]);

            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fmul.d ft1, ft0, ft2 \n"  // running_mean =
                                           // running_mean * invstd
                :
                : [n_frep] "r"(num_channels_work_for_core -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2");

            snrt_fpu_fence();                     // thought: do we need this?
            __builtin_ssr_barrier(SNRT_SSR_DM1);  // thought: do we need this?
            snrt_ssr_disable();
        }
    }
    uint32_t end_running_var_weight_inplace_mul = snrt_mcycle();
    snrt_cluster_hw_barrier();

    // compute grad_weight first step, grad_bias first step, grad_ifmap
    // this is where the tiling would come in place

    uint32_t start_grad_ifmap_and_partial_bias_weight = snrt_mcycle();
    uint32_t num_points_work_in_tile, num_points_work_for_core_for_tile;
    // for dma core
    uint32_t prev_point_start, num_points_work_in_prev_tile;
    bool is_last_iteration = false;
    for (uint32_t point_start = 0; point_start < num_points;
         point_start += tile_size_in_points) {
        if (point_start + tile_size_in_points >= num_points) {
            // on last iteration
            is_last_iteration = true;
            num_points_work_in_tile = num_points - point_start;
            num_points_work_for_core_for_tile = get_core_num_work_items(
                num_points_work_in_tile, num_compute_cores,
                compute_id);  // for the last tile, might be different
        } else if (point_start == 0) {
            // on first iteration
            num_points_work_in_tile = tile_size_in_points;

            num_points_work_for_core_for_tile = get_core_num_work_items(
                num_points_work_in_tile, num_compute_cores, compute_id);
        }

        if (snrt_is_dm_core()) {
            // technically we could optimize by loading in both sides ahead of
            // time. For now not going to do that
            if (point_start != 0) {
                // first buffer was already loaded in before
                grad_ofmap_load = snrt_dma_start_1d(
                    &grad_ofmap_scratch[num_points_work_in_tile * C * buf_flag],
                    &l->grad_ofmap[point_start * C],
                    num_points_work_in_tile * C * sizeof(double));
                ifmap_load = snrt_dma_start_1d(
                    &ifmap_scratch[num_points_work_in_tile * C * buf_flag],
                    &l->ifmap[point_start * C],
                    num_points_work_in_tile * C * sizeof(double));
            }
            snrt_dma_wait_all();
            // wait for compute cores to finish computing current tile,
            // signal to them grad_ofmap[buf_flag], ifmap[buf_flag],
            // grad_ifmap[!buf_flag] done?
            snrt_cluster_hw_barrier();
            // write out the buffer that was just computed
            if (point_start != 0) {
                grad_ifmap_write = snrt_dma_start_1d(
                    &l->grad_ifmap[prev_point_start * C],
                    &grad_ifmap_scratch[tile_size_in_points * C *
                                        (!buf_flag)],  // take !buf_flag dma
                                                       // core is one iteration
                                                       // ahead of compute core
                    num_points_work_in_prev_tile * C * sizeof(double));
                buf_flag = !buf_flag;
            }

            prev_point_start = point_start;
            num_points_work_in_prev_tile = num_points_work_in_tile;

        } else {
            // since we didn't flip buf_flag with the dma load, buf_flag starts
            // at 0. Signifies the current tile being worked on.

            // dma core will signal to us when we can start next computation
            snrt_cluster_hw_barrier();
            batchnorm_backward_tile(
                &grad_ofmap_scratch
                    [(buf_flag * tile_size_in_points + compute_id) * C],
                &grad_ifmap_scratch
                    [(buf_flag * tile_size_in_points + compute_id) * C],
                &ifmap_scratch[(buf_flag * tile_size_in_points + compute_id) *
                               C],
                running_mean_scratch, weight_scratch, invstd_scratch,
                &grad_bias_scratch[compute_id * C],
                &grad_weight_scratch[compute_id * C], C, num_compute_cores,
                num_points_work_for_core_for_tile, point_start == 0,
                is_last_iteration);
            buf_flag = !buf_flag;
        }
    }

    uint32_t end_grad_ifmap_and_partial_bias_weight = snrt_mcycle();
    // no barrier here so that compute core can immediately start the second
    // reduction
    snrt_cluster_hw_barrier();

    // reduce from [num_threads, C] to [C] by splitting over C
    // just reduce back into the first buffer.
    uint32_t start_grad_bias_weight_reduction_2 = snrt_mcycle();
    if (snrt_is_dm_core()) {
        // write out the last tile
        grad_ifmap_write = snrt_dma_start_1d(
            &l->grad_ifmap[prev_point_start * C],
            &grad_ifmap_scratch[tile_size_in_points * C *
                                (!buf_flag)],  // take !buf_flag dma
                                               // core is one iteration
                                               // ahead of compute core
            num_points_work_in_prev_tile * C * sizeof(double));
    } else {
        for (uint32_t channel = compute_id; channel < C;
             channel += num_compute_cores) {
            register volatile double grad_bias_sum = 0;
            register volatile double grad_weight_sum = 0;
            snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, num_compute_cores,
                             C * sizeof(double));
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D,
                          &grad_bias_scratch[channel]);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D,
                          &grad_weight_scratch[channel]);

            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 2, 0, 0 \n"
                "fadd.d %[bias_sum], ft0, %[bias_sum] \n"
                "fadd.d %[weight_sum], ft1, %[weight_sum] \n"
                // NOTE: floating point addition is 3 cycles, causing stalls
                // here
                : [bias_sum] "+fr"(grad_bias_sum), [weight_sum] "+fr"(
                                                       grad_weight_sum)
                : [n_frep] "r"(num_compute_cores -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2");
            snrt_fpu_fence();
            snrt_ssr_disable();
            grad_bias_scratch[0 * C + channel] = grad_bias_sum;
            grad_weight_scratch[0 * C + channel] = grad_weight_sum;
        }
    }
    uint32_t end_grad_bias_weight_reduction_2 = snrt_mcycle();
    snrt_cluster_hw_barrier();
    // write back grad_bias and grad_weight. then wait for all transactions to
    // complete
    uint32_t start_dma_writeback = snrt_mcycle();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, C * sizeof(double));
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          C * sizeof(double));
        snrt_dma_wait_all();
    } else {
    }
    uint32_t end_dma_writeback = snrt_mcycle();
    snrt_cluster_hw_barrier();
}

static inline void batchnorm_backwards_training(batchnorm_backward_layer_t *l) {
    const uint32_t num_clusters = snrt_cluster_num();
    const uint32_t cluster_id = snrt_cluster_idx();
    const uint32_t num_compute_cores = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx(); 

    // Calculate output dimensions
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    double eps = l->eps;
    uint32_t num_points = N * H * W;

    uint32_t buffer_len = C;
    uint32_t invstd_scratch_len = C;
    uint32_t grad_xi_hat_scratch_len = C;
    uint32_t grad_var_scratch_len = C;
    uint32_t grad_mean_scratch_len = C;
    uint32_t grad_bias_scratch_len = C * num_compute_cores,
             grad_weight_scratch_len = C * num_compute_cores;
    uint32_t grad_ofmap_len = num_points * C, 
             grad_ifmap_len = num_points * C;

    // dataflow:
    double *ptr = (double *)snrt_l1_start_addr();
    double *invstd_scratch = ptr;
    ptr += invstd_scratch_len;
    double *grad_xi_hat_scratch = ptr;
    ptr += grad_xi_hat_scratch_len;
    double *grad_var_scratch = ptr;
    ptr += grad_var_scratch_len;
    double *grad_mean_scratch = ptr;
    ptr += grad_mean_scratch_len;
    double *grad_bias_scratch = ptr;
    ptr += grad_bias_scratch_len;
    double *grad_weight_scratch = ptr;
    ptr += grad_weight_scratch_len;
    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *grad_ifmap_scratch = ptr;
    ptr += grad_ifmap_len;

    // Compute grad_bias_scratch and grad_weight_scratch
    
    // grad_bias_scratch = sum of grad_yi
    // grad_weight_scratch = sum of grad_yi * xi_hat 
    //                       where (xi_hat = (xi - mean) / sqrt(var + eps))
    if (snrt_is_dm_core()) {
        // might need to tile this later
        snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap,
                          grad_ofmap_len * sizeof(double));
        snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
    } else if (snrt_is_compute_core()) {
        uint32_t start_invstd_comp = snrt_mcycle();
        for (uint32_t channel = compute_id; channel < C;
             channel += num_compute_cores) {
            invstd_scratch[channel] = 1 / sqrt(l->running_var[channel] + eps);
        }
        uint32_t end_invstd_comp = snrt_mcycle();
        // wait for grad_ofmap to be loaded in
        snrt_fpu_fence();
        snrt_cluster_hw_barrier();
        // reduce from [num_points, C] to [num_threads, C] by splitting over
        // num_points 
        // Read from ofmap.
        uint32_t start_grad_bias_weight_reduction_1 = snrt_mcycle();
        for (uint32_t i = compute_id; i < num_points; i += num_compute_cores) {
            for (uint32_t channel = 0; channel < C; ++channel) {
                double dy = grad_ofmap_scratch[i * C + channel];
                double x = l->ifmap[i * C + channel];
                double mean = l->running_mean[channel];
                double var = l->running_var[channel];
                double dot_res = dy * (x - mean) / sqrt(var + eps);
                // currently accessing channel 1, 2, 3, 4, 1, 2, 3, 4
                grad_bias_scratch[compute_id * C + channel] +=
                    grad_ofmap_scratch[i * C + channel];
                grad_weight_scratch[compute_id * C + channel] +=
                    dot_res * invstd_scratch[channel];
            }
        }
        uint32_t end_grad_bias_weight_reduction_1 = snrt_mcycle();
        snrt_fpu_fence();
        snrt_cluster_hw_barrier();
        // reduce from [num_threads, C] to [C] by splitting over C
        // just reduce back into the first buffer.
        uint32_t start_grad_bias_weight_reduction_2 = snrt_mcycle();
        for (uint32_t channel = compute_id; channel < C;
             channel += num_compute_cores) {
            register volatile double grad_bias_sum = 0;
            register volatile double grad_weight_sum = 0;
            snrt_ssr_loop_1d(SNRT_SSR_DM0, num_compute_cores,
                             C * sizeof(double));
            snrt_ssr_loop_1d(SNRT_SSR_DM1, num_compute_cores,
                             C * sizeof(double));
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D,
                          &grad_bias_scratch[channel]);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D,
                          &grad_weight_scratch[channel]);

            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 2, 0, 0 \n"
                "fadd.d %[bias_sum], ft0, %[bias_sum] \n"
                "fadd.d %[weight_sum], ft1, %[weight_sum] \n"
                : [bias_sum] "+fr"(grad_bias_sum), [weight_sum] "+fr"(
                                                       grad_weight_sum)
                : [n_frep] "r"(num_compute_cores -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2");
            snrt_fpu_fence();
            snrt_ssr_disable();
            grad_bias_scratch[0 * C + channel] = grad_bias_sum;
            grad_weight_scratch[0 * C + channel] = grad_weight_sum;
        }

        uint32_t end_grad_bias_weight_reduction_2 = snrt_mcycle();
    }
    snrt_cluster_hw_barrier();

    // Compute grad_ifmap_scratch

    // invstd = 1 / sqrt(var + eps)
    // grad_xi_hat_scratch = grad_yi * weight
    // grad_var_scratch = sum of grad_xi_hat_scratch * (xi - mean) 
    //                    * -1/2 * (var + eps)^(-3/2)
    // grad_mean_scratch = sum of grad_xi_hat_scratch * - invstd
    // grad_ifmap_scratch = grad_xi_hat_scratch * 1 / sqrt(var + eps) 
    //                   + grad_var_scratch * 2 * (xi - mean) / N 
    //                   + grad_mean_scratch / N
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, C * sizeof(double));
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          C * sizeof(double));
        // wait for compute of grad_ifmap
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();
        snrt_cluster_hw_barrier();

        snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch,
                          grad_ifmap_len * sizeof(double));
        snrt_dma_wait_all();
    } else if (snrt_is_compute_core()) {
        uint32_t start_invstd_scratch_inplace_augmentation = snrt_mcycle();
        for (uint32_t channel = compute_id; channel < C;
             channel += num_compute_cores) {
            invstd_scratch[channel] *=
                l->weight[channel];  // can dma weight, can also frep
        }
        uint32_t end_invstd_scratch_inplace_augmentation = snrt_mcycle();
        snrt_fpu_fence();
        snrt_cluster_hw_barrier();

        uint32_t start_grad_xi_hat_computation = snrt_mcycle();
        for (uint32_t channel = 0; channel < C; ++channel) {
            grad_xi_hat_scratch[i * C + channel] =
                grad_ofmap_scratch[i * C + channel] *
                weight[channel];
        }
        uint32_t end_grad_xi_hat_computation = snrt_mcycle();
        snrt_fpu_fence();
        snrt_cluster_hw_barrier();

        uint32_t grad_var_scratch_computation = snrt_mcycle();
        for (uint32_t channel = compute_id; channel < C;
             channel += num_compute_cores) {
            double sum = 0;
            for (uint32_t i = 0; i < num_points; ++i) {
                sum += grad_xi_hat_scratch[i * C + channel] *
                       (l->ifmap[i * C + channel] - l->running_mean[channel]);
            }
            grad_var_scratch[channel] = sum * -1 / 2 * pow(l->running_var[channel] + eps, -3 / 2);
        }
        uint32_t end_grad_var_scratch_computation = snrt_mcycle();
        snrt_fpu_fence();
        snrt_cluster_hw_barrier();

        uint32_t grad_mean_scratch_computation = snrt_mcycle();
        for (uint32_t channel = compute_id; channel < C;
             channel += num_compute_cores) {
            double sum = 0;
            for (uint32_t i = 0; i < num_points; ++i) {
                sum += grad_xi_hat_scratch[i * C + channel] *
                       - invstd_scratch[channel];
            }
            grad_mean_scratch[channel] = sum;
        }
        uint32_t end_grad_mean_scratch_computation = snrt_mcycle();
        snrt_fpu_fence();
        snrt_cluster_hw_barrier();

        uint32_t grad_ifmap_scratch_computation = snrt_mcycle();
        for (uint32_t channel = compute_id; channel < C;
             channel += num_compute_cores) {
            for (uint32_t i = 0; i < num_points; ++i) {
                double x = l->ifmap[i * C + channel];
                double mean = l->running_mean[channel];
                grad_ifmap_scratch[i * C + channel] =
                    grad_xi_hat_scratch[i * C + channel] *
                        invstd_scratch[channel] +
                    grad_var_scratch[channel] *
                        2 * (x - mean) / N +
                    grad_mean_scratch[channel] / N;
            }
        }
        uint32_t end_grad_ifmap_scratch_computation = snrt_mcycle();
        snrt_fpu_fence();
        snrt_cluster_hw_barrier();
    }
}
