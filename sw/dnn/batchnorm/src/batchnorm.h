// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <stdbool.h>

#include "dnn.h"

#include "batchnorm_data_structures.h"
#include "batchnorm_reference.h"
#include "batchnorm_utils.h"
#include "dnn.h"
#include "printf.h"
#include "snrt.h"

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define ceildiv(a, b) ((((a)-1) / (b)) + 1)

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

static inline void batchnorm_backward(batchnorm_backward_layer_t *l) {
    uint32_t start = snrt_mcycle();
    // data is in NHWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT0,
                                     SNRT_PERF_CNT_ICACHE_STALL);
    reset_and_start_perf_single_core(compute_id, SNRT_PERF_CNT1,
                                     SNRT_PERF_CNT_TCDM_CONGESTED);
    // reset_and_start_perf_single_core(0, SNRT_PERF_CNT1,
    //                                  SNRT_PERF_CNT_ICACHE_HIT);
    // reset_and_start_perf_single_core(0, SNRT_PERF_CNT2,
    //                                  SNRT_PERF_CNT_ICACHE_MISS);
    // reset_and_start_perf_single_core(0, SNRT_PERF_CNT3,
    //                                  SNRT_PERF_CNT_ICACHE_DOUBLE_HIT);
    // reset_and_start_perf_single_core(0, SNRT_PERF_CNT4,
    //                                  SNRT_PERF_CNT_ICACHE_PREFETCH);
    // Calculate output dimensions

    // thought: this is so much contention
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    uint32_t num_points = N * H * W;

    const uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);
    const uint32_t channel_block_offset =
        get_offset_for_core_work_blocked(C, num_compute_cores, compute_id);

    ptrdiff_t grad_bias_scratch_len = C, grad_weight_scratch_len = C;

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t) * 2;
    double *ptr = (double *)raw_ptr;
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

    // Dynamically compute tile sizes
    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   ((1 << SNRT_LOG2_STACK_SIZE) + 8) *
                       (snrt_cluster_core_num() + 1));  // + 1 for safety
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;
    // first 2: ofmap, ifmap (overlaid with grad_ifmap)
    // second 2: double buffer
    // C: there are C channels per point
    ptrdiff_t tile_size_in_points = (space_left) / (2 * 2 * C);

    ptrdiff_t grad_ofmap_len = tile_size_in_points * C * 2;
    ptrdiff_t grad_ifmap_len = grad_ofmap_len, ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;
    double *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    bool buf_flag = 0;

    snrt_dma_txid_t running_var_load, weight_load, running_mean_load,
        grad_ofmap_load, ifmap_load, grad_ifmap_write;

    // Incrementally increase tile size.
    // Reason is because we want to minimize wait on the first iteration
    // how much time do i have? C/num_compute_cores * 40-50, approximately.
    // estimate 7 doubles per cycle
    // fixed cost below 1KB=128 doubles is too high.
    uint32_t doubles_loadable =
        max(ceildiv(C, num_compute_cores) * 50 * 7, 128);
    uint32_t points_loadable = doubles_loadable / C;
    uint32_t work_in_tile =
        min(min(points_loadable, tile_size_in_points), num_points);
    // num_points = num_points/2;
    uint32_t work_left = num_points;
    uint32_t work_mod_2 = work_in_tile % 2;
    uint32_t work_div_2_sub_1 = work_in_tile / 2 - 1;
    DUMP(work_in_tile);
    DUMP(tile_size_in_points);
    if (snrt_is_dm_core()) {
        work_left -= work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_2 = work_mod_2;
        dm_comm->work_div_2_sub_1 = work_div_2_sub_1;  // this is the frep value
    }

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
        snrt_ssr_loop_1d(SNRT_SSR_DM0, num_channels_work_for_core,
                         num_compute_cores * sizeof(double));
        snrt_ssr_loop_1d(SNRT_SSR_DM1, num_channels_work_for_core,
                         num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    // compute invstd, load weight and running_mean in
    uint32_t start_invstd_calc = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(weight_scratch, l->weight, C * sizeof(double));
        snrt_dma_start_1d(running_mean_scratch, l->running_mean,
                          C * sizeof(double));
        // load first tile in but only as much as we can in parallel while sqrt
        // runs
        snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap,
                          work_in_tile * C * sizeof(double));
        snrt_dma_start_1d(ifmap_scratch, l->ifmap,
                          work_in_tile * C * sizeof(double));

        buf_flag = !buf_flag;
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
    uint32_t end_invstd_calc = SNRT_SECTIONED_MCYCLE();

    uint32_t start_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();

    uint32_t end_running_var_weight_inplace_mul = SNRT_SECTIONED_MCYCLE();
    // if (snrt_is_dm_core()) {
    //     snrt_dma_wait_all();
    //     snrt_dma_start_1d(temp, grad_ofmap_scratch, 8 * C * sizeof(double));
    //     snrt_dma_wait_all();
    // }
    // return;
    // snrt_cluster_hw_barrier();

    // compute grad_weight, grad_bias, grad_ifmap. Tile only if we can't fit all
    // the points in one tile.
    if (work_in_tile == num_points) {
        uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
        // no looping needed
        if (snrt_is_dm_core()) {
            // finish loads
            snrt_dma_wait_all();

            // notify ready
            snrt_cluster_hw_barrier();
            // wait for compute to be done
            snrt_cluster_hw_barrier();
            snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch,
                              work_in_tile * C * sizeof(double));
        } else {
            snrt_cluster_hw_barrier();
            if (num_channels_work_for_core > 0) {
                // TODO; shift these all before hand
                batchnorm_backward_fp64_no_loop(
                    &grad_ofmap_scratch[compute_id],
                    &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                    &running_mean_scratch[compute_id],
                    &weight_scratch[compute_id], &invstd_scratch[compute_id],
                    &grad_bias_scratch[compute_id],
                    &grad_weight_scratch[compute_id], C, work_in_tile,
                    work_mod_2, num_channels_work_for_core, num_compute_cores,
                    true, false);
            }

            snrt_cluster_hw_barrier();
        }

        uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    } else {
        batchnorm_backward_main_loop(
            C, work_left, work_in_tile, work_mod_2, work_div_2_sub_1, dm_comm,
            tile_size_in_points, compute_id, num_compute_cores, l,
            grad_ofmap_scratch, ifmap_scratch, grad_ifmap_scratch,
            grad_weight_scratch, grad_bias_scratch, invstd_scratch,
            running_mean_scratch, weight_scratch, buf_flag);
    }

    uint32_t start_grad_bias_weight_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_grad_bias_weight_reduction_2 = SNRT_SECTIONED_MCYCLE();
    // write back grad_bias and grad_weight. then wait for all transactions to
    // complete
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, grad_bias_scratch, C * sizeof(double));
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch,
                          C * sizeof(double));
        snrt_dma_wait_all();
    } else {
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT0);
    end_perf_and_dump_single_core(compute_id, SNRT_PERF_CNT1);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT2);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT3);
    // end_perf_and_dump_single_core(0, SNRT_PERF_CNT4);
}

static inline void batchnorm_backward_training_tiling(
    batchnorm_backward_training_layer_t *l) {
    uint32_t start = snrt_mcycle();

    // data is in HWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total?
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id = snrt_cluster_core_idx();

    // Calculate output dimensions
    const uint32_t N = 1;
    const uint32_t H = l->IH;
    const uint32_t W = l->IW;
    const uint32_t C = l->CI;
    const uint32_t num_points = N * H * W;
    const uint32_t num_data = num_points * C;

    const uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);
    const uint32_t channel_block_offset =
        get_offset_for_core_work_blocked(C, num_compute_cores, compute_id);
    const uint32_t num_points_work_per_channel_for_core =
        get_core_num_work_items(num_points, num_compute_cores, compute_id);

    ptrdiff_t grad_weight_len = C, sum_len = C, dotp_len = C;

    // dataflow:
    void *raw_ptr = (void *)snrt_l1_start_addr();
    dm_comm_t *dm_comm = (dm_comm_t *)raw_ptr;
    raw_ptr += sizeof(dm_comm_t);
    double *ptr = (double *)raw_ptr;
    double *invstd_scratch = ptr;
    ptr += C;
    double *sum_scratch = ptr;
    ptr += C;
    double *dotp_scratch = ptr;
    ptr += C;
    double *k_scratch = ptr;
    ptr += C;
    double *grad_mean_scratch = ptr;
    ptr += C;
    double *weight_scratch = ptr;
    ptr += C;
    double *current_mean_scratch = ptr;
    ptr += C;
    double *grad_weight_scratch = ptr;
    ptr += C;

    double *used_tcdm_end_addr =
        (double *)(snrt_l1_end_addr() -
                   (snrt_l1_end_addr() - snrt_l1_start_addr()) /
                       4);  // use 3/4 for now
    ptrdiff_t space_left = used_tcdm_end_addr - ptr;
    ptrdiff_t tile_size_in_points = (space_left) / (2 * 2 * C);

    ptrdiff_t grad_ofmap_len = tile_size_in_points * C * 2;
    ptrdiff_t grad_ifmap_len = grad_ofmap_len, ifmap_len = grad_ifmap_len;

    double *grad_ofmap_scratch = ptr;
    ptr += grad_ofmap_len;
    double *ifmap_scratch = ptr;
    ptr += ifmap_len;
    double *grad_ifmap_scratch = ifmap_scratch;  // reuse the buffer

    bool buf_flag = 0;

    snrt_dma_txid_t invstd_load, curr_var_load, weight_load, curr_mean_load,
        grad_ofmap_load, ifmap_load;

    uint32_t doubles_loadable = ceildiv(C, num_compute_cores) * 50 * 7;
    uint32_t points_loadable = doubles_loadable / C;
    uint32_t work_in_tile = min(min(points_loadable, tile_size_in_points), num_points);
    uint32_t work_left = num_points;
    uint32_t work_mod_3 = work_in_tile % 3;
    uint32_t work_div_3_sub_1 = work_in_tile / 3 - 1;
    uint32_t work_mod_4 = work_in_tile % 4;
    uint32_t work_div_4_sub_1 = work_in_tile / 4 - 1;
    if (snrt_is_dm_core()) {
        work_left -= work_in_tile;
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_3 = work_mod_3;
        dm_comm->work_div_3_sub_1 = work_div_3_sub_1;  // this is the frep value
        DUMP(10);
    }

    uint32_t start_dma_load = snrt_mcycle();
    if (snrt_is_dm_core()) {
        grad_ofmap_load = snrt_dma_start_1d(grad_ofmap_scratch, l->grad_ofmap,
                                            grad_ofmap_len * sizeof(double));
        ifmap_load =
            snrt_dma_start_1d(ifmap_scratch, l->ifmap, ifmap_len * sizeof(double));
        curr_mean_load =
            snrt_dma_start_1d(current_mean_scratch, l->current_mean, C * sizeof(double));
        curr_var_load =
            snrt_dma_start_1d(invstd_scratch, l->current_var, C * sizeof(double));
        weight_load = snrt_dma_start_1d(weight_scratch, l->weight, C * sizeof(double));
        snrt_dma_wait(grad_ofmap_load);
        snrt_dma_wait(ifmap_load);
        snrt_dma_wait(curr_mean_load);
        DUMP(11);
    } else if (snrt_is_compute_core()) {
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, num_points,
                         num_channels_work_for_core, C * sizeof(double),
                         num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_invstd_computations = SNRT_SECTIONED_MCYCLE();
    uint32_t end_invstd_computations = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_sum_dotp_reduction = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_wait(curr_var_load);
        DUMP(12);
    } else {
        if (num_channels_work_for_core > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &grad_ofmap_scratch[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &ifmap_scratch[compute_id]);
            for (uint32_t channel = compute_id; channel < C;
                 channel += num_compute_cores) {
                register volatile double sum = 0;
                register volatile double dotp = 0;
                register double current_mean = current_mean_scratch[channel];
                const register double ZERO = 0;
                snrt_ssr_enable();
                // TODO use ssr_repeat instead of fadd.d , , %[zero]
                asm volatile(
                    "frep.o %[n_frep], 5, 0, 0 \n"
                    "fadd.d ft3, ft0, %[zero] \n"
                    "fadd.d %[sum], ft3, %[sum] \n"
                    "fsub.d ft4, ft1, %[current_mean]\n"
                    "fmul.d ft4, ft4, ft3\n"
                    "fadd.d %[dotp], ft4, %[dotp]\n"
                    : [sum] "+fr"(sum), [dotp] "+fr"(dotp)
                    : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
                      [n_frep] "r"(num_points - 1)
                    : "ft0", "ft1", "ft2", "ft3", "ft4");
                snrt_fpu_fence();
                snrt_ssr_disable();
                sum_scratch[channel] = sum;
                dotp_scratch[channel] = dotp;
            }
            __builtin_ssr_barrier(SNRT_SSR_DM1);
        }
    }
    uint32_t end_compute_sum_dotp_reduction = SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();
    uint32_t end_compute_sum_dotp_reduction_2 = SNRT_SECTIONED_MCYCLE();

    uint32_t start_compute_invstd_k_grad_mean_grad_weight =
        SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, sum_scratch, C * sizeof(double));
        dm_comm->num_points_work_in_tile = work_in_tile;
        dm_comm->work_mod_4 = work_mod_4;
        dm_comm->work_div_4_sub_1 = work_div_4_sub_1;
        snrt_dma_wait(weight_load);
        buf_flag = !buf_flag;
    } else if (snrt_is_compute_core()) {
        if (num_channels_work_for_core > 0) {
            register double num_points_reg = num_points;
            const register double ZERO = 0;
            snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, num_channels_work_for_core,
                             num_compute_cores * sizeof(double));

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, &invstd_scratch[compute_id]);
            register double eps = l->eps;
            const register double ONE = 1;
            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 3, 0, 0 \n"
                "fadd.d ft3, ft0, %[eps]\n"
                "fsqrt.d ft3, ft3\n"
                "fdiv.d ft1, %[ONE], ft3\n"
                :
                : [eps] "fr"(eps), [ONE] "fr"(ONE),
                  [n_frep] "r"(num_channels_work_for_core - 1)
                : "ft0", "ft1", "ft2", "ft3");
            __builtin_ssr_barrier(SNRT_SSR_DM1);

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, &grad_weight_scratch[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, &dotp_scratch[compute_id]);
            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fmul.d ft1, ft0, ft2 \n"
                :
                : [n_frep] "r"(num_channels_work_for_core - 1)
                : "ft0", "ft1", "ft2");
            __builtin_ssr_barrier(SNRT_SSR_DM1);

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &invstd_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, &k_scratch[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, &grad_weight_scratch[compute_id]);
            asm volatile(
                "frep.o %[n_frep], 2, 0, 0 \n"
                "fmul.d ft3, ft0, ft2 \n"
                "fdiv.d ft1, ft3, %[num_points] \n"
                :
                : [n_frep] "r"(num_channels_work_for_core - 1),
                  [num_points] "fr"(num_points_reg), [zero] "fr"(ZERO)
                : "ft0", "ft1", "ft2", "ft3", "ft4");
            __builtin_ssr_barrier(SNRT_SSR_DM1);

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &sum_scratch[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, &grad_mean_scratch[compute_id]);
            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fdiv.d ft1, ft0, %[num_points] \n"
                :
                : [n_frep] "r"(num_channels_work_for_core - 1),
                  [num_points] "fr"(num_points_reg)
                : "ft0", "ft1", "ft2");
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }
    }
    uint32_t end_compute_invstd_k_grad_mean_grad_weight =
        SNRT_SECTIONED_MCYCLE();
    snrt_cluster_hw_barrier();

    if (work_in_tile == num_points) {
        uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();
        if (snrt_is_dm_core()) {
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
            snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch,
                              work_in_tile * C * sizeof(double));
        } else {
            snrt_cluster_hw_barrier();
            if (num_channels_work_for_core > 0) {
                batchnorm_backward_training_tile_fp64_no_loop(
                    &grad_ofmap_scratch[compute_id], &grad_ifmap_scratch[compute_id],
                    &ifmap_scratch[compute_id], &current_mean_scratch[compute_id],
                    &weight_scratch[compute_id], &invstd_scratch[compute_id], &k_scratch[compute_id],
                    &grad_mean_scratch[compute_id], C, work_in_tile, work_mod_4, work_div_4_sub_1,
                    num_channels_work_for_core, num_compute_cores, true, false);
            }
            // notify finish
            snrt_cluster_hw_barrier();
        }
        uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
    } else {
        DUMP(0);
        batchnorm_backward_training_main_loop(C, work_left, work_in_tile, work_mod_4, work_div_4_sub_1, dm_comm,
                                     tile_size_in_points, compute_id, num_compute_cores, l, grad_ofmap_scratch,
                                     ifmap_scratch, grad_ifmap_scratch, k_scratch, grad_mean_scratch, invstd_scratch,
                                     current_mean_scratch, weight_scratch, buf_flag);
    }
    uint32_t start_dma_writeback = SNRT_SECTIONED_MCYCLE();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_weight, grad_weight_scratch, C * sizeof(double));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
    uint32_t done = snrt_mcycle();
}

static inline void batchnorm_backward_training(
    batchnorm_backward_training_layer_t *l) {
    // data is in HWC format
    const uint32_t num_compute_cores = snrt_cluster_compute_core_num();
    const uint32_t compute_id = snrt_cluster_core_idx();

    // Calculate output dimensions
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    uint32_t num_points = N * H * W;
    uint32_t num_data = num_points * C;

    uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);
    uint32_t num_points_work_per_channel_for_core =
        get_core_num_work_items(num_points, num_compute_cores, compute_id);

    uint32_t grad_weight_len = C * num_compute_cores, sum_len = C, dotp_len = C;
    uint32_t grad_ofmap_len = num_data, grad_ifmap_len = num_data,
             ifmap_len = num_data;

    // Using TCDM as scratchpad
    double *ptr = (double *)snrt_l1_start_addr();

    // Intermediate value
    double *invstd = ptr;
    ptr += C;
    double *dotp = ptr;
    ptr += dotp_len;
    double *k = ptr;
    ptr += C;
    double *grad_mean = ptr;
    ptr += C;

    // Input value
    double *weight = ptr;
    ptr += C;
    double *curr_mean = ptr;
    ptr += C;
    double *grad_ofmap = ptr;
    ptr += num_data;
    double *ifmap = ptr;
    ptr += num_data;

    // Output
    double *grad_ifmap = ptr;
    ptr += grad_ifmap_len;
    double *grad_weight = ptr;
    ptr += grad_weight_len;
    double *sum = ptr;
    ptr += sum_len;

    // Load data
    snrt_dma_txid_t invstd_load, curr_var_load, weight_load, curr_mean_load,
        grad_ofmap_load, ifmap_load;

    uint32_t start_dma_load = snrt_mcycle();
    if (snrt_is_dm_core()) {
        grad_ofmap_load = snrt_dma_start_1d(grad_ofmap, l->grad_ofmap,
                                            grad_ofmap_len * sizeof(double));
        ifmap_load =
            snrt_dma_start_1d(ifmap, l->ifmap, ifmap_len * sizeof(double));
        curr_mean_load =
            snrt_dma_start_1d(curr_mean, l->current_mean, C * sizeof(double));
        curr_var_load =
            snrt_dma_start_1d(invstd, l->current_var, C * sizeof(double));
        weight_load = snrt_dma_start_1d(weight, l->weight, C * sizeof(double));
        snrt_dma_wait(grad_ofmap_load);
        snrt_dma_wait(ifmap_load);
        snrt_dma_wait(curr_mean_load);
    } else if (snrt_is_compute_core()) {
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, num_points,
                         num_channels_work_for_core, C * sizeof(double),
                         num_compute_cores * sizeof(double));
    }
    uint32_t end_dma_load = snrt_mcycle();
    snrt_cluster_hw_barrier();

    uint32_t start_invstd_computations = snrt_mcycle();
    uint32_t end_invstd_computations = snrt_mcycle();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_sum_dotp_reduction = snrt_mcycle();
    if (snrt_is_dm_core()) {
        snrt_dma_wait(curr_var_load);
    } else {
        if (num_channels_work_for_core > 0) {
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &grad_ofmap[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, &ifmap[compute_id]);
            for (uint32_t channel = compute_id; channel < C;
                 channel += num_compute_cores) {
                register volatile double sum_reg = 0;
                register volatile double dotp_reg = 0;
                register double curr_mean_reg = curr_mean[channel];
                const register double ZERO = 0;
                snrt_ssr_enable();
                asm volatile(
                    "frep.o %[n_frep], 5, 0, 0 \n"
                    "fadd.d ft3, ft0, %[zero] \n"
                    "fadd.d %[sum], ft3, %[sum] \n"
                    "fsub.d ft4, ft1, %[curr_mean]\n"
                    "fmul.d ft4, ft4, ft3\n"
                    "fadd.d %[dotp], ft4, %[dotp]\n"
                    : [sum] "+fr"(sum_reg), [dotp] "+fr"(dotp_reg)
                    : [curr_mean] "fr"(curr_mean_reg), [zero] "fr"(ZERO),
                      [n_frep] "r"(num_points - 1)
                    : "ft0", "ft1", "ft2", "ft3", "ft4");
                snrt_fpu_fence();
                snrt_ssr_disable();
                sum[channel] = sum_reg;
                dotp[channel] = dotp_reg;
            }
            __builtin_ssr_barrier(SNRT_SSR_DM1);
        }
    }
    uint32_t end_compute_sum_dotp_reduction = snrt_mcycle();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_sum_dotp_reduction_2 = snrt_mcycle();
    uint32_t end_compute_sum_dotp_reduction_2 = snrt_mcycle();

    uint32_t start_compute_k_grad_mean = snrt_mcycle();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_bias, sum, C * sizeof(double));
        snrt_dma_wait(weight_load);
    } else if (snrt_is_compute_core()) {
        if (num_channels_work_for_core > 0) {
            register double num_points_reg = num_points;
            const register double ZERO = 0;
            snrt_ssr_loop_1d(SNRT_SSR_DM_ALL, num_channels_work_for_core,
                             num_compute_cores * sizeof(double));

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &invstd[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, &invstd[compute_id]);
            register double eps = l->eps;
            const register double ONE = 1;
            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 3, 0, 0 \n"
                "fadd.d ft3, ft0, %[eps]\n"
                "fsqrt.d ft3, ft3\n"
                "fdiv.d ft1, %[ONE], ft3\n"
                :
                : [eps] "fr"(eps), [ONE] "fr"(ONE),
                  [n_frep] "r"(num_channels_work_for_core - 1)
                : "ft0", "ft1", "ft2", "ft3");

            snrt_fpu_fence();
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();

            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, &invstd[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, &grad_weight[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_1D, &dotp[compute_id]);
            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 1, 0, 0 \n"
                "fmul.d ft1, ft0, ft2 \n"
                :
                : [n_frep] "r"(num_channels_work_for_core - 1)
                : "ft0", "ft1", "ft2");
            snrt_fpu_fence();
            __builtin_ssr_barrier(SNRT_SSR_DM1);
            snrt_ssr_disable();
        }
    }
    uint32_t end_compute_k_grad_mean = snrt_mcycle();
    snrt_cluster_hw_barrier();

    uint32_t start_compute_grad_ifmap = snrt_mcycle();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_weight, grad_weight, C * sizeof(double));
    } else {
        if (num_channels_work_for_core > 0) {
            snrt_ssr_loop_2d(SNRT_SSR_DM_ALL, num_points,
                             num_channels_work_for_core, C * sizeof(double),
                             num_compute_cores * sizeof(double));
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, &ifmap[compute_id]);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, &grad_ifmap[compute_id]);
            snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, &grad_ofmap[compute_id]);
            for (uint32_t channel = compute_id; channel < C;
                 channel += num_compute_cores) {
                register double curr_mean_reg = curr_mean[channel];
                register double k_reg = dotp[channel] * invstd[channel] * invstd[channel] / num_points;
                register double weight_times_invstd_reg =
                    weight[channel] * invstd[channel];
                register double grad_mean_times_weight_times_invstd_reg = sum[channel] / num_points * weight_times_invstd_reg;
                snrt_ssr_enable();
                asm volatile(
                    "frep.o %[n_frep], 3, 0, 0 \n"
                    "fsub.d ft3, ft0, %[curr_mean] \n"
                    "fnmsub.d ft4, ft3, %[k], ft2\n"
                    "fmsub.d ft1, ft4, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                    :
                    : [curr_mean] "fr"(curr_mean_reg), [k] "fr"(k_reg),
                      [grad_mean_times_weight_times_invstd] "fr"(grad_mean_times_weight_times_invstd_reg),
                      [weight_times_invstd] "fr"(weight_times_invstd_reg),
                      [n_frep] "r"(num_points - 1)
                    : "ft0", "ft1", "ft2", "ft3", "ft4");
                snrt_fpu_fence();
                snrt_ssr_disable();
            }
            __builtin_ssr_barrier(SNRT_SSR_DM1);
        }
    }
    uint32_t end_compute_grad_ifmap = snrt_mcycle();
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l->grad_ifmap, grad_ifmap,
                          C * num_points * sizeof(double));
    }
}
