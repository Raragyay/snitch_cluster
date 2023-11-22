#pragma once

#include <stdbool.h>
#include "batchnorm_data_structures.h"
#include "snrt.h"

#define PERF_DEBUG 0
#define min(a, b) ((a) < (b) ? (a) : (b))
#define ceildiv(a, b) ((((a)-1) / (b)) + 1)

static inline uint32_t get_core_num_work_items(uint32_t num_work_items,
                                               uint32_t num_compute_cores,
                                               uint32_t compute_id) {
    return num_work_items / num_compute_cores +
           (compute_id < (num_work_items % num_compute_cores));
}

static inline uint32_t get_offset_for_core_work_blocked(
    uint32_t num_work_items, uint32_t num_compute_cores, uint32_t compute_id) {
    return num_work_items / num_compute_cores * compute_id +
           min(compute_id, (num_work_items % num_compute_cores));
}

static inline void reset_and_start_perf_single_core(
    uint32_t compute_id, enum snrt_perf_cnt counter_idx,
    enum snrt_perf_cnt_type counter_type) {
#if PERF_DEBUG
    if (compute_id == 0) {
        snrt_reset_perf_counter(counter_idx);

        // Start performance counters
        snrt_start_perf_counter(counter_idx, counter_type, 0);
        DUMP(111000 + counter_idx);
    }
    snrt_cluster_hw_barrier();
#endif
}

static inline void end_perf_and_dump_single_core(
    uint32_t compute_id, enum snrt_perf_cnt counter_idx) {
#if PERF_DEBUG
    uint32_t res;
    if (compute_id == 0) {
        snrt_stop_perf_counter(counter_idx);

        res = snrt_get_perf_counter(counter_idx);
        DUMP(222000 + counter_idx);
        DUMP(res);
    }
    snrt_cluster_hw_barrier();
#endif
}

static inline snrt_dma_txid_t initiate_dma_1d_or_2d(uint64_t dst, uint64_t src,
                                                    size_t size,
                                                    size_t dst_stride,
                                                    size_t src_stride,
                                                    size_t repeat, bool is_1d) {
    if (is_1d) {
        return snrt_dma_start_1d((void*)dst, (void*)src, size * repeat);
    } else {
        return snrt_dma_start_2d((void*)dst, (void*)src, size, dst_stride,
                                 src_stride, repeat);
    }
}

static inline void batchnorm_backward_tile_fp64(
    const double* grad_ofmap_scratch,
    double*
        grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
    const double* ifmap_scratch,
    const double* running_mean_times_invstd_scratch,
    const double* weight_times_invstd_scratch, const double* invstd_scratch,
    double* grad_bias_scratch, double* grad_weight_scratch, uint32_t TILE_CI,
    uint32_t compute_id, uint32_t num_compute_cores,
    uint32_t num_points_work_for_core_in_tile, bool is_first_iteration,
    bool is_last_iteration) {
    // access pattern: iterate over the different channels, then over
    // the different points split over points
    // outside loop: channels
    // inside loop: points
    if (num_points_work_for_core_in_tile == 0) {
        return;
    }
    if (is_first_iteration || is_last_iteration) {
        snrt_ssr_loop_2d(
            SNRT_SSR_DM_ALL,
            num_points_work_for_core_in_tile,  // dimension of inner loop
            TILE_CI,                           // dimension of outer loop
            TILE_CI * sizeof(double),  // stride per inner loop iteration
            sizeof(double));           // stride per outer loop iteration
    }

    // thought: how could I minimize the # of reads to grad_ofmap?
    // dy is used for: grad_bias (addition)
    //                 grad_weight (dy * (x[i,C]-running_mean[C]) * invstd[C])
    //                             (can it become a fused somehow? not really..
    //                             can precompute invstd * running_mean though)
    //                             then you get an fmsub(x[i,C], invstd[C],
    //                             invstd[C]*running_mean[C])
    //                 grad_ifmap (dy * invstd[C] * weight[C])
    // from this I think that the best result is to tile dy and x.
    // need to also tile the write out to grad_ifmap. This fills up all 3 ssrs.

    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_enable();
    for (uint32_t channel = 0; channel < TILE_CI; ++channel) {
        register double running_mean_times_invstd =
            *running_mean_times_invstd_scratch;
        register double weight_times_invstd = *weight_times_invstd_scratch;
        register double invstd = *invstd_scratch;
        register double grad_weight_0 = 0;
        register double grad_weight_1 = 0;
        register double grad_weight_2 = 0;
        register double grad_bias_0 = 0;
        register double grad_bias_1 = 0;
        register double grad_bias_2 = 0;
        register double ZERO = 0;  // can consider fcvt instead
        // Can only manual unroll 3 times since the max for frep is 16
        // thought: after issuing this, increment all the channels manually.
        // That way we dual issue a bit better
        if (num_points_work_for_core_in_tile >= 3) {
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
                  [n_frep] "r"(num_points_work_for_core_in_tile / 3 -
                               1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
                  "ft8");
        }
        // invstd_scratch += 1;
        // weight_times_invstd_scratch += 1;
        // running_mean_times_invstd_scratch += 1;
        // if (channel != 0) {
        //     grad_bias_scratch += 1;
        //     grad_weight_scratch += 1;
        // }
        // Inline the asm to force pseudo-dual issue
        asm volatile(
            "addi %[invstd_scratch], %[invstd_scratch], 8\n"
            "addi %[weight_times_invstd_scratch], "
            "%[weight_times_invstd_scratch], 8\n"
            "addi %[running_mean_times_invstd_scratch], "
            "%[running_mean_times_invstd_scratch], 8\n"
            "beqz %[channel], 1f\n"
            "addi %[grad_bias_scratch], %[grad_bias_scratch], 8\n"
            "addi %[grad_weight_scratch], %[grad_weight_scratch], 8\n"
            "1:\n"
            : [invstd_scratch] "+r"(invstd_scratch),
              [weight_times_invstd_scratch] "+r"(weight_times_invstd_scratch),
              [running_mean_times_invstd_scratch] "+r"(
                  running_mean_times_invstd_scratch),
              [grad_bias_scratch] "+r"(grad_bias_scratch),
              [grad_weight_scratch] "+r"(grad_weight_scratch)
            : [channel] "r"(channel)
            : "ft0", "ft1", "ft2");

        switch (num_points_work_for_core_in_tile % 3) {
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
        // in plain C:
        // if (is_first_iteration) {
        //     grad_bias_scratch[channel] =
        //         grad_bias_0 + grad_bias_1 + grad_bias_2;
        //     grad_weight_scratch[channel] =
        //         grad_weight_0 + grad_weight_1 + grad_weight_2;

        // } else {
        //     grad_bias_scratch[channel] +=
        //         grad_bias_0 + grad_bias_1 + grad_bias_2;
        //     grad_weight_scratch[channel] +=
        //         grad_weight_0 + grad_weight_1 + grad_weight_2;
        // }
        register double temp_grad_bias, temp_grad_weight;
        asm volatile(
            "bnez %[is_first_iteration], 3f\n"
            "fld %[temp_grad_bias], 0(%[grad_bias_scratch])\n"
            "fld %[temp_grad_weight], 0(%[grad_weight_scratch])\n"
            "fadd.d %[grad_bias_0], %[temp_grad_bias], %[grad_bias_0]\n"
            "fadd.d %[grad_weight_0], %[temp_grad_weight], %[grad_weight_0]\n"
            "3:\n"
            "fadd.d %[grad_bias_1], %[grad_bias_1], %[grad_bias_2]\n"
            "fadd.d %[grad_weight_1], %[grad_weight_1], %[grad_weight_2]\n"
            "fadd.d %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
            "fadd.d %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
            "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
            "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
            : [grad_bias_scratch] "+r"(grad_bias_scratch),
              [grad_weight_scratch] "+r"(grad_weight_scratch),
              [temp_grad_bias] "+fr"(temp_grad_bias),
              [temp_grad_weight] "+fr"(temp_grad_weight)
            : [grad_weight_0] "fr"(grad_weight_0),
              [grad_weight_1] "fr"(grad_weight_1),
              [grad_weight_2] "fr"(grad_weight_2),
              [grad_bias_0] "fr"(grad_bias_0), [grad_bias_1] "fr"(grad_bias_1),
              [grad_bias_2] "fr"(grad_bias_2),
              [is_first_iteration] "r"(is_first_iteration)
            : "ft0", "ft1", "ft2");
        snrt_fpu_fence();
    }
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void batchnorm_backward_main_loop(
    bool loop_points, bool loop_channels, uint32_t C, uint32_t TILE_CI,
    uint32_t num_points, uint32_t tile_size_in_points, uint32_t compute_id,
    uint32_t num_compute_cores, batchnorm_backward_layer_t* l,
    double* grad_ofmap_scratch, double* ifmap_scratch,
    double* grad_ifmap_scratch, double* grad_weight_scratch,
    double* grad_bias_scratch, double* invstd_scratch,
    double* running_mean_times_invstd_scratch,
    double* weight_times_invstd_scratch, bool buf_flag) {
    uint32_t start_main_loop = snrt_mcycle();
    uint32_t num_points_work_in_tile, num_points_work_for_core_for_tile,
        blocked_offset;
    uint32_t prev_point_start, num_points_work_in_prev_tile, prev_channel_start;
    bool is_last_iteration = false;
    if (loop_points && loop_channels) {
        DUMP(0);
        for (uint32_t point_start = 0; point_start < num_points;
             point_start += tile_size_in_points) {
            if (point_start + tile_size_in_points >= num_points) {
                // on last iteration
                is_last_iteration = true;
                num_points_work_in_tile = num_points - point_start;
                num_points_work_for_core_for_tile = get_core_num_work_items(
                    num_points_work_in_tile, num_compute_cores,
                    compute_id);  // for the last tile, might be different
                blocked_offset = get_offset_for_core_work_blocked(
                    num_points_work_in_tile, num_compute_cores, compute_id);
            } else if (point_start == 0) {
                // on first iteration
                num_points_work_in_tile = tile_size_in_points;

                num_points_work_for_core_for_tile = get_core_num_work_items(
                    num_points_work_in_tile, num_compute_cores, compute_id);
                blocked_offset = get_offset_for_core_work_blocked(
                    num_points_work_in_tile, num_compute_cores, compute_id);
            }
            for (uint32_t channel_start = 0; channel_start < C;
                 channel_start += TILE_CI) {
                // iterate over CI here?

                if (snrt_is_dm_core()) {
                    // technically we could optimize by loading in both sides
                    // ahead of time. For now not going to do that
                    if (point_start != 0 || channel_start != 0) {
                        // first buffer was already loaded in before
                        snrt_dma_start_2d(
                            &grad_ofmap_scratch[tile_size_in_points * TILE_CI *
                                                buf_flag],
                            &l->grad_ofmap[point_start * C + channel_start],
                            TILE_CI * sizeof(double), TILE_CI * sizeof(double),
                            C * sizeof(double), tile_size_in_points);
                        snrt_dma_start_2d(
                            &ifmap_scratch[tile_size_in_points * TILE_CI *
                                           buf_flag],
                            &l->ifmap[point_start * C + channel_start],
                            TILE_CI * sizeof(double), TILE_CI * sizeof(double),
                            C * sizeof(double), tile_size_in_points);
                    }
                    snrt_dma_wait_all();
                    // wait for compute cores to finish computing current tile,
                    // signal to them grad_ofmap[buf_flag], ifmap[buf_flag],
                    // grad_ifmap[!buf_flag] done?
                    snrt_cluster_hw_barrier();
                    // write out the buffer that was just computed
                    // IDEA: should be able to skip this on last iteration as
                    // well
                    if (point_start != 0 || channel_start != 0) {
                        snrt_dma_start_2d(
                            &l->grad_ifmap[prev_point_start * C +
                                           prev_channel_start],
                            &grad_ifmap_scratch[tile_size_in_points * TILE_CI *
                                                (!buf_flag)],  // take !buf_flag
                                                               // since dma core
                                                               // is one
                                                               // iteration
                                                               // ahead of
                                                               // compute core
                            TILE_CI * sizeof(double), C * sizeof(double),
                            TILE_CI * sizeof(double),
                            num_points_work_in_prev_tile);
                        buf_flag = !buf_flag;
                    }
                    prev_point_start = point_start;
                    prev_channel_start = channel_start;
                    num_points_work_in_prev_tile = num_points_work_in_tile;

                } else {
                    // since we didn't flip buf_flag with the dma load, buf_flag
                    // starts at 0. Signifies the current tile being worked on.

                    // dma core will signal to us when we can start next
                    // computation
                    snrt_cluster_hw_barrier();
                    batchnorm_backward_tile_fp64(
                        &grad_ofmap_scratch[(buf_flag * tile_size_in_points +
                                             blocked_offset) *
                                            TILE_CI],
                        &grad_ifmap_scratch[(buf_flag * tile_size_in_points +
                                             blocked_offset) *
                                            TILE_CI],
                        &ifmap_scratch[(buf_flag * tile_size_in_points +
                                        blocked_offset) *
                                       TILE_CI],
                        &running_mean_times_invstd_scratch[channel_start],
                        &weight_times_invstd_scratch[channel_start],
                        &invstd_scratch[channel_start],
                        &grad_bias_scratch[compute_id * C + channel_start],
                        &grad_weight_scratch[compute_id * C + channel_start],
                        TILE_CI, compute_id, num_compute_cores,
                        num_points_work_for_core_for_tile,
                        point_start == 0 && channel_start == 0,
                        is_last_iteration && channel_start == 0);
                    buf_flag = !buf_flag;
                }
            }
        }
    } else if (loop_points && !loop_channels) {
        DUMP(1);
        for (uint32_t point_start = 0; point_start < num_points;
             point_start += tile_size_in_points) {
            if (point_start + tile_size_in_points >= num_points) {
                // on last iteration
                is_last_iteration = true;
                num_points_work_in_tile = num_points - point_start;
                num_points_work_for_core_for_tile = get_core_num_work_items(
                    num_points_work_in_tile, num_compute_cores,
                    compute_id);  // for the last tile, might be different
                blocked_offset = get_offset_for_core_work_blocked(
                    num_points_work_in_tile, num_compute_cores, compute_id);
            } else if (point_start == 0) {
                // on first iteration
                num_points_work_in_tile = tile_size_in_points;

                num_points_work_for_core_for_tile = get_core_num_work_items(
                    num_points_work_in_tile, num_compute_cores, compute_id);
                blocked_offset = get_offset_for_core_work_blocked(
                    num_points_work_in_tile, num_compute_cores, compute_id);
            }
            // iterate over CI here?

            if (snrt_is_dm_core()) {
                // technically we could optimize by loading in both sides ahead
                // of time. For now not going to do that
                if (point_start != 0) {
                    // first buffer was already loaded in before
                    snrt_dma_start_1d(
                        &grad_ofmap_scratch[num_points_work_in_tile * C *
                                            buf_flag],
                        &l->grad_ofmap[point_start * C],
                        num_points_work_in_tile * C * sizeof(double));
                    snrt_dma_start_1d(
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
                // IDEA: should be able to skip this on last iteration as well
                if (point_start != 0) {
                    snrt_dma_start_1d(
                        &l->grad_ifmap[prev_point_start * C],
                        &grad_ifmap_scratch[tile_size_in_points * C *
                                            (!buf_flag)],  // take !buf_flag dma
                                                           // core is one
                                                           // iteration ahead of
                                                           // compute core
                        num_points_work_in_prev_tile * C * sizeof(double));
                    buf_flag = !buf_flag;
                }
                prev_point_start = point_start;
                num_points_work_in_prev_tile = num_points_work_in_tile;

            } else {
                // since we didn't flip buf_flag with the dma load, buf_flag
                // starts at 0. Signifies the current tile being worked on.

                // dma core will signal to us when we can start next computation
                snrt_cluster_hw_barrier();
                batchnorm_backward_tile_fp64(
                    &grad_ofmap_scratch
                        [(buf_flag * tile_size_in_points + blocked_offset) * C],
                    &grad_ifmap_scratch
                        [(buf_flag * tile_size_in_points + blocked_offset) * C],
                    &ifmap_scratch
                        [(buf_flag * tile_size_in_points + blocked_offset) * C],
                    running_mean_times_invstd_scratch,
                    weight_times_invstd_scratch, invstd_scratch,
                    &grad_bias_scratch[compute_id * C],
                    &grad_weight_scratch[compute_id * C], C, compute_id,
                    num_compute_cores, num_points_work_for_core_for_tile,
                    point_start == 0, is_last_iteration);
                buf_flag = !buf_flag;
            }
        }
    } else if (!loop_points && loop_channels) {
        DUMP(2);
        num_points_work_for_core_for_tile =
            get_core_num_work_items(num_points, num_compute_cores, compute_id);
        blocked_offset = get_offset_for_core_work_blocked(
            num_points, num_compute_cores, compute_id);
        for (uint32_t channel_start = 0; channel_start < C;
             channel_start += TILE_CI) {
            if (snrt_is_dm_core()) {
                // technically we could optimize by loading in both sides
                // ahead of time. For now not going to do that
                if (channel_start != 0) {
                    // first buffer was already loaded in before
                    snrt_dma_start_2d(
                        &grad_ofmap_scratch[num_points * TILE_CI * buf_flag],
                        &l->grad_ofmap[channel_start], TILE_CI * sizeof(double),
                        TILE_CI * sizeof(double), C * sizeof(double),
                        num_points);
                    snrt_dma_start_2d(
                        &ifmap_scratch[num_points * TILE_CI * buf_flag],
                        &l->ifmap[channel_start], TILE_CI * sizeof(double),
                        TILE_CI * sizeof(double), C * sizeof(double),
                        num_points);
                }
                snrt_dma_wait_all();
                // wait for compute cores to finish computing current tile,
                // signal to them grad_ofmap[buf_flag], ifmap[buf_flag],
                // grad_ifmap[!buf_flag] done?
                snrt_cluster_hw_barrier();
                // write out the buffer that was just computed
                // IDEA: should be able to skip this on last iteration as
                // well
                if (channel_start != 0) {
                    snrt_dma_start_2d(
                        &l->grad_ifmap[prev_channel_start],
                        &grad_ifmap_scratch[tile_size_in_points * TILE_CI *
                                            (!buf_flag)],  // take !buf_flag
                                                           // since dma core
                                                           // is one
                                                           // iteration
                                                           // ahead of
                                                           // compute core
                        TILE_CI * sizeof(double), C * sizeof(double),
                        TILE_CI * sizeof(double),
                        num_points);  // don't need to save num_points
                    buf_flag = !buf_flag;
                }
                prev_channel_start = channel_start;

            } else {
                // since we didn't flip buf_flag with the dma load, buf_flag
                // starts at 0. Signifies the current tile being worked on.

                // dma core will signal to us when we can start next
                // computation
                snrt_cluster_hw_barrier();
                batchnorm_backward_tile_fp64(
                    &grad_ofmap_scratch
                        [(buf_flag * num_points + blocked_offset) * TILE_CI],
                    &grad_ifmap_scratch
                        [(buf_flag * num_points + blocked_offset) * TILE_CI],
                    &ifmap_scratch[(buf_flag * num_points + blocked_offset) *
                                   TILE_CI],
                    &running_mean_times_invstd_scratch[channel_start],
                    &weight_times_invstd_scratch[channel_start],
                    &invstd_scratch[channel_start],
                    &grad_bias_scratch[compute_id * C + channel_start],
                    &grad_weight_scratch[compute_id * C + channel_start],
                    TILE_CI, compute_id, num_compute_cores,
                    num_points_work_for_core_for_tile, channel_start == 0,
                    channel_start == 0);
                buf_flag = !buf_flag;
            }
        }
    } else if (!loop_points && !loop_channels) {
        DUMP(3);
        num_points_work_for_core_for_tile =
            get_core_num_work_items(num_points, num_compute_cores, compute_id);
        blocked_offset = get_offset_for_core_work_blocked(
            num_points, num_compute_cores, compute_id);
        if (snrt_is_dm_core()) {
        } else {
            batchnorm_backward_tile_fp64(
                &grad_ofmap_scratch[blocked_offset * C],
                &grad_ifmap_scratch

                    [blocked_offset * C],
                &ifmap_scratch[blocked_offset * C],
                running_mean_times_invstd_scratch, weight_times_invstd_scratch,
                invstd_scratch, &grad_bias_scratch[compute_id * C],
                &grad_weight_scratch[compute_id * C], C, compute_id,
                num_compute_cores, num_points_work_for_core_for_tile, true,
                true);
        }
    }

    uint32_t end_main_loop = snrt_mcycle();
    snrt_cluster_hw_barrier();
    if (snrt_is_dm_core()) {
        if (loop_points && loop_channels) {
            snrt_dma_start_2d(
                &l->grad_ifmap[prev_point_start * C + prev_channel_start],
                &grad_ifmap_scratch[tile_size_in_points * TILE_CI *
                                    (!buf_flag)],  // take !buf_flag since
                                                   // dma core is one
                                                   // iteration ahead of
                                                   // compute core
                TILE_CI * sizeof(double), C * sizeof(double),
                TILE_CI * sizeof(double), num_points_work_in_prev_tile);
        } else if (loop_points && !loop_channels) {
            snrt_dma_start_1d(
                &l->grad_ifmap[prev_point_start * C],
                &grad_ifmap_scratch[tile_size_in_points * C *
                                    (!buf_flag)],  // take !buf_flag dma
                                                   // core is one iteration
                                                   // ahead of compute core
                num_points_work_in_prev_tile * C * sizeof(double));
        } else if (!loop_points && loop_channels) {
            snrt_dma_start_2d(
                &l->grad_ifmap[prev_channel_start],
                &grad_ifmap_scratch[num_points * TILE_CI *
                                    (!buf_flag)],  // take !buf_flag
                                                   // since dma core
                                                   // is one
                                                   // iteration
                                                   // ahead of
                                                   // compute core
                TILE_CI * sizeof(double), C * sizeof(double),
                TILE_CI * sizeof(double),
                num_points);  // don't need to save num_points
        } else if (!loop_points && !loop_channels) {
            snrt_dma_start_1d(l->grad_ifmap, grad_ifmap_scratch,
                              num_points * C * sizeof(double));
        }
    }
}