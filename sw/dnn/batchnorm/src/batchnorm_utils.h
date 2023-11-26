#pragma once

#include <stdbool.h>
#include "batchnorm_data_structures.h"
#include "snrt.h"

#define PERF_DEBUG 0
#define PERF_WHOLE_BLOCK 1

#if PERF_WHOLE_BLOCK
#define SNRT_SECTIONED_MCYCLE() 0xdeadbeef
#else
#define SNRT_SECTIONED_MCYCLE() (snrt_mcycle())
#endif
#define min(a, b) ((a) < (b) ? (a) : (b))
#define ceildiv(a, b) ((((a)-1) / (b)) + 1)

static inline uint32_t __attribute__((__const__))
get_core_num_work_items(uint32_t num_work_items, uint32_t num_compute_cores, uint32_t compute_id) {
    return num_work_items / num_compute_cores + (compute_id < (num_work_items % num_compute_cores));
}

static inline uint32_t __attribute__((__const__))
get_offset_for_core_work_blocked(uint32_t num_work_items, uint32_t num_compute_cores, uint32_t compute_id) {
    return num_work_items / num_compute_cores * compute_id + min(compute_id, (num_work_items % num_compute_cores));
}

// use ALIGN_UP for powers of 2 - it'll be much faster
static inline uint32_t __attribute__((const)) align_up_non_power_of_2(uint32_t n, uint32_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

static inline void reset_and_start_perf_single_core(uint32_t compute_id, enum snrt_perf_cnt counter_idx,
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

static inline void end_perf_and_dump_single_core(uint32_t compute_id, enum snrt_perf_cnt counter_idx) {
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

static inline snrt_dma_txid_t initiate_dma_1d_or_2d(uint64_t dst, uint64_t src, size_t size, size_t dst_stride,
                                                    size_t src_stride, size_t repeat, bool is_1d) {
    if (is_1d) {
        return snrt_dma_start_1d((void*)dst, (void*)src, size * repeat);
    } else {
        return snrt_dma_start_2d((void*)dst, (void*)src, size, dst_stride, src_stride, repeat);
    }
}

static inline void __attribute__((always_inline))
batchnorm_backward_tile_fp64(const double* grad_ofmap_scratch,
                             double* grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
                             const double* ifmap_scratch, const double* running_mean_scratch,
                             const double* weight_scratch, const double* invstd_scratch, double* grad_bias_scratch,
                             double* grad_weight_scratch, uint32_t C,
                             uint32_t num_points_work_for_core_in_tile,  // requires: > 0
                             uint32_t work_mod,                          // precompute to avoid icache branch misses
                             uint32_t num_channels_to_process,           //  requires: > 0
                             uint32_t channel_stride, bool is_first_iteration, bool force_configure) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    if (is_first_iteration || force_configure) {
        snrt_ssr_loop_2d(SNRT_SSR_DM_ALL,
                         num_points_work_for_core_in_tile,  // dimension of inner loop
                         num_channels_to_process,           // dimension of outer loop
                         C * sizeof(double),                // stride per inner loop iteration: 1 point
                         channel_stride * sizeof(double));  // stride per outer loop iteration
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
    bool frep = num_points_work_for_core_in_tile >= 3;
    register volatile uint32_t i = 0;  // updated during frep for pseudo-dual issue
    register double ZERO asm("ft9");   // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n" : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    register double grad_weight_0 = ZERO;
    register double grad_weight_1 = ZERO;
    register double grad_weight_2 = ZERO;
    register double grad_bias_0 = ZERO;
    register double grad_bias_1 = ZERO;
    register double grad_bias_2 = ZERO;
    register double invstd = *invstd_scratch;
    register double weight_times_invstd = *weight_scratch;
    register double running_mean_times_invstd = *running_mean_scratch;
    // do 1 loop
    do {  // while (i < num_channels_to_process)
        // Can only manual unroll 3 times since the max for frep is 16
        asm volatile(
            "fmul.d "
            "%[running_mean_times_invstd],%[running_mean_times_invstd],%["
            "invstd]\n"
            "fmul.d "
            "%[weight_times_invstd],%[weight_times_invstd],%[invstd]\n"
            : [running_mean_times_invstd] "+fr"(running_mean_times_invstd), [weight_times_invstd] "+fr"(
                                                                                weight_times_invstd)
            : [invstd] "fr"(invstd)
            : "ft0", "ft1", "ft2");
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 15, 0, 0 \n"
                "fadd.d ft3, ft0, %[zero] \n"
                "fadd.d ft5, ft0, %[zero] \n"
                "fadd.d ft7, ft0, %[zero] \n"
                "fmsub.d ft4, ft2, %[invstd], "
                "%[running_mean_times_invstd]\n"
                "fmsub.d ft6, ft2, %[invstd], "
                "%[running_mean_times_invstd]\n"
                "fmsub.d ft8, ft2, %[invstd], "
                "%[running_mean_times_invstd]\n"
                "fadd.d %[grad_bias_0], ft3, %[grad_bias_0]\n"
                "fadd.d %[grad_bias_1], ft5, %[grad_bias_1]\n"
                "fadd.d %[grad_bias_2], ft7, %[grad_bias_2]\n"
                "fmadd.d %[grad_weight_0], ft4, ft3, %[grad_weight_0]\n"
                "fmadd.d %[grad_weight_1], ft6, ft5, %[grad_weight_1]\n"
                "fmadd.d %[grad_weight_2], ft8, ft7, %[grad_weight_2]\n"
                "fmul.d ft1, ft3, %[weight_times_invstd]\n"
                "fmul.d ft1, ft5, %[weight_times_invstd]\n"
                "fmul.d ft1, ft7, %[weight_times_invstd]\n"
                : [grad_weight_0] "+fr"(grad_weight_0), [grad_weight_1] "+fr"(grad_weight_1),
                  [grad_weight_2] "+fr"(grad_weight_2), [grad_bias_0] "+fr"(grad_bias_0),
                  [grad_bias_1] "+fr"(grad_bias_1), [grad_bias_2] "+fr"(grad_bias_2)
                : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
                  [weight_times_invstd] "fr"(weight_times_invstd), [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                  [n_frep] "r"(num_points_work_for_core_in_tile / 3 - 1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[grad_bias_scratch], "
            "%[grad_bias_scratch],%[channel_stride_in_bytes]\n"
            "add %[grad_weight_scratch], %[grad_weight_scratch], "
            "%[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "beq %[num_channels_to_process], %[i], 2f\n"  // shortcut when
                                                          // only 1 channel
            "add %[invstd_scratch], %[invstd_scratch], "
            "%[channel_stride_in_bytes]\n"
            "add %[weight_scratch], "
            "%[weight_scratch],%[channel_stride_in_bytes]\n"
            "add "
            "%[running_mean_scratch],%[running_mean_scratch],%[channel_"
            "stride_"
            "in_bytes]\n "
            "2:\n"
            : [invstd_scratch] "+r"(invstd_scratch), [weight_scratch] "+r"(weight_scratch),
              [running_mean_scratch] "+r"(running_mean_scratch), [grad_bias_scratch] "+r"(grad_bias_scratch),
              [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride), [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");

        register uint32_t mod_temp;
        asm volatile(
            "beqz %[work_mod], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod], 1\n"  // is mod equal to 1?
            "bnez %[mod_temp], 1f\n"              // mod is 1, jump. Otherwise handle 2
                                                  // case
            "2:\n"
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
            "j 0f\n"
            "1:\n"
            "fadd.d ft3, ft0, %[zero] \n"
            "fmsub.d ft4, ft2, %[invstd], "
            "%[running_mean_times_invstd]\n"
            "fadd.d %[grad_bias_0], ft3, %[grad_bias_0]\n"
            "fmul.d ft1, ft3, %[weight_times_invstd]\n"
            "fmadd.d %[grad_weight_0], ft4, ft3, %[grad_weight_0]\n"
            "0:\n"
            : [grad_weight_0] "+fr"(grad_weight_0), [grad_weight_1] "+fr"(grad_weight_1),
              [grad_bias_0] "+fr"(grad_bias_0), [grad_bias_1] "+fr"(grad_bias_1), [mod_temp] "=r"(mod_temp)
            : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
              [weight_times_invstd] "fr"(weight_times_invstd), [invstd] "fr"(invstd), [zero] "fr"(ZERO),
              [work_mod] "r"(work_mod)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

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
            // interleave 0 resetting between stalls for addition
            "fsgnj.d %[grad_bias_2],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[grad_weight_2],%[ZERO],%[ZERO]\n"
            // don't need to synchronize here because the integer core can't
            // issue these instructions until the previous increments have
            // happened
            "fld %[invstd],0(%[invstd_scratch])\n"
            "fadd.d %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
            "fadd.d %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
            "fsgnj.d %[grad_bias_1],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[grad_weight_1],%[ZERO],%[ZERO]\n"
            "fld %[weight_times_invstd],0(%[weight_scratch])\n"
            "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
            "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
            "fsgnj.d %[grad_bias_0],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[grad_weight_0],%[ZERO],%[ZERO]\n"
            "fld %[running_mean_times_invstd],"
            "0(%[running_mean_scratch])\n"
            : [temp_grad_bias] "+fr"(temp_grad_bias), [temp_grad_weight] "+fr"(temp_grad_weight),
              [grad_weight_0] "+fr"(grad_weight_0), [grad_weight_1] "+fr"(grad_weight_1),
              [grad_bias_0] "+fr"(grad_bias_0), [grad_bias_1] "+fr"(grad_bias_1),
              [running_mean_times_invstd] "=fr"(running_mean_times_invstd),
              [weight_times_invstd] "=fr"(weight_times_invstd), [invstd] "=fr"(invstd)
            : [is_first_iteration] "r"(is_first_iteration), [ZERO] "fr"(ZERO), [invstd_scratch] "r"(invstd_scratch),
              [weight_scratch] "r"(weight_scratch), [running_mean_scratch] "r"(running_mean_scratch),
              [grad_bias_scratch] "r"(grad_bias_scratch), [grad_weight_scratch] "r"(grad_weight_scratch),
              [grad_weight_2] "fr"(grad_weight_2), [grad_bias_2] "fr"(grad_bias_2)
            : "ft0", "ft1", "ft2");
    } while (i < num_channels_to_process);
    // don't need to fpu_fence since last 3 instructions are inconsequential
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void __attribute__((always_inline))
batchnorm_backward_tile_fp64_looped(const double* grad_ofmap_scratch,
                                    double* grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
                                    const double* ifmap_scratch, const double* running_mean_scratch,
                                    const double* weight_scratch, const double* invstd_scratch,
                                    double* grad_bias_scratch, double* grad_weight_scratch, uint32_t C,
                                    uint32_t work_in_tile,  // requires: > 0
                                    uint32_t work_mod_3,    // precompute to avoid icache branch misses
                                    uint32_t tile_size_in_points,
                                    uint32_t num_channels_to_process,  //  requires: > 0
                                    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = 0;
    register uint32_t next_work_mod_3;
    register bool frep = work_in_tile >= 3;
    register double ZERO asm("ft9");  // can consider fcvt instead
    bool is_last_iteration = false;
    asm volatile("fcvt.d.w %[ZERO], zero\n" : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");

    bool buf_flag = 0;
    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_points * C * sizeof(double);
    const uint32_t channel_array_reset_dist = channel_stride * num_channels_to_process * sizeof(double);
    // TODO: fix num_channels_work_for_core == 0.
    do {
        // DUMP(1);
        // DUMP(111);
        // DUMP(work_in_tile);
        // DUMP(num_channels_to_process);
        // DUMP(work_mod_3);
        // inputs have been loaded in, notify previous tile done if not on 1st
        // loop
        snrt_cluster_hw_barrier();
        if (prev_work != work_in_tile) {
            DUMP(33);
            snrt_ssr_loop_2d(SNRT_SSR_DM_ALL,
                             work_in_tile,                      // dimension of inner loop
                             num_channels_to_process,           // dimension of outer loop
                             C * sizeof(double),                // stride per inner loop iteration: 1 point
                             channel_stride * sizeof(double));  // stride per outer loop iteration
        }
        prev_work = work_in_tile;

        // DUMP(ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
        snrt_ssr_enable();
        register volatile uint32_t i = 0;  // updated during frep for pseudo-dual issue
        register double grad_weight_0 = ZERO;
        register double grad_weight_1 = ZERO;
        register double grad_weight_2 = ZERO;
        register double grad_bias_0 = ZERO;
        register double grad_bias_1 = ZERO;
        register double grad_bias_2 = ZERO;
        register double invstd = *invstd_scratch;
        register double weight_times_invstd = *weight_scratch;
        register double running_mean_times_invstd = *running_mean_scratch;
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            // Can only manual unroll 3 times since the max for frep is 16
            asm volatile(
                "fmul.d "
                "%[running_mean_times_invstd],%[running_mean_times_invstd],%["
                "invstd]\n"
                "fmul.d "
                "%[weight_times_invstd],%[weight_times_invstd],%[invstd]\n"
                : [running_mean_times_invstd] "+fr"(running_mean_times_invstd), [weight_times_invstd] "+fr"(
                                                                                    weight_times_invstd)
                : [invstd] "fr"(invstd)
                : "ft0", "ft1", "ft2");
            if (frep) {  // in theory I only need to do once per outer loop, but the code duplication for that is
                         // ungodly
                asm volatile(
                    "frep.o %[n_frep], 15, 0, 0 \n"
                    "fadd.d ft3, ft0, %[zero] \n"
                    "fadd.d ft5, ft0, %[zero] \n"
                    "fadd.d ft7, ft0, %[zero] \n"
                    "fmsub.d ft4, ft2, %[invstd], "
                    "%[running_mean_times_invstd]\n"
                    "fmsub.d ft6, ft2, %[invstd], "
                    "%[running_mean_times_invstd]\n"
                    "fmsub.d ft8, ft2, %[invstd], "
                    "%[running_mean_times_invstd]\n"
                    "fadd.d %[grad_bias_0], ft3, %[grad_bias_0]\n"
                    "fadd.d %[grad_bias_1], ft5, %[grad_bias_1]\n"
                    "fadd.d %[grad_bias_2], ft7, %[grad_bias_2]\n"
                    "fmadd.d %[grad_weight_0], ft4, ft3, %[grad_weight_0]\n"
                    "fmadd.d %[grad_weight_1], ft6, ft5, %[grad_weight_1]\n"
                    "fmadd.d %[grad_weight_2], ft8, ft7, %[grad_weight_2]\n"
                    "fmul.d ft1, ft3, %[weight_times_invstd]\n"
                    "fmul.d ft1, ft5, %[weight_times_invstd]\n"
                    "fmul.d ft1, ft7, %[weight_times_invstd]\n"
                    : [grad_weight_0] "+fr"(grad_weight_0), [grad_weight_1] "+fr"(grad_weight_1),
                      [grad_weight_2] "+fr"(grad_weight_2), [grad_bias_0] "+fr"(grad_bias_0),
                      [grad_bias_1] "+fr"(grad_bias_1), [grad_bias_2] "+fr"(grad_bias_2)
                    : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
                      [weight_times_invstd] "fr"(weight_times_invstd), [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                      [n_frep] "r"(work_in_tile / 3 - 1)  // we repeat n_frep+1 times
                    : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "memory");
            }

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "beqz %[i], 1f\n"
                "add %[grad_bias_scratch], %[grad_bias_scratch], %[channel_stride_in_bytes]\n"
                "add %[grad_weight_scratch], %[grad_weight_scratch], %[channel_stride_in_bytes]\n"
                "1:\n"
                "addi %[i], %[i], 1\n"
                "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
                "add %[weight_scratch], %[weight_scratch],%[channel_stride_in_bytes]\n"
                "add %[running_mean_scratch],%[running_mean_scratch],%[channel_stride_in_bytes]\n "
                : [invstd_scratch] "+r"(invstd_scratch), [weight_scratch] "+r"(weight_scratch),
                  [running_mean_scratch] "+r"(running_mean_scratch), [grad_bias_scratch] "+r"(grad_bias_scratch),
                  [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+&r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride), [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2", "memory");

            // TODO
            // if (is_last_channel) {
            //     invstd_scratch -= channel_stride * num_channels_to_process;
            //     running_mean_scratch -= channel_stride * num_channels_to_process;
            //     weight_scratch -= channel_stride * num_channels_to_process;
            //     buf_flag = !buf_flag;
            //     work_in_tile = (dm_comm + buf_flag)->num_points_work_in_tile;
            //     work_mod_3 = (dm_comm + buf_flag)->work_mod_3;
            //     frep = work_in_tile >= 3;
            //     // flip grad_ofmap_scratch
            //     if (prev_work != work_in_tile) {
            //         // do ssr 2d
            //     }
            //     prev_work = work_in_tile;
            // }

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                "sub %[invstd_scratch], %[invstd_scratch], %[channel_array_reset_dist]\n"
                "sub %[weight_scratch], %[weight_scratch],%[channel_array_reset_dist]\n"
                "sub %[running_mean_scratch],%[running_mean_scratch],%[channel_array_reset_dist]\n "
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_3], %[work_mod_3_offset](%[dm_comm])\n"
                // "lw %[is_last_iteration], %[is_last_iteration_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 3\n"  // cmp frep <3, then negate in next instruction
                "xori %[frep], %[frep], 1\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "add %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "sub %[grad_ifmap_scratch], %[grad_ifmap_scratch], %[buf_flag_offset]\n"
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"
                : [buf_flag] "+r"(buf_flag), [invstd_scratch] "+r"(invstd_scratch),
                  [weight_scratch] "+r"(weight_scratch), [running_mean_scratch] "+r"(running_mean_scratch),
                  [work_in_tile] "=r"(work_in_tile), [next_work_mod_3] "=r"(next_work_mod_3),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep), [is_last_iteration] "=r"(is_last_iteration),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch), [grad_ifmap_scratch] "+r"(grad_ifmap_scratch),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i), [num_channels_to_process] "r"(num_channels_to_process),
                  [channel_array_reset_dist] "r"(channel_array_reset_dist),
                  [work_in_tile_offset] "i"(offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_3_offset] "i"(offsetof(dm_comm_t, work_mod_3)),
                  [is_last_iteration_offset] "i"(offsetof(dm_comm_t, is_last_iteration)), [dm_comm] "r"(dm_comm),
                  [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "memory");

            asm volatile(
                "beqz %[work_mod_3], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_3], 1\n"  // is mod equal to 1?
                "bnez %[mod_temp], 1f\n"                // mod is 1, jump. Otherwise handle 2
                                                        // case
                "2:\n"
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
                "j 0f\n"
                "1:\n"
                "fadd.d ft3, ft0, %[zero] \n"
                "fmsub.d ft4, ft2, %[invstd], "
                "%[running_mean_times_invstd]\n"
                "fadd.d %[grad_bias_0], ft3, %[grad_bias_0]\n"
                "fmul.d ft1, ft3, %[weight_times_invstd]\n"
                "fmadd.d %[grad_weight_0], ft4, ft3, %[grad_weight_0]\n"
                "0:\n"
                : [grad_weight_0] "+fr"(grad_weight_0), [grad_weight_1] "+fr"(grad_weight_1),
                  [grad_bias_0] "+fr"(grad_bias_0), [grad_bias_1] "+fr"(grad_bias_1), [mod_temp] "=r"(temp)
                : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
                  [weight_times_invstd] "fr"(weight_times_invstd), [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                  [work_mod_3] "r"(work_mod_3)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "memory");

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
                "fld %[temp_grad_bias], 0(%[grad_bias_scratch])\n"
                "fld %[temp_grad_weight], 0(%[grad_weight_scratch])\n"
                "fadd.d %[grad_bias_0], %[temp_grad_bias], %[grad_bias_0]\n"
                "fadd.d %[grad_weight_0], %[temp_grad_weight], %[grad_weight_0]\n"
                "fadd.d %[grad_bias_1], %[grad_bias_1], %[grad_bias_2]\n"
                "fadd.d %[grad_weight_1], %[grad_weight_1], %[grad_weight_2]\n"
                // interleave 0 resetting between stalls for addition
                "fsgnj.d %[grad_bias_2],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[grad_weight_2],%[ZERO],%[ZERO]\n"
                // don't need to synchronize here because the integer core can't
                // issue these instructions until the previous increments have
                // happened
                "fld %[invstd],0(%[invstd_scratch])\n"
                "fadd.d %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
                "fadd.d %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
                "fsgnj.d %[grad_bias_1],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[grad_weight_1],%[ZERO],%[ZERO]\n"
                "fld %[weight_times_invstd],0(%[weight_scratch])\n"
                "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
                "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
                "fsgnj.d %[grad_bias_0],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[grad_weight_0],%[ZERO],%[ZERO]\n"
                "fld %[running_mean_times_invstd],"
                "0(%[running_mean_scratch])\n"
                : [temp_grad_bias] "+fr"(temp_grad_bias), [temp_grad_weight] "+fr"(temp_grad_weight),
                  [grad_weight_0] "+fr"(grad_weight_0), [grad_weight_1] "+fr"(grad_weight_1),
                  [grad_bias_0] "+fr"(grad_bias_0), [grad_bias_1] "+fr"(grad_bias_1),
                  [running_mean_times_invstd] "=fr"(running_mean_times_invstd),
                  [weight_times_invstd] "=fr"(weight_times_invstd), [invstd] "=fr"(invstd)
                : [ZERO] "fr"(ZERO), [invstd_scratch] "r"(invstd_scratch), [weight_scratch] "r"(weight_scratch),
                  [running_mean_scratch] "r"(running_mean_scratch), [grad_bias_scratch] "r"(grad_bias_scratch),
                  [grad_weight_scratch] "r"(grad_weight_scratch), [grad_weight_2] "fr"(grad_weight_2),
                  [grad_bias_2] "fr"(grad_bias_2)
                : "ft0", "ft1", "ft2", "memory");
        } while (i < num_channels_to_process);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        snrt_ssr_disable();
        work_mod_3 = next_work_mod_3;
        grad_weight_scratch -= channel_stride * (num_channels_to_process - 1);
        grad_bias_scratch -= channel_stride * (num_channels_to_process - 1);
        // DUMP(222);
        // DUMP(work_in_tile);
    } while (work_in_tile != 0);
    // notify last tile done
    snrt_cluster_hw_barrier();
}

static inline void batchnorm_backward_main_loop(uint32_t C, uint32_t work_left,  // only present for dma
                                                uint32_t initial_work_in_tile,   // only present for dma
                                                uint32_t initial_work_mod_3, dm_comm_t* dm_comm,
                                                uint32_t tile_size_in_points, uint32_t compute_id,
                                                uint32_t num_compute_cores, batchnorm_backward_layer_t* l,
                                                double* grad_ofmap_scratch, double* ifmap_scratch,
                                                double* grad_ifmap_scratch, double* grad_weight_scratch,
                                                double* grad_bias_scratch, double* invstd_scratch,
                                                double* running_mean_scratch, double* weight_scratch, bool buf_flag) {
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();

    uint32_t num_channels_work_for_core = get_core_num_work_items(C, num_compute_cores, compute_id);

    if (snrt_is_dm_core()) {
        snrt_dma_wait_all();

        DUMP(ifmap_scratch);
        // DUMP(ifmap_scratch[0]);
        // signal first iteration
        // compute cores don't have to read dm comm the first time
        snrt_cluster_hw_barrier();
        // skip the first iteration in looping
        uint32_t point_start = initial_work_in_tile;
        uint32_t work_in_tile = initial_work_in_tile;
        bool is_last_iteration = false;
        uint32_t prev_point_start = 0;
        uint32_t num_points_work_in_prev_tile = initial_work_in_tile;
        // split the remaining work "nicely"
        uint32_t min_loops = ceildiv(work_left, tile_size_in_points);
        // align up to multiple of 3 because that avoids stalling in fpu the best
        uint32_t ideal_work_in_tile =
            min(align_up_non_power_of_2(ceildiv(work_left, min_loops), 3), tile_size_in_points);
        // uint32_t ideal_work_in_tile = 96;  // TODO CHANGE BACK
        while (work_left > 0) {
            // uint32_t estimated_max_tileable_work = tile_size_in_points;
            // (work_in_tile * ceildiv(C, num_compute_cores) * 5 *
            //  NUM_DOUBLES_LOADED_PER_CYCLE) /
            // (3 * C);
            work_in_tile = min(ideal_work_in_tile, work_left);
            // DUMP(work_left);
            DUMP(work_in_tile);
            work_left -= work_in_tile;
            // update comms
            dm_comm->num_points_work_in_tile = work_in_tile;
            dm_comm->work_mod_3 = work_in_tile % 3;
            dm_comm->is_last_iteration = work_left == 0;
            // comm what the next iteration will be
            // wait for potential previous gradifmap write out?
            snrt_dma_wait_all();
            snrt_dma_start_1d(&grad_ofmap_scratch[tile_size_in_points * C * buf_flag], &l->grad_ofmap[point_start * C],
                              work_in_tile * C * sizeof(double));
            snrt_dma_start_1d(&ifmap_scratch[tile_size_in_points * C * buf_flag], &l->ifmap[point_start * C],
                              work_in_tile * C * sizeof(double));
            snrt_cluster_hw_barrier();
            DUMP(55);
            snrt_dma_wait_all();
            // signal to core that current tile is ready to be computed on
            snrt_cluster_hw_barrier();
            DUMP(56);

            // DUMP(prev_point_start);

            snrt_dma_start_1d(&l->grad_ifmap[prev_point_start * C],
                              &grad_ifmap_scratch[tile_size_in_points * C * (!buf_flag)],  // take !buf_flag dma
                                                                                           // core is one
                                                                                           // iteration ahead of
                                                                                           // compute core
                              num_points_work_in_prev_tile * C * sizeof(double));
            prev_point_start = point_start;
            num_points_work_in_prev_tile = work_in_tile;
            point_start += work_in_tile;
            buf_flag = !buf_flag;
        }
        dm_comm->num_points_work_in_tile = 0;
        dm_comm->work_mod_3 = 0;
        dm_comm->is_last_iteration = true;
        // signal last iteration that there is no more work
        snrt_cluster_hw_barrier();
        // wait for last tile to finish
        snrt_cluster_hw_barrier();
        snrt_dma_start_1d(&l->grad_ifmap[prev_point_start * C],
                          &grad_ifmap_scratch[tile_size_in_points * C * (!buf_flag)],  // take !buf_flag dma
                                                                                       // core is one iteration
                                                                                       // ahead of compute core
                          num_points_work_in_prev_tile * C * sizeof(double));
    } else {
        if (num_channels_work_for_core == 0) {
            snrt_cluster_hw_barrier();
            while (initial_work_in_tile != 0) {
                // wait for dma to compute result
                snrt_cluster_hw_barrier();
                initial_work_in_tile = dm_comm->num_points_work_in_tile;
                // "signal" work is done
                snrt_cluster_hw_barrier();
            }
        } else {
            batchnorm_backward_tile_fp64_looped(
                &grad_ofmap_scratch[compute_id], &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                &running_mean_scratch[compute_id], &weight_scratch[compute_id], &invstd_scratch[compute_id],
                &grad_bias_scratch[compute_id], &grad_weight_scratch[compute_id], C, initial_work_in_tile,
                initial_work_mod_3, tile_size_in_points, num_channels_work_for_core, num_compute_cores, dm_comm);
        }
    }

    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
}