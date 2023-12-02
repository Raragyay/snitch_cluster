#pragma once

#include <stdbool.h>
#include "batchnorm_data_structures.h"
#include "snrt.h"

#define PERF_DEBUG 1
#define PERF_WHOLE_BLOCK 0

#if PERF_WHOLE_BLOCK
#define SNRT_SECTIONED_MCYCLE() 0xdeadbeef
#else
#define SNRT_SECTIONED_MCYCLE() (snrt_mcycle())
#endif
#define min(a, b) ((a) < (b) ? (a) : (b))
#define ceildiv(a, b) ((((a)-1) / (b)) + 1)
#if PERF_DEBUG
NAMED_DUMP(uint32_t, PERF_RESULT, 0x7C4)
#endif

static inline uint32_t __attribute__((__const__))
get_core_num_work_items(uint32_t num_work_items, uint32_t num_compute_cores,
                        uint32_t compute_id) {
    return num_work_items / num_compute_cores +
           (compute_id < (num_work_items % num_compute_cores));
}

static inline uint32_t __attribute__((__const__))
get_offset_for_core_work_blocked(uint32_t num_work_items,
                                 uint32_t num_compute_cores,
                                 uint32_t compute_id) {
    return num_work_items / num_compute_cores * compute_id +
           min(compute_id, (num_work_items % num_compute_cores));
}

// use ALIGN_UP for powers of 2 - it'll be much faster
static inline uint32_t __attribute__((const))
align_up_non_power_of_2(uint32_t n, uint32_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
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
        dump_PERF_RESULT(res);
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

static inline void __attribute__((always_inline))
batchnorm_backward_fp64_no_loop(
    const double* grad_ofmap_scratch, double* grad_ifmap_scratch,
    const double* ifmap_scratch, const double* running_mean_scratch,
    const double* weight_scratch, const double* invstd_scratch,
    double* grad_bias_scratch, double* grad_weight_scratch, uint32_t C,
    uint32_t num_points_work_for_core,  // requires: > 0
    uint32_t work_mod_2,  // precompute to avoid icache branch misses
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, bool is_first_iteration, bool force_configure) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    if (is_first_iteration || force_configure) {
        snrt_ssr_loop_2d(
            SNRT_SSR_DM_ALL,
            num_points_work_for_core,  // dimension of inner loop
            num_channels_to_process,   // dimension of outer loop
            C * sizeof(double),  // stride per inner loop iteration: 1 point
            channel_stride *
                sizeof(double));  // stride per outer loop iteration
        snrt_ssr_repeat(SNRT_SSR_DM0, 3);
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
    bool frep = num_points_work_for_core >= 2;
    uint32_t work_div_2_sub_1 = num_points_work_for_core / 2 -
                                1;  // can underflow, but then frep won't happen
    register volatile uint32_t i =
        0;  // updated during frep for pseudo-dual issue
    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    register double grad_weight_0 = ZERO;
    register double grad_weight_1 = ZERO;
    register double grad_bias_0 = ZERO;
    register double grad_bias_1 = ZERO;
    register double invstd = *invstd_scratch;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_enable();
    register double weight_times_invstd = *weight_scratch;
    register double running_mean_times_invstd = *running_mean_scratch;
    // do 1 loop
    do {  // while (i < num_channels_to_process)
        asm volatile(
            "fmul.d %[running_mean_times_invstd],%[running_mean_times_invstd],%[invstd]\n"
            "fmul.d %[weight_times_invstd],%[weight_times_invstd],%[invstd]\n"
            : [running_mean_times_invstd] "+fr"(running_mean_times_invstd),
              [weight_times_invstd] "+fr"(weight_times_invstd)
            : [invstd] "fr"(invstd)
            : "ft0", "ft1", "ft2");
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 8, 0, 0 \n"
                "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fmsub.d ft6, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fadd.d %[grad_bias_0], ft0, %[grad_bias_0]\n"
                "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                "fmadd.d %[grad_weight_0], ft4, ft0, %[grad_weight_0]\n"
                "fadd.d %[grad_bias_1], ft0, %[grad_bias_1]\n"
                "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                "fmadd.d %[grad_weight_1], ft6, ft0, %[grad_weight_1]\n"
                : [grad_weight_0] "+fr"(grad_weight_0),
                  [grad_weight_1] "+fr"(grad_weight_1),
                  [grad_bias_0] "+fr"(grad_bias_0),
                  [grad_bias_1] "+fr"(grad_bias_1)
                : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
                  [weight_times_invstd] "fr"(weight_times_invstd),
                  [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                  [n_frep] "r"(work_div_2_sub_1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[grad_bias_scratch], %[grad_bias_scratch],%[channel_stride_in_bytes]\n"
            "add %[grad_weight_scratch], %[grad_weight_scratch], %[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "beq %[num_channels_to_process], %[i], 2f\n"  // shortcut when only
                                                          // 1 channel
            "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
            "add %[weight_scratch], %[weight_scratch],%[channel_stride_in_bytes]\n"
            "add %[running_mean_scratch],%[running_mean_scratch],%[channel_stride_in_bytes]\n"
            "2:\n"
            : [invstd_scratch] "+r"(invstd_scratch),
              [weight_scratch] "+r"(weight_scratch),
              [running_mean_scratch] "+r"(running_mean_scratch),
              [grad_bias_scratch] "+r"(grad_bias_scratch),
              [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");

        asm volatile(
            "beqz %[work_mod_2], 0f\n"  // mod is 0
            "1:\n"
            "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
            "fadd.d %[grad_bias_0], ft0, %[grad_bias_0]\n"
            "fmul.d ft1, ft0, %[weight_times_invstd]\n"
            "fmadd.d %[grad_weight_0], ft4, ft0, %[grad_weight_0]\n"
            "0:\n"
            : [grad_weight_0] "+fr"(grad_weight_0), [grad_bias_0] "+fr"(
                                                        grad_bias_0)
            : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
              [weight_times_invstd] "fr"(weight_times_invstd),
              [invstd] "fr"(invstd), [zero] "fr"(ZERO),
              [work_mod_2] "r"(work_mod_2)
            : "ft0", "ft1", "ft2", "ft4");

        // in plain C:
        // if (is_first_iteration) {
        //     grad_bias_scratch[channel] =
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] =
        //         grad_weight_0 + grad_weight_1;

        // } else {
        //     grad_bias_scratch[channel] +=
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] +=
        //         grad_weight_0 + grad_weight_1;
        // }
        // invstd = *invstd_scratch;
        // weight = *weight_scratch;
        // running_mean = *running_mean_scratch;
        // grad_bias_0 = grad_bias_1 = grad_weight_0 = grad_weight_1 = 0;
        register double temp_grad_bias, temp_grad_weight;
        asm volatile(
            "bnez %[is_first_iteration], 3f\n"
            "fld %[temp_grad_bias], 0(%[grad_bias_scratch])\n"
            "fld %[temp_grad_weight], 0(%[grad_weight_scratch])\n"
            "fadd.d %[grad_bias_0], %[temp_grad_bias], %[grad_bias_0]\n"
            "fadd.d %[grad_weight_0], %[temp_grad_weight], %[grad_weight_0]\n"
            "3:\n"
            // interleave 0 resetting and loading between fadd latency
            // don't need to synchronize here because the integer core can't
            // issue these instructions until the previous increments have
            // happened
            "fld %[invstd],0(%[invstd_scratch])\n"
            "fld %[weight_times_invstd],0(%[weight_scratch])\n"
            "fadd.d %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
            "fadd.d %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
            "fld %[running_mean_times_invstd],0(%[running_mean_scratch])\n"
            "fsgnj.d %[grad_bias_1],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[grad_weight_1],%[ZERO],%[ZERO]\n"
            "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
            "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
            "fsgnj.d %[grad_bias_0],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[grad_weight_0],%[ZERO],%[ZERO]\n"
            : [temp_grad_bias] "+fr"(temp_grad_bias),
              [temp_grad_weight] "+fr"(temp_grad_weight),
              [grad_weight_0] "+fr"(grad_weight_0),
              [grad_weight_1] "+fr"(grad_weight_1),
              [grad_bias_0] "+fr"(grad_bias_0),
              [grad_bias_1] "+fr"(grad_bias_1),
              [running_mean_times_invstd] "=fr"(running_mean_times_invstd),
              [weight_times_invstd] "=fr"(weight_times_invstd),
              [invstd] "=fr"(invstd)
            : [is_first_iteration] "r"(is_first_iteration), [ZERO] "fr"(ZERO),
              [invstd_scratch] "r"(invstd_scratch),
              [weight_scratch] "r"(weight_scratch),
              [running_mean_scratch] "r"(running_mean_scratch),
              [grad_bias_scratch] "r"(grad_bias_scratch),
              [grad_weight_scratch] "r"(grad_weight_scratch)
            : "ft0", "ft1", "ft2");
    } while (i < num_channels_to_process);
    // don't need to fpu_fence since last 3 instructions are inconsequential
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void __attribute__((always_inline))
batchnorm_backward_tile_fp64_looped(
    const double* grad_ofmap_scratch,
    double*
        grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
    const double* ifmap_scratch, const double* running_mean_scratch,
    const double* weight_scratch, const double* invstd_scratch,
    double* grad_bias_scratch, double* grad_weight_scratch, uint32_t C,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_mod_2,    // precompute to avoid icache branch misses
    uint32_t work_div_2_sub_1, uint32_t tile_size_in_points,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_3;
    register bool frep = work_in_tile >= 3;
    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");

    bool buf_flag = 0;
    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_points * C * sizeof(double);
    const uint32_t channel_array_reset_dist =
        channel_stride * num_channels_to_process * sizeof(double);
    const uint32_t inner_loop_stride = C * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);
    DUMP(33);
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        work_in_tile,             // dimension of inner loop
        num_channels_to_process,  // dimension of outer loop
        inner_loop_stride,        // stride per inner loop iteration: 1 point
        outer_loop_stride);       // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 3);

    snrt_ssr_enable();
    // TODO: fix num_channels_work_for_core == 0.
    do {
        register volatile uint32_t i =
            0;  // updated during frep for pseudo-dual issue
        register double grad_weight_0 = ZERO;
        register double grad_weight_1 = ZERO;
        register double grad_weight_2 = ZERO;
        register double grad_bias_0 = ZERO;
        register double grad_bias_1 = ZERO;
        register double grad_bias_2 = ZERO;
        register double invstd = *invstd_scratch;
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
        register double weight_times_invstd = *weight_scratch;
        register double running_mean_times_invstd = *running_mean_scratch;
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            asm volatile(
                "fmul.d "
                "%[running_mean_times_invstd],%[running_mean_times_invstd],%["
                "invstd]\n"
                "fmul.d "
                "%[weight_times_invstd],%[weight_times_invstd],%[invstd]\n"
                : [running_mean_times_invstd] "+fr"(running_mean_times_invstd),
                  [weight_times_invstd] "+fr"(weight_times_invstd)
                : [invstd] "fr"(invstd)
                : "ft0", "ft1", "ft2");
            if (frep) {
                asm volatile(
                    "frep.o %[n_frep], 8, 0, 0 \n"
                    "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
                    "fmsub.d ft6, ft2, %[invstd], %[running_mean_times_invstd]\n"
                    "fadd.d %[grad_bias_0], ft0, %[grad_bias_0]\n"
                    "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                    "fmadd.d %[grad_weight_0], ft4, ft0, %[grad_weight_0]\n"
                    "fadd.d %[grad_bias_1], ft0, %[grad_bias_1]\n"
                    "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                    "fmadd.d %[grad_weight_1], ft6, ft0, %[grad_weight_1]\n"
                    : [grad_weight_0] "+fr"(grad_weight_0),
                      [grad_weight_1] "+fr"(grad_weight_1),
                      [grad_bias_0] "+fr"(grad_bias_0),
                      [grad_bias_1] "+fr"(grad_bias_1)
                    : [running_mean_times_invstd] "fr"(
                          running_mean_times_invstd),
                      [weight_times_invstd] "fr"(weight_times_invstd),
                      [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                      [n_frep] "r"(
                          work_div_2_sub_1)  // we repeat n_frep+1 times
                    : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
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
                : [invstd_scratch] "+r"(invstd_scratch),
                  [weight_scratch] "+r"(weight_scratch),
                  [running_mean_scratch] "+r"(running_mean_scratch),
                  [grad_bias_scratch] "+r"(grad_bias_scratch),
                  [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+&r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2", "memory");

            // TODO
            // if (is_last_channel) {
            //     invstd_scratch -= channel_stride * num_channels_to_process;
            //     running_mean_scratch -= channel_stride *
            //     num_channels_to_process; weight_scratch -= channel_stride *
            //     num_channels_to_process; buf_flag = !buf_flag;
            //     snrt_cluster_hw_barrier();
            //     work_in_tile = (dm_comm)->num_points_work_in_tile;
            //     work_mod_2 = (dm_comm)->work_mod_2;
            //     work_div_2_sub_1 = (dm_comm)->work_div_2_sub_1;
            //     frep = work_in_tile >= 3;
            //     if (prev_work != work_in_tile) {
            //         prev_work = work_in_tile;
            //         update b0 for 2d ssr
            //     }
            //     if (buf_flag) {
            //         grad_ofmap_scratch += buf_flag_offset / sizeof(double);
            //         grad_ifmap_scratch += buf_flag_offset / sizeof(double);
            //         ifmap_scratch += buf_flag_offset / sizeof(double);
            //     } else {
            //         grad_ofmap_scratch -= buf_flag_offset / sizeof(double);
            //         grad_ifmap_scratch -= buf_flag_offset / sizeof(double);
            //         ifmap_scratch -= buf_flag_offset / sizeof(double);
            //     }
            // }

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                "sub %[invstd_scratch], %[invstd_scratch], %[channel_array_reset_dist]\n"
                "sub %[weight_scratch], %[weight_scratch],%[channel_array_reset_dist]\n"
                "sub %[running_mean_scratch],%[running_mean_scratch],%[channel_array_reset_dist]\n "
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_3], %[work_mod_2_offset](%[dm_comm])\n"
                "lw %[work_div_2_sub_1], %[work_div_2_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 3\n"  // cmp frep < 3, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                                                            // the bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
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
                // // write ssr config if this is not the last iteration
                // "beqz %[work_in_tile], 5f\n"
                // "scfgwi %[grad_ofmap_scratch],%[DM_0] | %[REG_RPTR_2D]<<5\n"
                // "scfgwi %[grad_ifmap_scratch],%[DM_1] | %[REG_WPTR_2D]<<5\n"
                // "scfgwi %[ifmap_scratch],%[DM_2] | %[REG_RPTR_2D]<<5\n"
                // "5:\n"
                : [buf_flag] "+r"(buf_flag),
                  [invstd_scratch] "+r"(invstd_scratch),
                  [weight_scratch] "+r"(weight_scratch),
                  [running_mean_scratch] "+r"(running_mean_scratch),
                  [work_in_tile] "=r"(work_in_tile),
                  [next_work_mod_3] "=r"(next_work_mod_3),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep),
                  [work_div_2_sub_1] "=r"(work_div_2_sub_1),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch),
                  [grad_ifmap_scratch] "+r"(grad_ifmap_scratch),
                  [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i),
                  [num_channels_to_process] "r"(num_channels_to_process),
                  [channel_array_reset_dist] "r"(channel_array_reset_dist),
                  [work_in_tile_offset] "i"(
                      offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_2_offset] "i"(offsetof(dm_comm_t, work_mod_2)),
                  [work_div_2_sub_1_offset] "i"(
                      offsetof(dm_comm_t, work_div_2_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS),
                  [REG_WPTR_2D] "i"(REG_WPTR + 2),
                  [REG_RPTR_2D] "i"(REG_RPTR + 2),
                  [DM_ALL] "i"(SNRT_SSR_DM_ALL), [DM_0] "i"(SNRT_SSR_DM0),
                  [DM_1] "i"(SNRT_SSR_DM1), [DM_2] "i"(SNRT_SSR_DM2),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1),
                  [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride),
                  [dm_comm] "r"(dm_comm), [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0", "memory");

            asm volatile(
                "beqz %[work_mod_2], 0f\n"  // mod is 0
                "1:\n"
                "fmsub.d ft4, ft2, %[invstd], %[running_mean_times_invstd]\n"
                "fadd.d %[grad_bias_0], ft0, %[grad_bias_0]\n"
                "fmul.d ft1, ft0, %[weight_times_invstd]\n"
                "fmadd.d %[grad_weight_0], ft4, ft0, %[grad_weight_0]\n"
                "0:\n"
                : [grad_weight_0] "+fr"(grad_weight_0), [grad_bias_0] "+fr"(
                                                            grad_bias_0)
                : [running_mean_times_invstd] "fr"(running_mean_times_invstd),
                  [weight_times_invstd] "fr"(weight_times_invstd),
                  [invstd] "fr"(invstd), [zero] "fr"(ZERO),
                  [work_mod_2] "r"(work_mod_2)
                : "ft0", "ft1", "ft2", "ft4");

            // in plain C:
            // if (is_first_iteration) {
            //     grad_bias_scratch[channel] =
            //         grad_bias_0 + grad_bias_1;
            //     grad_weight_scratch[channel] =
            //         grad_weight_0 + grad_weight_1;

            // } else {
            //     grad_bias_scratch[channel] +=
            //         grad_bias_0 + grad_bias_1;
            //     grad_weight_scratch[channel] +=
            //         grad_weight_0 + grad_weight_1;
            // }
            // invstd = *invstd_scratch;
            // weight = *weight_scratch;
            // running_mean = *running_mean_scratch;
            // grad_bias_0 = grad_bias_1 = grad_weight_0 = grad_weight_1 = 0;
            register double temp_grad_bias, temp_grad_weight;
            asm volatile(
                "fld %[temp_grad_bias], 0(%[grad_bias_scratch])\n"
                "fld %[temp_grad_weight], 0(%[grad_weight_scratch])\n"
                "fadd.d %[grad_bias_0], %[temp_grad_bias], %[grad_bias_0]\n"
                "fadd.d %[grad_weight_0], %[temp_grad_weight], %[grad_weight_0]\n"
                // interleave 0 resetting and loading between fadd latency
                // don't need to synchronize here because the integer core can't
                // issue these instructions until the previous increments have
                // happened
                "fld %[invstd],0(%[invstd_scratch])\n"
                "fld %[weight_times_invstd],0(%[weight_scratch])\n"
                "fadd.d %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
                "fadd.d %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
                "fld %[running_mean_times_invstd],0(%[running_mean_scratch])\n"
                "fsgnj.d %[grad_bias_1],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[grad_weight_1],%[ZERO],%[ZERO]\n"
                "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
                "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
                "fsgnj.d %[grad_bias_0],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[grad_weight_0],%[ZERO],%[ZERO]\n"
                : [temp_grad_bias] "+fr"(temp_grad_bias),
                  [temp_grad_weight] "+fr"(temp_grad_weight),
                  [grad_weight_0] "+fr"(grad_weight_0),
                  [grad_weight_1] "+fr"(grad_weight_1),
                  [grad_bias_0] "+fr"(grad_bias_0),
                  [grad_bias_1] "+fr"(grad_bias_1),
                  [running_mean_times_invstd] "=fr"(running_mean_times_invstd),
                  [weight_times_invstd] "=fr"(weight_times_invstd),
                  [invstd] "=fr"(invstd)
                : [ZERO] "fr"(ZERO), [invstd_scratch] "r"(invstd_scratch),
                  [weight_scratch] "r"(weight_scratch),
                  [running_mean_scratch] "r"(running_mean_scratch),
                  [grad_bias_scratch] "r"(grad_bias_scratch),
                  [grad_weight_scratch] "r"(grad_weight_scratch)
                : "ft0", "ft1", "ft2");
        } while (i < num_channels_to_process);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        // snrt_ssr_disable();
        // notify that computations for this tile are done
        work_mod_2 = next_work_mod_3;
        grad_weight_scratch -= channel_stride * (num_channels_to_process - 1);
        grad_bias_scratch -= channel_stride * (num_channels_to_process - 1);
        // DUMP(222);
        // DUMP(work_in_tile);
    } while (work_in_tile != 0);
    // // notify last tile done
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}

static inline void __attribute__((always_inline))
batchnorm_backward_fp32_no_loop(
    const v2s* grad_ofmap_scratch,
    v2s* grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
    const v2s* ifmap_scratch, const v2s* running_mean_scratch,
    const v2s* weight_scratch, const v2s* invstd_scratch,
    v2s* grad_bias_scratch, v2s* grad_weight_scratch,
    uint32_t num_bytes_per_point,
    uint32_t num_points_work_for_core,  // requires: > 0
    uint32_t work_mod_2,  // precompute to avoid icache branch misses
    uint32_t num_doubles_to_process,  //  requires: > 0
    uint32_t channel_stride) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    DUMP(num_points_work_for_core);
    DUMP(num_doubles_to_process);
    DUMP(num_bytes_per_point);
    snrt_ssr_loop_2d(
        SNRT_SSR_DM_ALL,
        num_points_work_for_core,  // dimension of inner loop
        num_doubles_to_process,    // dimension of outer loop
        num_bytes_per_point,       // stride per inner loop iteration: 1 point
        channel_stride * sizeof(double));  // stride per outer loop iteration
    snrt_ssr_repeat(SNRT_SSR_DM0, 3);

    // thought: how to minimize # flops?
    // dy is used for: grad_bias: grad_bias[C] += dy
    //                 grad_ifmap: grad_ifmap[C] =
    //                    (dy * invstd[C] * weight[C])
    //                 grad_weight: grad_weight[C] +=
    //                    (dy * (x[i,C]-running_mean[C]) * invstd[C])
    // for grad_bias: vfadd is sufficient (grad_bias <- dy + grad_bias)
    // for grad_ifmap: vfmul is sufficient (grad_ifmap <- dy * (invstd*weight))
    // for grad_weight: previously i did an fmsub. but there is no vfmsub right
    // now.
    //              Intermediate steps harder though:
    //              (dy * (x[i,C]-running_mean[C]) * invstd[C])
    //              = (dy * (x[i,C]*invstd[C]-(running_mean[C] * invstd[C])))
    // Option 1: sub, mul, mul. 3 instr
    // Option 2: mul, sub, mul. 3 instr
    // Conclusion: I think you have to do 3 instructions without fmadd/fmsub

    bool frep = num_points_work_for_core >= 2;
    uint32_t work_div_2_sub_1 = num_points_work_for_core / 2 -
                                1;  // can underflow, but then frep won't happen
    register volatile uint32_t i =
        0;                         // updated during frep for pseudo-dual issue
    register v2s ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"  // vfcvt.s.x raises exception
                                             // despite smallfloat spec
                 : [ZERO] "=fr"(ZERO.f64)::"ft0", "ft1", "ft2");
    register v2s grad_weight_0 = ZERO;
    register v2s grad_weight_1 = ZERO;
    register v2s grad_bias_0 = ZERO;
    register v2s grad_bias_1 = ZERO;
    register v2s invstd;
    invstd.f64 = invstd_scratch->f64;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_enable();
    register v2s weight_times_invstd;
    weight_times_invstd.f64 = weight_scratch->f64;
    register v2s running_mean;
    running_mean.f64 = running_mean_scratch->f64;
    // do 1 loop
    do {  // while (i < num_channels_to_process)
        // Can only manual unroll 3 times since the max for frep is 16
        asm volatile(
            "vfmul.s %[weight_times_invstd],%[weight_times_invstd],%[invstd]\n"
            : [weight_times_invstd] "+fr"(weight_times_invstd.f64)
            : [invstd] "fr"(invstd.f64)
            : "ft0", "ft1", "ft2");
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 10, 0, 0 \n"
                // for grad_ifmap: x - running_mean
                "vfsub.s ft3, ft2, %[running_mean]\n"
                "vfsub.s ft5, ft2, %[running_mean]\n"
                // for grad_ifmap: dy * invstd
                "vfmul.s ft4, %[invstd], ft0\n"
                "vfadd.s %[grad_bias_0], ft0, %[grad_bias_0]\n"
                "vfmul.s ft1, %[weight_times_invstd], ft0\n"
                "vfmul.s ft6, %[invstd], ft0\n"
                "vfadd.s %[grad_bias_1], ft0, %[grad_bias_1]\n"
                "vfmul.s ft1, %[weight_times_invstd], ft0\n"
                // for grad_ifmap: (x - running_mean) * (dy * invstd)
                "vfmac.s %[grad_weight_0], ft3, ft4\n"
                "vfmac.s %[grad_weight_1], ft5, ft6\n"
                : [grad_weight_0] "+fr"(grad_weight_0.f64),
                  [grad_weight_1] "+fr"(grad_weight_1.f64),
                  [grad_bias_0] "+fr"(grad_bias_0.f64),
                  [grad_bias_1] "+fr"(grad_bias_1.f64)
                : [running_mean] "fr"(running_mean.f64),
                  [weight_times_invstd] "fr"(weight_times_invstd.f64),
                  [invstd] "fr"(invstd.f64), [zero] "fr"(ZERO.f64),
                  [n_frep] "r"(work_div_2_sub_1)  // we repeat n_frep+1 times
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[grad_bias_scratch], %[grad_bias_scratch],%[channel_stride_in_bytes]\n"
            "add %[grad_weight_scratch], %[grad_weight_scratch], %[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "beq %[num_doubles_to_process], %[i], 2f\n"  // shortcut when only
                                                         // 1 double to process
            "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
            "add %[weight_scratch], %[weight_scratch],%[channel_stride_in_bytes]\n"
            "add %[running_mean_scratch],%[running_mean_scratch],%[channel_stride_in_bytes]\n"
            "2:\n"
            : [invstd_scratch] "+r"(invstd_scratch),
              [weight_scratch] "+r"(weight_scratch),
              [running_mean_scratch] "+r"(running_mean_scratch),
              [grad_bias_scratch] "+r"(grad_bias_scratch),
              [grad_weight_scratch] "+r"(grad_weight_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_doubles_to_process] "r"(num_doubles_to_process)
            : "ft0", "ft1", "ft2");

        asm volatile(
            "beqz %[work_mod_2], 0f\n"  // mod is 0
            "1:\n"
            // for grad_ifmap: x - running_mean
            "vfsub.s ft3, ft2, %[running_mean]\n"
            // for grad_ifmap: dy * invstd
            "vfmul.s ft4, %[invstd], ft0\n"
            "vfadd.s %[grad_bias_0], ft0, %[grad_bias_0]\n"
            "vfmul.s ft1, %[weight_times_invstd], ft0\n"
            // for grad_ifmap: (x - running_mean) * (dy * invstd)
            "vfmac.s %[grad_weight_0], ft3, ft4\n"
            "0:\n"
            : [grad_weight_0] "+fr"(grad_weight_0.f64), [grad_bias_0] "+fr"(
                                                            grad_bias_0.f64)
            : [running_mean] "fr"(running_mean.f64),
              [weight_times_invstd] "fr"(weight_times_invstd.f64),
              [invstd] "fr"(invstd.f64), [zero] "fr"(ZERO.f64),
              [work_mod_2] "r"(work_mod_2)
            : "ft0", "ft1", "ft2", "ft3", "ft4");

        // in plain C:
        // if (is_first_iteration) {
        //     grad_bias_scratch[channel] =
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] =
        //         grad_weight_0 + grad_weight_1;

        // } else {
        //     grad_bias_scratch[channel] +=
        //         grad_bias_0 + grad_bias_1;
        //     grad_weight_scratch[channel] +=
        //         grad_weight_0 + grad_weight_1;
        // }
        // invstd = *invstd_scratch;
        // weight = *weight_scratch;
        // running_mean = *running_mean_scratch;
        // grad_bias_0 = grad_bias_1 = grad_weight_0 = grad_weight_1 = 0;
        register double temp_grad_bias, temp_grad_weight;
        asm volatile(
            // interleave 0 resetting and loading between fadd latency
            // don't need to synchronize here because the integer core can't
            // issue these instructions until the previous increments have
            // happened
            "fld %[invstd],0(%[invstd_scratch])\n"
            "fld %[weight_times_invstd],0(%[weight_scratch])\n"
            "vfadd.s %[grad_bias_0], %[grad_bias_1], %[grad_bias_0]\n"
            "vfadd.s %[grad_weight_0], %[grad_weight_1], %[grad_weight_0]\n"
            "fld %[running_mean],0(%[running_mean_scratch])\n"
            "vfsgnj.s %[grad_bias_1],%[ZERO],%[ZERO]\n"
            "vfsgnj.s %[grad_weight_1],%[ZERO],%[ZERO]\n"
            "fsd %[grad_bias_0], 0(%[grad_bias_scratch])\n"
            "fsd %[grad_weight_0], 0(%[grad_weight_scratch])\n"
            "vfsgnj.s %[grad_bias_0],%[ZERO],%[ZERO]\n"
            "vfsgnj.s %[grad_weight_0],%[ZERO],%[ZERO]\n"
            : [grad_weight_0] "+fr"(grad_weight_0.f64),
              [grad_weight_1] "+fr"(grad_weight_1.f64),
              [grad_bias_0] "+fr"(grad_bias_0.f64),
              [grad_bias_1] "+fr"(grad_bias_1.f64),
              [running_mean] "=fr"(running_mean.f64),
              [weight_times_invstd] "=fr"(weight_times_invstd.f64),
              [invstd] "=fr"(invstd.f64)
            : [ZERO] "fr"(ZERO.f64), [invstd_scratch] "r"(invstd_scratch),
              [weight_scratch] "r"(weight_scratch),
              [running_mean_scratch] "r"(running_mean_scratch),
              [grad_bias_scratch] "r"(grad_bias_scratch),
              [grad_weight_scratch] "r"(grad_weight_scratch)
            : "ft0", "ft1", "ft2");
    } while (i < num_doubles_to_process);
    // don't need to fpu_fence since last 3 instructions are inconsequential
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    snrt_ssr_disable();
}

static inline void batchnorm_backward_main_loop(
    uint32_t C, uint32_t work_left,  // only present for dma
    uint32_t initial_work_in_tile, uint32_t initial_work_mod_3,
    uint32_t initial_work_div_3_sub_1, dm_comm_t* dm_comm,
    uint32_t tile_size_in_points, uint32_t compute_id,
    uint32_t num_compute_cores, batchnorm_backward_layer_t* l,
    double* grad_ofmap_scratch, double* ifmap_scratch,
    double* grad_ifmap_scratch, double* grad_weight_scratch,
    double* grad_bias_scratch, double* invstd_scratch,
    double* running_mean_scratch, double* weight_scratch, bool buf_flag) {
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();

    uint32_t num_channels_work_for_core =
        get_core_num_work_items(C, num_compute_cores, compute_id);

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
        // align up to multiple of 2 because that avoids stalling in fpu the
        // best
        uint32_t ideal_work_in_tile = min(
            ALIGN_UP(ceildiv(work_left, min_loops), 2), tile_size_in_points);
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
            dm_comm->work_mod_2 = work_in_tile % 2;
            dm_comm->work_div_2_sub_1 = work_in_tile / 2 - 1;
            // comm what the next iteration will be
            // wait for potential previous gradifmap write out?
            snrt_dma_wait_all();
            snrt_dma_start_1d(
                &grad_ofmap_scratch[tile_size_in_points * C * buf_flag],
                &l->grad_ofmap[point_start * C],
                work_in_tile * C * sizeof(double));
            snrt_dma_start_1d(
                &ifmap_scratch[tile_size_in_points * C * buf_flag],
                &l->ifmap[point_start * C], work_in_tile * C * sizeof(double));
            // signal to core that current tile has information ready
            snrt_cluster_hw_barrier();
            DUMP(55);
            snrt_dma_wait_all();
            // wait for previous tile to be finished computing, signify current
            // tile inputs done loading
            snrt_cluster_hw_barrier();
            DUMP(56);

            // DUMP(prev_point_start);

            snrt_dma_start_1d(
                &l->grad_ifmap[prev_point_start * C],
                &grad_ifmap_scratch[tile_size_in_points * C *
                                    (!buf_flag)],  // take !buf_flag dma
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
        dm_comm->work_mod_2 = 0;
        dm_comm->work_div_2_sub_1 = 0xdeadbeef;
        // signal last iteration that there is no more work
        snrt_cluster_hw_barrier();
        // wait for last tile to finish
        snrt_cluster_hw_barrier();
        snrt_dma_start_1d(
            &l->grad_ifmap[prev_point_start * C],
            &grad_ifmap_scratch[tile_size_in_points * C *
                                (!buf_flag)],  // take !buf_flag dma
                                               // core is one iteration
                                               // ahead of compute core
            num_points_work_in_prev_tile * C * sizeof(double));
    } else {
        if (num_channels_work_for_core == 0) {
            // start up first tile
            snrt_cluster_hw_barrier();
            while (initial_work_in_tile != 0) {
                // wait for dma to compute result and signify work is done
                snrt_cluster_hw_barrier();
                initial_work_in_tile = dm_comm->num_points_work_in_tile;
                // "signal" work is done
                snrt_cluster_hw_barrier();
            }
        } else {
            batchnorm_backward_tile_fp64_looped(
                &grad_ofmap_scratch[compute_id],
                &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id],
                &running_mean_scratch[compute_id], &weight_scratch[compute_id],
                &invstd_scratch[compute_id], &grad_bias_scratch[compute_id],
                &grad_weight_scratch[compute_id], C, initial_work_in_tile,
                initial_work_mod_3, initial_work_div_3_sub_1,
                tile_size_in_points, num_channels_work_for_core,
                num_compute_cores, dm_comm);
        }
    }

    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_tile_fp64_no_loop_1(
    const double* grad_ofmap_scratch, const double* ifmap_scratch,
    const double* current_mean_scratch, double* sum_scratch, double* dotp_scratch,
    uint32_t C, uint32_t num_points_work_for_core_in_tile, uint32_t work_mod_3,
    uint32_t work_div_3_sub_1, uint32_t num_channels_to_process,
    uint32_t channel_stride, bool is_first_iteration, bool force_configure) {
    DUMP(22);
    if (is_first_iteration || force_configure) {
        snrt_ssr_loop_2d(
            SNRT_SSR_DM_ALL,
            num_points_work_for_core_in_tile,  // dimension of inner loop
            num_channels_to_process,           // dimension of outer loop
            C * sizeof(double),  // stride per inner loop iteration: 1 point
            channel_stride *
                sizeof(double));  // stride per outer loop iteration
    }
    bool frep = num_points_work_for_core_in_tile >= 3;
    register volatile uint32_t i = 0;
    register double ZERO asm("ft9");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n"
                 : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    register double sum_0 = ZERO;
    register double sum_1 = ZERO;
    register double sum_2 = ZERO;
    register double dotp_0 = ZERO;
    register double dotp_1 = ZERO;
    register double dotp_2 = ZERO;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, ifmap_scratch);
    do {
        register double current_mean = *current_mean_scratch;
        snrt_ssr_enable();
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 15, 0, 0 \n"
                "fadd.d ft3, ft0, %[zero] \n"
                "fadd.d ft5, ft0, %[zero] \n"
                "fadd.d ft7, ft0, %[zero] \n"
                "fsub.d ft4, ft1, %[current_mean]\n"
                "fsub.d ft6, ft1, %[current_mean]\n"
                "fsub.d ft8, ft1, %[current_mean]\n"
                "fmul.d ft4, ft4, ft3\n"
                "fmul.d ft6, ft6, ft5\n"
                "fmul.d ft8, ft8, ft7\n"
                "fadd.d %[sum_0], ft3, %[sum_0] \n"
                "fadd.d %[sum_1], ft5, %[sum_1] \n"
                "fadd.d %[sum_2], ft7, %[sum_2] \n"
                "fadd.d %[dotp_0], ft4, %[dotp_0]\n"
                "fadd.d %[dotp_1], ft6, %[dotp_1]\n"
                "fadd.d %[dotp_2], ft8, %[dotp_2]\n"
                : [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1), [sum_2] "+fr"(sum_2),
                [dotp_0] "+fr"(dotp_0), [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2)
                : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
                [n_frep] "r"(work_div_3_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "beqz %[i], 1f\n"
            "add %[sum_scratch], %[sum_scratch],%[channel_stride_in_bytes]\n"
            "add %[dotp_scratch], %[dotp_scratch], %[channel_stride_in_bytes]\n"
            "1:\n"
            "addi %[i], %[i], 1\n"
            "beq %[num_channels_to_process], %[i], 2f\n"  // shortcut when only 1 channel
            "add %[current_mean_scratch], %[current_mean_scratch], %[channel_stride_in_bytes]\n"
            "2:\n"
            : [current_mean_scratch] "+r"(current_mean_scratch),
              [sum_scratch] "+r"(sum_scratch),
              [dotp_scratch] "+r"(dotp_scratch), [i] "+r"(i),
              [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
              [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");
        
        register uint32_t mod_temp;
        asm volatile(
            "beqz %[work_mod_3], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod_3], 1\n"  // is last bit 1? if no, then mod is 2
            "bnez %[mod_temp], 1f\n"                // jump to 1 if yes
            "2:\n"
            "fadd.d ft3, ft0, %[zero] \n"
            "fadd.d ft5, ft0, %[zero] \n"
            "fsub.d ft4, ft1, %[current_mean]\n"
            "fsub.d ft6, ft1, %[current_mean]\n"
            "fmul.d ft4, ft4, ft3\n"
            "fmul.d ft6, ft6, ft5\n"
            "fadd.d %[sum_0], ft3, %[sum_0] \n"
            "fadd.d %[sum_1], ft5, %[sum_1] \n"
            "fadd.d %[dotp_0], ft4, %[dotp_0]\n"
            "fadd.d %[dotp_1], ft6, %[dotp_1]\n"
            "j 0f\n"
            "1:\n"
            "fadd.d ft3, ft0, %[zero] \n"
            "fsub.d ft4, ft1, %[current_mean]\n"
            "fmul.d ft4, ft4, ft3\n"
            "fadd.d %[sum_0], ft3, %[sum_0] \n"
            "fadd.d %[dotp_0], ft4, %[dotp_0]\n"
            "0:\n"
            : [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1), [sum_2] "+fr"(sum_2),
              [dotp_0] "+fr"(dotp_0), [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2),
              [mod_temp] "=r"(mod_temp)
            : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
              [work_mod_3] "r"(work_mod_3),
              [n_frep] "r"(work_div_3_sub_1)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

        register double temp_sum, temp_dotp;
        asm volatile(
            "fld %[temp_sum], 0(%[sum_scratch])\n"
            "fld %[temp_dotp], 0(%[dotp_scratch])\n"
            "fadd.d %[sum_0], %[temp_sum], %[sum_0]\n"
            "fadd.d %[dotp_0], %[temp_dotp], %[dotp_0]\n"
            "fadd.d %[sum_0], %[sum_1], %[sum_0]\n"
            "fadd.d %[dotp_0], %[dotp_1], %[dotp_0]\n"
            "fsgnj.d %[sum_1],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[dotp_1],%[ZERO],%[ZERO]\n"
            "fadd.d %[sum_0], %[sum_2], %[sum_0]\n"
            "fadd.d %[dotp_0], %[dotp_2], %[dotp_0]\n"
            "fsgnj.d %[sum_2],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[dotp_2],%[ZERO],%[ZERO]\n"
            "fsd %[sum_0], 0(%[sum_scratch])\n"
            "fsd %[dotp_0], 0(%[dotp_scratch])\n"
            "fsgnj.d %[sum_0],%[ZERO],%[ZERO]\n"
            "fsgnj.d %[dotp_0],%[ZERO],%[ZERO]\n"
            : [temp_sum] "+fr"(temp_sum), [temp_dotp] "+fr"(temp_dotp),
              [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1), [sum_2] "+fr"(sum_2),
              [dotp_0] "+fr"(dotp_0), [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2)
            : [ZERO] "fr"(ZERO),
              [sum_scratch] "r"(sum_scratch), [dotp_scratch] "r"(dotp_scratch)
            : "ft0", "ft1", "ft2");
        snrt_fpu_fence();
        snrt_ssr_disable();
    } while (i < num_channels_to_process);
    __builtin_ssr_barrier(SNRT_SSR_DM1);
    DUMP(33);
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_tile_fp64_looped_1(
    const double* grad_ofmap_scratch, const double* ifmap_scratch, 
    const double* current_mean_scratch, double* sum_scratch, double* dotp_scratch,
    uint32_t C,
    uint32_t work_in_tile,  // requires: > 0
    uint32_t work_mod_3,    // precompute to avoid icache branch misses
    uint32_t work_div_3_sub_1, uint32_t tile_size_in_points,
    uint32_t num_channels_to_process,  //  requires: > 0
    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_3;
    register bool frep = work_in_tile >= 3;
    register double ZERO asm("ft11");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n" : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");
    bool buf_flag = 0;
    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_points * C * sizeof(double);
    const uint32_t channel_array_reset_dist = channel_stride * num_channels_to_process * sizeof(double);
    const uint32_t inner_loop_stride = C * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);
    snrt_ssr_loop_2d(SNRT_SSR_DM_ALL,
                     work_in_tile,             // dimension of inner loop
                     num_channels_to_process,  // dimension of outer loop
                     inner_loop_stride,        // stride per inner loop iteration: 1 point
                     outer_loop_stride);       // stride per outer loop iteration
    snrt_ssr_enable();
    // TODO: fix num_channels_work_for_core == 0.
    do {
        // DUMP(tile_size_in_points);
        // DUMP(C);
        // DUMP(sizeof(double));
        register volatile uint32_t i = 0;  // updated during frep for pseudo-dual issue
        register double sum_0 = ZERO;
        register double sum_1 = ZERO;
        register double sum_2 = ZERO;
        register double dotp_0 = ZERO;
        register double dotp_1 = ZERO;
        register double dotp_2 = ZERO;
        register double current_mean = *current_mean_scratch;
        snrt_cluster_hw_barrier();
        // DUMP(work_in_tile);
        // DUMP(1111);
        // DUMP(grad_ofmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, grad_ofmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, ifmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            // Can only manual unroll 5 times since the max for frep is 16
            if (frep) {
                asm volatile(
                    "frep.o %[n_frep], 15, 0, 0 \n"
                    "fadd.d ft3, ft0, %[zero] \n"
                    "fadd.d ft5, ft0, %[zero] \n"
                    "fadd.d ft7, ft0, %[zero] \n"
                    "fsub.d ft4, ft1, %[current_mean]\n"
                    "fsub.d ft6, ft1, %[current_mean]\n"
                    "fsub.d ft8, ft1, %[current_mean]\n"
                    "fmul.d ft4, ft4, ft3\n"
                    "fmul.d ft6, ft6, ft5\n"
                    "fmul.d ft8, ft8, ft7\n"
                    "fadd.d %[sum_0], ft3, %[sum_0] \n"
                    "fadd.d %[sum_1], ft5, %[sum_1] \n"
                    "fadd.d %[sum_2], ft7, %[sum_2] \n"
                    "fadd.d %[dotp_0], ft4, %[dotp_0]\n"
                    "fadd.d %[dotp_1], ft6, %[dotp_1]\n"
                    "fadd.d %[dotp_2], ft8, %[dotp_2]\n"
                    : [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1), [sum_2] "+fr"(sum_2),
                      [dotp_0] "+fr"(dotp_0), [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2)
                    : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
                      [n_frep] "r"(work_div_3_sub_1)  // we repeat n_frep+1 times
                    : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8");
            }

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "beqz %[i], 1f\n"
                "add %[sum_scratch], %[sum_scratch],%[channel_stride_in_bytes]\n"
                "add %[dotp_scratch], %[dotp_scratch], %[channel_stride_in_bytes]\n"
                "1:\n"
                "addi %[i], %[i], 1\n"
                "add %[current_mean_scratch], %[current_mean_scratch], %[channel_stride_in_bytes]\n"
                : [current_mean_scratch] "+r"(current_mean_scratch),
                  [sum_scratch] "+r"(sum_scratch),
                  [dotp_scratch] "+r"(dotp_scratch), [i] "+&r"(i),
                  [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                  [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                "sub %[current_mean_scratch], %[current_mean_scratch], %[channel_array_reset_dist]\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters
                                    // because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_3], %[work_mod_3_offset](%[dm_comm])\n"
                "lw %[work_div_3_sub_1], %[work_div_3_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 3\n"  // cmp frep < 3, then
                                                      // negate in next
                                                      // instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need
                                                            // to update ssr. If
                                                            // so, just update
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as
                                                      // prev_work instead of a
                                                      // temporary
                "4:\n"
                "beqz %[buf_flag], 3f\n"
                // buf_flag is 1, add to the scratches
                "add %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "add %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "j 2f\n"
                "3:\n"
                // buf_flag is 0, subtract back to original
                "sub %[grad_ofmap_scratch], %[grad_ofmap_scratch], %[buf_flag_offset]\n"
                "sub %[ifmap_scratch], %[ifmap_scratch], %[buf_flag_offset]\n"
                "2:\n"                    
                : [buf_flag] "+r"(buf_flag),
                  [current_mean_scratch] "+r"(current_mean_scratch),
                  [work_in_tile] "=r"(work_in_tile), [next_work_mod_3] "=r"(next_work_mod_3),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep), [work_div_3_sub_1] "=r"(work_div_3_sub_1),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch), [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i), [num_channels_to_process] "r"(num_channels_to_process),
                  [channel_array_reset_dist] "r"(channel_array_reset_dist),
                  [work_in_tile_offset] "i"(offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_3_offset] "i"(offsetof(dm_comm_t, work_mod_3)),
                  [work_div_3_sub_1_offset] "i"(offsetof(dm_comm_t, work_div_3_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS), [DM_ALL] "i"(SNRT_SSR_DM_ALL),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1), [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride), [dm_comm] "r"(dm_comm),
                  [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0");

            asm volatile(
                "beqz %[work_mod_3], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_3], 1\n"  // is last bit 1? if yes, then mod is 1
                "bnez %[mod_temp], 1f\n"                // jump to 1 if yes
                "2:\n"
                "fadd.d ft3, ft0, %[zero] \n"
                "fadd.d ft5, ft0, %[zero] \n"
                "fsub.d ft4, ft1, %[current_mean]\n"
                "fsub.d ft6, ft1, %[current_mean]\n"
                "fmul.d ft4, ft4, ft3\n"
                "fmul.d ft6, ft6, ft5\n"
                "fadd.d %[sum_0], ft3, %[sum_0] \n"
                "fadd.d %[sum_1], ft5, %[sum_1] \n"
                "fadd.d %[dotp_0], ft4, %[dotp_0]\n"
                "fadd.d %[dotp_1], ft6, %[dotp_1]\n"
                "j 0f\n"
                "1:\n"
                "fadd.d ft3, ft0, %[zero] \n"
                "fsub.d ft4, ft1, %[current_mean]\n"
                "fmul.d ft4, ft4, ft3\n"
                "fadd.d %[sum_0], ft3, %[sum_0] \n"
                "fadd.d %[dotp_0], ft4, %[dotp_0]\n"
                "0:\n"
                : [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1), [sum_2] "+fr"(sum_2),
                [dotp_0] "+fr"(dotp_0), [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2),
                [mod_temp] "=r"(temp)
                : [current_mean] "fr"(current_mean), [zero] "fr"(ZERO),
                [work_mod_3] "r"(work_mod_3),
                [n_frep] "r"(work_div_3_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

            register double temp_sum, temp_dotp;
            asm volatile(
                "fld %[temp_sum], 0(%[sum_scratch])\n"
                "fld %[temp_dotp], 0(%[dotp_scratch])\n"
                "fadd.d %[sum_0], %[temp_sum], %[sum_0]\n"
                "fadd.d %[dotp_0], %[temp_dotp], %[dotp_0]\n"
                // interleave 0 resetting and loading between fadd latency
                // don't need to synchronize here because the integer core can't
                // issue these instructions until the previous increments have
                // happened
                "fld %[current_mean],0(%[current_mean_scratch])\n"
                "fadd.d %[sum_0], %[sum_2], %[sum_0]\n"
                "fadd.d %[dotp_0], %[dotp_2], %[dotp_0]\n"
                "fsgnj.d %[sum_2],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[dotp_2],%[ZERO],%[ZERO]\n"
                "fadd.d %[sum_0], %[sum_1], %[sum_0]\n"
                "fadd.d %[dotp_0], %[dotp_1], %[dotp_0]\n"
                "fsgnj.d %[sum_1],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[dotp_1],%[ZERO],%[ZERO]\n"
                "fsd %[sum_0], 0(%[sum_scratch])\n"
                "fsd %[dotp_0], 0(%[dotp_scratch])\n"
                "fsgnj.d %[sum_0],%[ZERO],%[ZERO]\n"
                "fsgnj.d %[dotp_0],%[ZERO],%[ZERO]\n"
                : [temp_sum] "+fr"(temp_sum), [temp_dotp] "+fr"(temp_dotp),
                  [sum_0] "+fr"(sum_0), [sum_1] "+fr"(sum_1), [sum_2] "+fr"(sum_2),
                  [dotp_0] "+fr"(dotp_0), [dotp_1] "+fr"(dotp_1), [dotp_2] "+fr"(dotp_2),
                  [current_mean] "=fr"(current_mean)
                : [ZERO] "fr"(ZERO),
                  [current_mean_scratch] "r"(current_mean_scratch),
                  [sum_scratch] "r"(sum_scratch),
                  [dotp_scratch] "r"(dotp_scratch)
                : "ft0", "ft1", "ft2");
        } while (i < num_channels_to_process);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        // __builtin_ssr_barrier(SNRT_SSR_DM1);
        work_mod_3 = next_work_mod_3;
        sum_scratch -= channel_stride * (num_channels_to_process - 1);
        dotp_scratch -= channel_stride * (num_channels_to_process - 1);
    } while (work_in_tile != 0);
    // notify last tile done
    snrt_ssr_disable();
    snrt_cluster_hw_barrier();
}

static inline void batchnorm_backward_training_main_loop_1(uint32_t C, uint32_t work_left,  // only present for dma
                                                uint32_t initial_work_in_tile, uint32_t initial_work_mod_3,
                                                uint32_t initial_work_div_3_sub_1, dm_comm_t* dm_comm,
                                                uint32_t tile_size_in_points, uint32_t compute_id,
                                                uint32_t num_compute_cores, batchnorm_backward_training_layer_t* l,
                                                double* grad_ofmap_scratch, double* ifmap_scratch, double* current_mean_scratch,
                                                double* sum_scratch, double* dotp_scratch, bool buf_flag) {
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();

    uint32_t num_channels_work_for_core = get_core_num_work_items(C, num_compute_cores, compute_id);
    if (snrt_is_dm_core()) {
        snrt_dma_wait_all();
        // buf_flag should be 1 here.
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
            work_left -= work_in_tile;
            // update comms
            dm_comm->num_points_work_in_tile = work_in_tile;
            dm_comm->work_mod_3 = work_in_tile % 3;
            dm_comm->work_div_3_sub_1 = work_in_tile / 3 - 1;
            // comm what the next iteration will be
            // wait for potential previous gradifmap write out?
            snrt_dma_wait_all();
            snrt_dma_start_1d(&grad_ofmap_scratch[tile_size_in_points * C * buf_flag], &l->grad_ofmap[point_start * C],
                              work_in_tile * C * sizeof(double));
            snrt_dma_start_1d(&ifmap_scratch[tile_size_in_points * C * buf_flag], &l->ifmap[point_start * C],
                              work_in_tile * C * sizeof(double));
            snrt_cluster_hw_barrier();
            snrt_dma_wait_all();
            // signal to core that current tile is ready to be computed on
            snrt_cluster_hw_barrier();
            prev_point_start = point_start;
            num_points_work_in_prev_tile = work_in_tile;
            point_start += work_in_tile;
            buf_flag = !buf_flag;
        }
        dm_comm->num_points_work_in_tile = 0;
        dm_comm->work_mod_3 = 0;
        dm_comm->work_div_3_sub_1 = 0xdeadbeef;
        // signal last iteration that there is no more work
        snrt_cluster_hw_barrier();
        // wait for last tile to finish
        snrt_cluster_hw_barrier();
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
            batchnorm_backward_training_tile_fp64_looped_1(
                &grad_ofmap_scratch[compute_id], &ifmap_scratch[compute_id], &current_mean_scratch[compute_id],
                &sum_scratch[compute_id], &dotp_scratch[compute_id], C,
                initial_work_in_tile, initial_work_mod_3, initial_work_div_3_sub_1, tile_size_in_points,
                num_channels_work_for_core, num_compute_cores, dm_comm);
        }
    }
    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_tile_fp64_no_loop_2(
    const double* grad_ofmap_scratch, double* grad_ifmap_scratch, const double* ifmap_scratch,
    const double* current_mean_scratch, const double* weight_scratch, const double* invstd_scratch,
    const double* k_scratch, const double* grad_mean_scratch, uint32_t C,
    uint32_t num_points_work_for_core_in_tile, uint32_t work_mod_4,
    uint32_t work_div_4_sub_1, uint32_t num_channels_to_process,
    uint32_t channel_stride, bool is_first_iteration, bool force_configure) {
    
    if (is_first_iteration || force_configure) {
        snrt_ssr_loop_2d(
            SNRT_SSR_DM_ALL,
            num_points_work_for_core_in_tile,  // dimension of inner loop
            num_channels_to_process,           // dimension of outer loop
            C * sizeof(double),  // stride per inner loop iteration: 1 point
            channel_stride *
                sizeof(double));  // stride per outer loop iteration
    }
    bool frep = num_points_work_for_core_in_tile >= 4;
    register volatile uint32_t i = 0;
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, grad_ofmap_scratch);
    do {
        register double current_mean = *current_mean_scratch;
        register double k = *k_scratch;
        register double weight_times_invstd = *weight_scratch * *invstd_scratch;
        register double grad_mean_times_weight_times_invstd = *grad_mean_scratch * weight_times_invstd;
        snrt_ssr_enable();
        if (frep) {
            asm volatile(
                "frep.o %[n_frep], 12, 0, 0 \n"
                "fsub.d ft3, ft0, %[current_mean] \n"
                "fsub.d ft4, ft0, %[current_mean] \n"
                "fsub.d ft5, ft0, %[current_mean] \n"
                "fsub.d ft6, ft0, %[current_mean] \n"
                "fnmsub.d ft3, ft3, %[k], ft2\n"
                "fnmsub.d ft4, ft4, %[k], ft2\n"
                "fnmsub.d ft5, ft5, %[k], ft2\n"
                "fnmsub.d ft6, ft6, %[k], ft2\n"
                "fmsub.d ft1, ft3, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft4, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft5, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft6, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                :
                : [current_mean] "fr"(current_mean), [k] "fr"(k),
                [grad_mean_times_weight_times_invstd] "fr"(grad_mean_times_weight_times_invstd),
                [weight_times_invstd] "fr"(weight_times_invstd),
                [n_frep] "r"(work_div_4_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");
        }

        register uint32_t channel_stride_in_bytes;
        asm volatile(
            "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
            "addi %[i], %[i], 1\n"
            "beq %[num_channels_to_process], %[i], 2f\n"  // shortcut when only 1 channel
            "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
            "add %[weight_scratch], %[weight_scratch], %[channel_stride_in_bytes]\n"
            "add %[current_mean_scratch], %[current_mean_scratch], %[channel_stride_in_bytes]\n"
            "add %[k_scratch], %[k_scratch], %[channel_stride_in_bytes]\n"
            "add %[grad_mean_scratch], %[grad_mean_scratch], %[channel_stride_in_bytes]\n"
            "2:\n"
            : [invstd_scratch] "+r"(invstd_scratch), [weight_scratch] "+r"(weight_scratch),
            [current_mean_scratch] "+r"(current_mean_scratch), [k_scratch] "+r"(k_scratch),
            [grad_mean_scratch] "+r"(grad_mean_scratch), [i] "+r"(i),
            [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
            : [channel_stride] "r"(channel_stride),
            [num_channels_to_process] "r"(num_channels_to_process)
            : "ft0", "ft1", "ft2");
        
        register uint32_t mod_temp;
        asm volatile(
            "beqz %[work_mod_4], 0f\n"              // mod is 0
            "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no, then mod is 2
            "beqz %[mod_temp], 2f\n"                // jump to 2 if no
            "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no, then mod is 1
            "beqz %[mod_temp], 1f\n"                // jump to 1 if no
            "3:\n"
            "fsub.d ft3, ft0, %[current_mean] \n"
            "fsub.d ft4, ft0, %[current_mean] \n"
            "fsub.d ft5, ft0, %[current_mean] \n"
            "fnmsub.d ft3, ft3, %[k], ft2\n"
            "fnmsub.d ft4, ft4, %[k], ft2\n"
            "fnmsub.d ft5, ft5, %[k], ft2\n"
            "fmsub.d ft1, ft3, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
            "fmsub.d ft1, ft4, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
            "fmsub.d ft1, ft5, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
            "j 0f\n"
            "2:\n"
            "fsub.d ft3, ft0, %[current_mean] \n"
            "fsub.d ft4, ft0, %[current_mean] \n"
            "fnmsub.d ft3, ft3, %[k], ft2\n"
            "fnmsub.d ft4, ft4, %[k], ft2\n"
            "fmsub.d ft1, ft3, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
            "fmsub.d ft1, ft4, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
            "j 0f\n"
            "1:\n"
            "fsub.d ft3, ft0, %[current_mean] \n"
            "fnmsub.d ft3, ft3, %[k], ft2\n"
            "fmsub.d ft1, ft3, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
            "0:\n"
            : [mod_temp] "=r"(mod_temp)
            : [current_mean] "fr"(current_mean), [k] "fr"(k),
                [grad_mean_times_weight_times_invstd] "fr"(grad_mean_times_weight_times_invstd),
                [weight_times_invstd] "fr"(weight_times_invstd),
                [work_mod_4] "r"(work_mod_4)
            : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");
        snrt_fpu_fence();
        snrt_ssr_disable();
    } while (i < num_channels_to_process);
    __builtin_ssr_barrier(SNRT_SSR_DM1);
}

static inline void __attribute__((always_inline))
batchnorm_backward_training_tile_fp64_looped_2(const double* grad_ofmap_scratch,
                                    double* grad_ifmap_scratch,  // no restrict because grad_ifmap and ifmap used
                                    const double* ifmap_scratch, const double* current_mean_scratch,
                                    const double* weight_scratch, const double* invstd_scratch,
                                    const double* k_scratch, const double* grad_mean_scratch, 
                                    uint32_t C,
                                    uint32_t work_in_tile,  // requires: > 0
                                    uint32_t work_mod_4,    // precompute to avoid icache branch misses
                                    uint32_t work_div_4_sub_1, uint32_t tile_size_in_points,
                                    uint32_t num_channels_to_process,  //  requires: > 0
                                    uint32_t channel_stride, dm_comm_t* dm_comm) {
    // access pattern: iterate over the different channels, then over
    // the different points
    // Split work over channels to maximize efficacy of frep.
    // outside loop: channels
    // inside loop: points
    uint32_t prev_work = work_in_tile;
    register uint32_t next_work_mod_4;
    register bool frep = work_in_tile >= 4;
    register double ZERO asm("ft11");  // can consider fcvt instead
    asm volatile("fcvt.d.w %[ZERO], zero\n" : [ZERO] "=r"(ZERO)::"ft0", "ft1", "ft2");

    bool buf_flag = 0;
    // consider: inlining these as well later
    const uint32_t buf_flag_offset = tile_size_in_points * C * sizeof(double);
    const uint32_t channel_array_reset_dist = channel_stride * num_channels_to_process * sizeof(double);
    const uint32_t inner_loop_stride = C * sizeof(double);
    const uint32_t outer_loop_stride = channel_stride * sizeof(double);
    snrt_ssr_loop_2d(SNRT_SSR_DM_ALL,
                     work_in_tile,             // dimension of inner loop
                     num_channels_to_process,  // dimension of outer loop
                     inner_loop_stride,        // stride per inner loop iteration: 1 point
                     outer_loop_stride);       // stride per outer loop iteration
    // TODO: fix num_channels_work_for_core == 0.
    do {
        register volatile uint32_t i = 0;  // updated during frep for pseudo-dual issue
        snrt_cluster_hw_barrier();
        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, ifmap_scratch);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, grad_ifmap_scratch);
        snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, grad_ofmap_scratch);
        // do 1 loop
        do {  // while (i < num_channels_to_process)
            // Can only manual unroll 5 times since the max for frep is 16
            register double current_mean = *current_mean_scratch;
            register double k = *k_scratch;
            register double weight_times_invstd = *weight_scratch * *invstd_scratch;
            register double grad_mean_times_weight_times_invstd = *grad_mean_scratch * weight_times_invstd;
            snrt_ssr_enable();
            asm volatile(
                "frep.o %[n_frep], 12, 0, 0 \n"
                "fsub.d ft3, ft0, %[current_mean] \n"
                "fsub.d ft4, ft0, %[current_mean] \n"
                "fsub.d ft5, ft0, %[current_mean] \n"
                "fsub.d ft6, ft0, %[current_mean] \n"
                "fnmsub.d ft3, ft3, %[k], ft2\n"
                "fnmsub.d ft4, ft4, %[k], ft2\n"
                "fnmsub.d ft5, ft5, %[k], ft2\n"
                "fnmsub.d ft6, ft6, %[k], ft2\n"
                "fmsub.d ft1, ft3, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft4, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft5, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft6, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                :
                : [current_mean] "fr"(current_mean), [k] "fr"(k),
                [grad_mean_times_weight_times_invstd] "fr"(grad_mean_times_weight_times_invstd),
                [weight_times_invstd] "fr"(weight_times_invstd),
                [n_frep] "r"(work_div_4_sub_1)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6");

            register uint32_t channel_stride_in_bytes;
            asm volatile(
                "slli %[channel_stride_in_bytes], %[channel_stride], 3\n"  // log_2(sizeof(double))
                "addi %[i], %[i], 1\n"
                "add %[invstd_scratch], %[invstd_scratch], %[channel_stride_in_bytes]\n"
                "add %[weight_scratch], %[weight_scratch], %[channel_stride_in_bytes]\n"
                "add %[current_mean_scratch], %[current_mean_scratch], %[channel_stride_in_bytes]\n"
                "add %[k_scratch], %[k_scratch], %[channel_stride_in_bytes]\n"
                "add %[grad_mean_scratch], %[grad_mean_scratch], %[channel_stride_in_bytes]\n"
                "2:\n"
                : [invstd_scratch] "+r"(invstd_scratch), [weight_scratch] "+r"(weight_scratch),
                [current_mean_scratch] "+r"(current_mean_scratch), [k_scratch] "+r"(k_scratch),
                [grad_mean_scratch] "+r"(grad_mean_scratch), [i] "+r"(i),
                [channel_stride_in_bytes] "=r"(channel_stride_in_bytes)
                : [channel_stride] "r"(channel_stride),
                [num_channels_to_process] "r"(num_channels_to_process)
                : "ft0", "ft1", "ft2");

            register uint32_t temp;
            asm volatile(
                "bne %[i], %[num_channels_to_process], 2f\n"
                // extra check here for channels == 1. THen don't sub
                "sub %[invstd_scratch], %[invstd_scratch], %[channel_array_reset_dist]\n"
                "sub %[weight_scratch], %[weight_scratch], %[channel_array_reset_dist]\n"
                "sub %[current_mean_scratch], %[current_mean_scratch], %[channel_array_reset_dist]\n"
                "sub %[k_scratch], %[k_scratch], %[channel_array_reset_dist]\n"
                "sub %[grad_mean_scratch], %[grad_mean_scratch], %[channel_array_reset_dist]\n"
                "xori %[buf_flag], %[buf_flag], 1\n"
                "csrr x0, 0x7C2\n"  // wait for dma to compute parameters because I don't want to do math here
                "lw %[work_in_tile], %[work_in_tile_offset](%[dm_comm])\n"
                "lw %[next_work_mod_4], %[work_mod_4_offset](%[dm_comm])\n"
                "lw %[work_div_4_sub_1], %[work_div_4_sub_1_offset](%[dm_comm])\n"
                "slti %[frep], %[work_in_tile], 4\n"  // cmp frep < 4, then negate in next instruction
                "xori %[frep], %[frep], 1\n"
                "beq %[work_in_tile], %[prev_work], 4f\n"   // check if we need to update ssr. If so, just update the
                                                            // bounds
                "addi %[prev_work], %[work_in_tile], -1\n"  // a = --b0
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_BOUNDS_PLUS_0]<<5\n"  // write_ssr_config
                "mul %[prev_work], %[prev_work], %[inner_loop_stride]\n"
                "sub %[prev_work], %[outer_loop_stride], %[prev_work]\n"
                "scfgwi %[prev_work], %[DM_ALL] | %[REG_STRIDES_PLUS_1]<<5\n"
                // first stride still the same
                // a = b0 * s0
                // a = s1 - a
                // scfgwi %[REG_STRIDES_PLUS_1], %[DM_ALL] | %[a]<<5\n
                "mv %[prev_work], %[work_in_tile]\n"  // now use prev_work as prev_work instead of a temporary
                "4:\n"
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
                : [buf_flag] "+r"(buf_flag), [invstd_scratch] "+r"(invstd_scratch), [weight_scratch] "+r"(weight_scratch),
                  [current_mean_scratch] "+r"(current_mean_scratch), [k_scratch] "+r"(k_scratch), [grad_mean_scratch] "+r"(grad_mean_scratch),
                  [work_in_tile] "=r"(work_in_tile), [next_work_mod_4] "=r"(next_work_mod_4),
                  [prev_work] "+r"(prev_work), [frep] "+r"(frep), [work_div_4_sub_1] "=r"(work_div_4_sub_1),
                  [grad_ofmap_scratch] "+r"(grad_ofmap_scratch), [grad_ifmap_scratch] "+r"(grad_ifmap_scratch), [ifmap_scratch] "+r"(ifmap_scratch)
                : [i] "r"(i), [num_channels_to_process] "r"(num_channels_to_process),
                  [channel_array_reset_dist] "r"(channel_array_reset_dist),
                  [work_in_tile_offset] "i"(offsetof(dm_comm_t, num_points_work_in_tile)),
                  [work_mod_4_offset] "i"(offsetof(dm_comm_t, work_mod_4)),
                  [work_div_4_sub_1_offset] "i"(offsetof(dm_comm_t, work_div_4_sub_1)),
                  [REG_BOUNDS_PLUS_0] "i"(REG_BOUNDS), [DM_ALL] "i"(SNRT_SSR_DM_ALL),
                  [REG_STRIDES_PLUS_1] "i"(REG_STRIDES + 1), [inner_loop_stride] "r"(inner_loop_stride),
                  [outer_loop_stride] "r"(outer_loop_stride), [dm_comm] "r"(dm_comm),
                  [buf_flag_offset] "r"(buf_flag_offset)
                : "ft0", "ft1", "ft2", "x0", "memory");

            asm volatile(
                "beqz %[work_mod_4], 0f\n"              // mod is 0
                "andi %[mod_temp], %[work_mod_4], 1\n"  // is last bit 1? if no, then mod is 2
                "beqz %[mod_temp], 2f\n"                // jump to 2 if no
                "andi %[mod_temp], %[work_mod_4], 2\n"  // is last bit 1? if no, then mod is 1
                "beqz %[mod_temp], 1f\n"                // jump to 1 if no
                "3:\n"
                "fsub.d ft3, ft0, %[current_mean] \n"
                "fsub.d ft4, ft0, %[current_mean] \n"
                "fsub.d ft5, ft0, %[current_mean] \n"
                "fnmsub.d ft3, ft3, %[k], ft2\n"
                "fnmsub.d ft4, ft4, %[k], ft2\n"
                "fnmsub.d ft5, ft5, %[k], ft2\n"
                "fmsub.d ft1, ft3, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft4, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft5, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "j 0f\n"
                "2:\n"
                "fsub.d ft3, ft0, %[current_mean] \n"
                "fsub.d ft4, ft0, %[current_mean] \n"
                "fnmsub.d ft3, ft3, %[k], ft2\n"
                "fnmsub.d ft4, ft4, %[k], ft2\n"
                "fmsub.d ft1, ft3, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "fmsub.d ft1, ft4, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "j 0f\n"
                "1:\n"
                "fsub.d ft3, ft0, %[current_mean] \n"
                "fnmsub.d ft3, ft3, %[k], ft2\n"
                "fmsub.d ft1, ft3, %[weight_times_invstd], %[grad_mean_times_weight_times_invstd] \n"
                "0:\n"
                : [mod_temp] "=r"(temp)
                : [current_mean] "fr"(current_mean), [k] "fr"(k),
                    [grad_mean_times_weight_times_invstd] "fr"(grad_mean_times_weight_times_invstd),
                    [weight_times_invstd] "fr"(weight_times_invstd),
                    [work_mod_4] "r"(work_mod_4)
                : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5");
            snrt_fpu_fence();
            snrt_ssr_disable();
        } while (i < num_channels_to_process);
        // don't need to fpu_fence since last 3 instructions are inconsequential
        __builtin_ssr_barrier(SNRT_SSR_DM1);
        work_mod_4 = next_work_mod_4;
    } while (work_in_tile != 0);
    // notify last tile done
    snrt_cluster_hw_barrier();
}

static inline void batchnorm_backward_training_main_loop_2(uint32_t C, uint32_t work_left,  // only present for dma
                                                uint32_t initial_work_in_tile, uint32_t initial_work_mod_4,
                                                uint32_t initial_work_div_4_sub_1, dm_comm_t* dm_comm,
                                                uint32_t tile_size_in_points, uint32_t compute_id,
                                                uint32_t num_compute_cores, batchnorm_backward_training_layer_t* l,
                                                double* grad_ofmap_scratch, double* ifmap_scratch, double* grad_ifmap_scratch,
                                                double* k_scratch, double* grad_mean_scratch, double* invstd_scratch,
                                                double* current_mean_scratch, double* weight_scratch, bool buf_flag) {
    uint32_t start_main_loop = SNRT_SECTIONED_MCYCLE();

    uint32_t num_channels_work_for_core = get_core_num_work_items(C, num_compute_cores, compute_id);

    if (snrt_is_dm_core()) {
        snrt_dma_wait_all();
        // buf_flag should be 1 here.
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
        // align up to multiple of 4 because that avoids stalling in fpu the best
        uint32_t ideal_work_in_tile =
            min(align_up_non_power_of_2(ceildiv(work_left, min_loops), 4), tile_size_in_points);
        // uint32_t ideal_work_in_tile = 96;  // TODO CHANGE BACK
        while (work_left > 0) {
            // uint32_t estimated_max_tileable_work = tile_size_in_points;
            // (work_in_tile * ceildiv(C, num_compute_cores) * 5 *
            //  NUM_DOUBLES_LOADED_PER_CYCLE) /
            // (3 * C);
            work_in_tile = min(ideal_work_in_tile, work_left);
            work_left -= work_in_tile;
            // update comms
            dm_comm->num_points_work_in_tile = work_in_tile;
            dm_comm->work_mod_4 = work_in_tile % 4;
            dm_comm->work_div_4_sub_1 = work_in_tile / 4 - 1;
            // comm what the next iteration will be
            // wait for potential previous gradifmap write out?
            snrt_dma_wait_all();
            snrt_dma_start_1d(&grad_ofmap_scratch[tile_size_in_points * C * buf_flag], &l->grad_ofmap[point_start * C],
                              work_in_tile * C * sizeof(double));
            snrt_dma_start_1d(&ifmap_scratch[tile_size_in_points * C * buf_flag], &l->ifmap[point_start * C],
                              work_in_tile * C * sizeof(double));
            snrt_cluster_hw_barrier();
            snrt_dma_wait_all();
            // signal to core that current tile is ready to be computed on
            snrt_cluster_hw_barrier();
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
        dm_comm->work_mod_4 = 0;
        dm_comm->work_div_4_sub_1 = 0xdeadbeef;
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
            batchnorm_backward_training_tile_fp64_looped_2(
                &grad_ofmap_scratch[compute_id], &grad_ifmap_scratch[compute_id], &ifmap_scratch[compute_id], &current_mean_scratch[compute_id],
                &weight_scratch[compute_id], &invstd_scratch[compute_id], &k_scratch[compute_id], &grad_mean_scratch[compute_id], C,
                initial_work_in_tile, initial_work_mod_4, initial_work_div_4_sub_1, tile_size_in_points,
                num_channels_work_for_core, num_compute_cores, dm_comm);
        }
    }

    uint32_t end_main_loop = SNRT_SECTIONED_MCYCLE();
}