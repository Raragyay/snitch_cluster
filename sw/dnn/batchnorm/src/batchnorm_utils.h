#pragma once

#include "snrt.h"

#define PERF_DEBUG 0
#define min(a, b) ((a) < (b) ? (a) : (b))
#define ceildiv(a, b) ((((a) - 1) / (b)) + 1)
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
