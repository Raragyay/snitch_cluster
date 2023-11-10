// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"

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
        // perform transformation for each pixel, but only for channels [ci,
        // ci+l->tile_ci)
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

static inline void my_batchnorm(const batchnorm_layer_t *layer) {
    uint32_t N = 1, H = layer->IH, W = layer->IW, C = layer->CI;
    double *X = layer->ifmap;
    double *Y = layer->ofmap;
    double *gamma = layer->gamma;
    double *beta = layer->beta;
    if (!snrt_is_compute_core()) {
        return;
    }
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster
    // if (training == 1) {
    //     printf("training mode not supported\n");
    //     return;
    // }

    uint32_t num_points = N * H * W;
    // [1_R,1_G,1_B,2_R,2_G,2_B]
    for (uint32_t i = compute_id; i < num_points; i += num_compute_cores) {
        // in vector notation
        // Y[i * H * W * C + h * W * C + w * C] =
        //     (X[i * H * W * C + h * W * C + w * C] - input_mean) /
        //         sqrt(input_var + eps) * weight + bias
        for (uint32_t channel = 0; channel < C; ++channel) {
            Y[i * C + channel] =
                X[i * C + channel] * gamma[channel] + beta[channel];
            // Y[i * C + channel] = (X[i * C + channel] - input_mean[channel]) /
            //                          sqrt(input_var[channel] + eps) *
            //                          weight[channel] +
            //                      bias[channel];
        }
    }
    // conv layer in NHWC format
    // but conv_layer only has 1 batch

    // compute mean and variance

    // batch_norm_cpu_collect_stats_channels_last_impl
    // batch_norm_cpu_collect_linear_and_constant_terms

    // without beta or gamma passed in

    // if training is 0 (then we don't need to calculate the mean and variance
    // of X)
}
static inline void batchnorm_training(batchnorm_training_layer_t *layer) {
    // collect stats
    // update running mean and running var
    // pass << batch mean and batch var >> as parameters to compute beta/gamma
    // call batchnorm_layer
}

static inline void batchnorm_backwards(batchnorm_backward_layer_t *l) {
    // data is in HWC format
    const uint32_t num_clusters =
        snrt_cluster_num();  // how many clusters are there in total? currently
                             // 1 in the config i think
    const uint32_t cluster_id = snrt_cluster_idx();  // which cluster are we?
    const uint32_t num_compute_cores =
        snrt_cluster_compute_core_num();  // how many compute cores per cluster?
    const uint32_t compute_id =
        snrt_cluster_core_idx();  // which core are we in this cluster
    if (compute_id != 0) {
        return;
    }
    // Calculate output dimensions
    uint32_t N = 1;
    uint32_t H = l->IH;
    uint32_t W = l->IW;
    uint32_t C = l->CI;
    double eps = l->eps;

    uint32_t buffer_len = l->CI;
    uint32_t invstd_scratch_len = l->CI;
    uint32_t grad_bias_scratch_len = l->CI * num_compute_cores,
             grad_weight_scratch_len = l->CI * num_compute_cores;

    double *ptr = (double *)snrt_l1_start_addr();
    double *invstd_scratch = ptr;
    ptr += invstd_scratch_len;
    double *grad_bias_scratch = ptr;
    ptr += grad_bias_scratch_len;
    double *grad_weight_scratch = ptr;
    ptr += grad_weight_scratch_len;

    // double *gamma = ptr;
    // ptr += weights_size;
    // double *beta = ptr;
    // ptr += weights_size;
    // double *ofmap = ptr;
    // ptr += ofmap_size;

    // snrt_cluster_hw_barrier();
    // compute invstd. just single thread for now
    if (compute_id == 0) {
        #if 0
        // snrt_ssr_loop_1d(SNRT_SSR_DM0, C, sizeof(double));
        // snrt_ssr_loop_1d(SNRT_SSR_DM1, C, sizeof(double));

        // snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, l->running_var);
        // snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, invstd_scratch);
        // snrt_ssr_enable();
        // double eps = l->eps;
        
        // double one = 1;

        // // frep over C dimension
        // asm volatile(
        //     "frep.o %[n_frep], 3, 0, 0 \n"
        //     "fadd.d ft2, ft0, %[eps]\n"
        //     "fsqrt.d ft2, ft2\n"
        //     "fdiv.d ft1, %[one], ft2\n"
        //     :
        //     : [eps] "f"(eps), [one] "f"(one),
        //       [n_frep] "r"(C - 1)  // we repeat n_frep+1 times
        //     : "memory", "ft0", "ft1", "ft2");
        // snrt_fpu_fence();
        // // wait for writes to the ofmap to finish?
        // __builtin_ssr_barrier(SNRT_SSR_DM1);
        // snrt_ssr_disable();
        #endif

        for (uint32_t channel = 0; channel < C;++channel){
            invstd_scratch[channel] = 1/sqrt(l->running_var[channel]+eps);
        }
    }
    // snrt_cluster_hw_barrier();
    // reduce over [num_points, C] to [num_threads, C] by splitting over
    // num_points
    // for (uint32_t i = compute_id; i < num_points; i += compute_id) {
    // }

    // reduce over [num_threads, C] to [C] by splitting over C

    uint32_t num_points = N * H * W;
    // [1_R,1_G,1_B,2_R,2_G,2_B
    if (!snrt_is_compute_core()) {
        return;
    }
    for (uint32_t i = 0; i < num_points; i += 1) {
        // in vector notation
        for (uint32_t channel = 0; channel < C; ++channel) {
            double dy = l->grad_ofmap[i * C + channel];
            double x = l->ifmap[i * C + channel];
            double mean = l->running_mean[channel];
            double dot_res = dy * (x - mean);
            // printf("core %d, pixel %d, channel %d, adding %f to %f\n",
            //        compute_id, i, channel, dy, l->grad_bias[channel]);
            l->grad_bias[channel] += dy;
            l->grad_weight[channel] +=
                dot_res * invstd_scratch[channel];
            l->grad_ifmap[i * C + channel] =
                dy * invstd_scratch[channel] *
                l->weight[channel];
            // Y[i * C + channel] = (X[i * C + channel] - input_mean[channel]) /
            //                          sqrt(input_var[channel] + eps) *
            //                          weight[channel] +
            //                      bias[channel];
        }
    }
    // allocate space for a grad_bias
    // allocate space for a grad_weight (dotp)
    // allocate space for computing the 1/sqrt(running_var[c]+eps)*weight[c]

    // grad_bias[c] = sum(dy[i][c] for all pixels i)
    // grad_weight[c] = sum(dy[i][c]*(ifmap[i][c]-running_mean[c]) for all
    // pixels i)

    // in training mode: big equation
    // in eval mode: grad_ifmap[i][c] =
    // dy[i][c]*1/sqrt(running_var[c]+eps)*weight[c]
}