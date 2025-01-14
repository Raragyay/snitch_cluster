// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "blas.h"

/**
 * @struct gemm_layer_struct
 * @brief This structure contains all parameters necessary for GEMM.
 * @var gemm_layer_struct::M
 * Dimension of matrix product MxK * KxN
 * @var gemm_layer_struct::M_p
 * M divided by number of compute cores
 * @var gemm_layer_struct::N
 * Dimension of matrix product MxK * KxN
 * @var gemm_layer_struct::K
 * Dimension of matrix product MxK * KxN
 * @var gemm_layer_struct::TA
 * Transpose matrix A
 * @var gemm_layer_struct::TB
 * Transpose matrix B
 * @var gemm_layer_struct::TILE_M
 * Tile factor across M dimension
 * @var gemm_layer_struct::TILE_N
 * Tile factor across N dimension
 * @var gemm_layer_struct::TILE_K
 * Tile factor across K dimension
 * @var gemm_layer_struct::A
 * Pointer to matrix A
 * @var gemm_layer_struct::B
 * Pointer to matrix B
 * @var gemm_layer_struct::C
 * Pointer to matrix C
 * @var gemm_layer_struct::ALPHA
 * constant factor: A * B + ALPHA * C
 * @var gemm_layer_struct::dtype
 * Precision of GEMM
 * @var gemm_layer_struct::expand
 * Use expanding DOTP instructions
 */
typedef struct gemm_layer_struct {
    uint32_t M;
    uint32_t M_p;
    uint32_t N;
    uint32_t K;

    uint32_t TA;
    uint32_t TB;

    uint32_t TILE_M;
    uint32_t TILE_N;
    uint32_t TILE_K;

    void *A;
    void *B;
    void *C;

    uint32_t ALPHA;

    precision_t dtype;
    uint32_t expand;
} gemm_layer_t;
