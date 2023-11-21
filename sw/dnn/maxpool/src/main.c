// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SW testbench for profiling MaxPool Layer
// Automatically checks the correctness of the results

#include "dnn.h"

#include "benchmark_2.5.h"

// #include "printf.h"

int main() {
    maxpool_layer(&layer);

    snrt_global_barrier();

    // if (snrt_global_core_idx() == 1) for (int i = 0; i < 81; ++i) printf("%lf\n", ofmap[i]);

    return 0;
}
