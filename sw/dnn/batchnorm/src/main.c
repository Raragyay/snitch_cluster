// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "batchnorm_impl_params.h"
#include "dnn.h"

#include "data.h"

int main() {
    switch (impl_opt_level) {
        case SINGLE_CORE:
            batchnorm_backward_single_core(&backward_layer);
            break;
        case SINGLE_CORE_OPT:
            batchnorm_backward_single_core_opt(&backward_layer);
            break;
        case MULTICORE_OPT:
            batchnorm_backward(&backward_layer);
            break;
        default:
            return 1;
    }

    snrt_global_barrier();

    return 0;
}
