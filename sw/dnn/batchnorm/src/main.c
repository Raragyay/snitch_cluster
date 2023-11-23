// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "batchnorm_impl_params.h"
#include "dnn.h"

#include "data.h"

int main() {
    if (is_forward && is_training) {
        printf("Forward eval training not supported yet");
        return 1;
    } else if (is_forward && !is_training) {
        batchnorm_layer(&forward_eval_layer);
    } else if (!is_forward && is_training) {
        batchnorm_backward_training(&backward_training_layer);
    } else {
        switch (impl_opt_level) {
            case SINGLE_CORE:
                batchnorm_backward_single_core(&backward_eval_layer);
                break;
            case SINGLE_CORE_OPT:
                batchnorm_backward_single_core_opt(&backward_eval_layer);
                break;
            case MULTICORE_OPT:
                batchnorm_backward(&backward_eval_layer);
                break;
            default:
                return 1;
        }
    }

    snrt_global_barrier();

    return 0;
}
