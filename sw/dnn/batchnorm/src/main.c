// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "dnn.h"

#include "data.h"

int main() {
    batchnorm_backward_training(&backward_training_layer);

    snrt_global_barrier();

    return 0;
}
