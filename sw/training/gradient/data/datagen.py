#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Authors: Tim Fischer     <fischeti@iis.ee.ethz.ch>
#          Luca Bertaccini <lbertaccini@iis.ee.ethz.ch>

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import torch
from torch import nn
import torch.optim as optim
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
from data_utils import emit_license, format_scalar_definition, \
    format_array_definition, format_ifdef_wrapper  # noqa: E402

np.random.seed(42)

C_TYPES = {
    '64': 'double',
    '32': 'float',
    '16': '__fp16',
    '8': 'char'
}

NUMPY_TYPES = {
    '64': np.double,
    '32': np.single,
    '16': np.half,
    '8': np.ubyte
}

FP8_FORMATS = {
    'fp8': {'exp': 5, 'mant': 2},
    'fp8alt': {'exp': 4, 'mant': 3}
}

# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096


def golden_model(alpha,A,B,GRAD_C):
    return alpha*np.matmul( GRAD_C,np.transpose(B)), alpha*np.matmul(np.transpose(A), GRAD_C)


def emit_header(**kwargs):
    # Generate random input matrices
    dtype = NUMPY_TYPES[str(kwargs['prec'])]

    alpha = np.random.rand((1)).astype(dtype) [0]
    A = np.random.rand(kwargs['M'], kwargs['K']).astype(dtype)
    B = np.random.rand(kwargs['K'], kwargs['N']).astype(dtype)
    GRAD_C = np.random.rand(kwargs['M'], kwargs['N']).astype(dtype)
    OLD_GRAD_A = np.random.rand(kwargs['M'], kwargs['K']).astype(dtype)
    OLD_GRAD_B = np.random.rand(kwargs['K'], kwargs['N']).astype(dtype)

    GRAD_A, GRAD_B = golden_model(alpha,A,B,GRAD_C)

    data_str = [emit_license()]
    data_str += [format_scalar_definition('uint32_t', 'M', kwargs['M'])]
    data_str += [format_scalar_definition('uint32_t', 'N', kwargs['N'])]
    data_str += [format_scalar_definition('uint32_t', 'K', kwargs['K'])]
    # divisors = [i for i in range(1, 9) if kwargs['M'] % i == 0]
    # M_tiles = random.choice(divisors)
    # divisors = [i for i in range(1, 9) if kwargs['N'] % i == 0]
    # N_tiles = random.choice(divisors)
    # divisors = [i for i in range(1, 9) if kwargs['K'] % i == 0]
    # K_tiles = random.choice(divisors)

    data_str += [format_scalar_definition('uint32_t', 'M_tiles', kwargs['M_tiles'])]
    data_str += [format_scalar_definition('uint32_t', 'N_tiles', kwargs['N_tiles'])]
    data_str += [format_scalar_definition('uint32_t', 'K_tiles', kwargs['K_tiles'])]

    data_str += [format_scalar_definition('uint32_t', 'dtype_size', kwargs['prec'] // 8)]

    data_str += [format_scalar_definition(C_TYPES[str(kwargs['prec'])], 'alpha', alpha)]


    data_str += [format_array_definition(C_TYPES[str(kwargs['prec'])], 'A', A.flatten(),
                                         alignment=BURST_ALIGNMENT, section=kwargs['section'])]
    data_str += [format_array_definition(C_TYPES[str(kwargs['prec'])], 'B', B.flatten(),
                                         alignment=BURST_ALIGNMENT, section=kwargs['section'])]

    data_str += [format_array_definition(C_TYPES[str(kwargs['prec'])], 'GRAD_A', OLD_GRAD_A.flatten(),
                                         alignment=BURST_ALIGNMENT, section=kwargs['section'])]

    data_str += [format_array_definition(C_TYPES[str(kwargs['prec'])], 'GRAD_B', OLD_GRAD_B.flatten(),
                                         alignment=BURST_ALIGNMENT, section=kwargs['section'])]
    
    data_str += [format_array_definition(C_TYPES[str(kwargs['prec'])], 'GRAD_C', GRAD_C.flatten(),
                                         alignment=BURST_ALIGNMENT, section=kwargs['section'])]


    GRAD_A = format_array_definition(C_TYPES[str(kwargs['prec'])],
                                          'RES_GRAD_A',
                                          GRAD_A.flatten())
    

    GRAD_B = format_array_definition(C_TYPES[str(kwargs['prec'])],
                                          'RES_GRAD_B',
                                          GRAD_B.flatten())

    data_str += [format_ifdef_wrapper('BIST', GRAD_A)]

    data_str += [format_ifdef_wrapper('BIST', GRAD_B)]

    data_str = '\n\n'.join(data_str)

    return data_str


def main():
    parser = argparse.ArgumentParser(description='Generate data for kernels')
    parser.add_argument(
        "-c", "--cfg",
        type=pathlib.Path,
        required=True,
        help='Select param config file kernel'
    )
    parser.add_argument(
        '--section',
        type=str,
        help='Section to store matrices in')
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    param['section'] = args.section

    # Emit header file
    print(emit_header(**param))


if __name__ == '__main__':
    main()
