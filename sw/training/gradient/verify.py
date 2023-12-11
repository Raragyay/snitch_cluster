#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import sys
from pathlib import Path
import numpy as np
from data.datagen import golden_model
import torch
from torch import nn
import torch.optim as optim

sys.path.append(str(Path(__file__).parent / '../../../util/sim/'))
import verification  # noqa: E402
from elf import Elf  # noqa: E402
from data_utils import bytes_to_float, bytes_to_int  # noqa: E402


ERR_THRESHOLD = 0.001
PREC = '64'

NUMPY_TYPES = {
    '64': np.double,
    '32': np.single,
    '16': np.half,
    '8': np.ubyte
}


def main():
    # Run simulation and get outputs
    args = verification.parse_args()
    raw_results = verification.simulate(sim_bin=args.sim_bin,
                                        snitch_bin=args.snitch_bin,
                                        symbols_bin=args.symbols_bin,
                                        log=args.log,
                                        output_uids=['GRAD_A', 'GRAD_B'])
    grad_A_actual = np.array(bytes_to_float(raw_results['GRAD_A'], prec=PREC))
    grad_B_actual = np.array(bytes_to_float(raw_results['GRAD_B'], prec=PREC))
    actuals = np.concatenate((grad_A_actual,grad_B_actual))
    
    # Extract input operands from ELF file
    if args.symbols_bin:
        elf = Elf(args.symbols_bin)
    else:
        elf = Elf(args.snitch_bin)
    A = np.array(bytes_to_float(elf.get_symbol_contents('A'), prec=PREC))
    B = np.array(bytes_to_float(elf.get_symbol_contents('B'), prec=PREC))
    GRAD_C = np.array(bytes_to_float(elf.get_symbol_contents('GRAD_C'), prec=PREC))
    alpha = bytes_to_float(elf.get_symbol_contents('alpha'), prec=PREC)

    M = bytes_to_int(elf.get_symbol_contents('M'), prec='32', signedness='unsigned')[0]
    N = bytes_to_int(elf.get_symbol_contents('N'), prec='32', signedness='unsigned')[0]
    K = bytes_to_int(elf.get_symbol_contents('K'), prec='32', signedness='unsigned')[0]

    dtype = NUMPY_TYPES[PREC]
    A = np.reshape(A, (M,K)).astype(dtype)
    B = np.reshape(B, (K,N)).astype(dtype)
    GRAD_C = np.reshape(GRAD_C, (M,N)).astype(dtype)

    print("M",M,"N",N,"K",K)
    #print("A\n",np.transpose(A))    
    print("GRAD_C:\n", GRAD_C)
    print("B:\n",np.transpose(B))

    # Verify results

    grad_A_golden, grad_B_golden = golden_model(alpha, A,B,GRAD_C)
    goldens = np.concatenate((grad_A_golden.flatten(),grad_B_golden.flatten()))
    
    #goldens = grad_A_golden.flatten()
    #actuals = grad_A_actual

    absolute_err = np.absolute(  actuals - goldens )
    fail = np.any(absolute_err > ERR_THRESHOLD)
    if (fail):
        print("FAIL\n")
        verification.dump_results_to_csv([actuals, goldens, absolute_err],
                                         Path.cwd() / 'gradient_results.csv')
    else:
        print("SUCCESS\n")
        verification.dump_results_to_csv([actuals, goldens, absolute_err],
                                          Path.cwd() / 'gradient_results.csv')
    return int(fail)


if __name__ == "__main__":
    sys.exit(main())
