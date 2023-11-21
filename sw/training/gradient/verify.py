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

def main():
    # Run simulation and get outputs
    args = verification.parse_args()
    raw_results = verification.simulate(sim_bin=args.sim_bin,
                                        snitch_bin=args.snitch_bin,
                                        symbols_bin=args.symbols_bin,
                                        log=args.log,
                                        output_uids=['W','B',"n_iter", "fix_size_n", "fix_size_k","local_E_array"])
    weight_actual = np.array(bytes_to_float(raw_results['W'], prec=PREC))
    bias_actual = np.array(bytes_to_float(raw_results['B'], prec=PREC))
    actuals = np.concatenate((weight_actual,bias_actual))
    local_E_array = np.array(bytes_to_float(raw_results['local_E_array'], prec=PREC))

    n_iter = np.array(bytes_to_int(raw_results['n_iter'], prec='32', signedness='unsigned'))
    fix_size_n = bytes_to_int(raw_results['fix_size_n'], prec='32', signedness='unsigned')[0]
    fix_size_k = bytes_to_int(raw_results['fix_size_k'], prec='32', signedness='unsigned')[0]

    print("UB_I:",n_iter[0])
    print("UB_J:",n_iter[1])
    print("Fix size over n: ",fix_size_n)
    print("Fix size over k: ", fix_size_k)
    #print("Local_E_array:", local_E_array)

    # Extract input operands from ELF file
    if args.symbols_bin:
        elf = Elf(args.symbols_bin)
    else:
        elf = Elf(args.snitch_bin)
    I = np.array(bytes_to_float(elf.get_symbol_contents('I'), prec=PREC))
    W = np.array(bytes_to_float(elf.get_symbol_contents('W'), prec=PREC))
    B = np.array(bytes_to_float(elf.get_symbol_contents('B'), prec=PREC))
    E = np.array(bytes_to_float(elf.get_symbol_contents('E'), prec=PREC))
    M = bytes_to_int(elf.get_symbol_contents('M'), prec='32', signedness='unsigned')[0]
    N = bytes_to_int(elf.get_symbol_contents('N'), prec='32', signedness='unsigned')[0]
    K = bytes_to_int(elf.get_symbol_contents('K'), prec='32', signedness='unsigned')[0]
    e = bytes_to_float(elf.get_symbol_contents('e'), prec='64')



    # Verify results
    result_weight,result_bias = golden_model(I.reshape(1,K),W.reshape(K,N),B.reshape(1,N),E.reshape(1,N),e,K,N)
    result_weight = np.ravel(torch.transpose(result_weight, 1, 0).detach().numpy())
    result_bias = np.ravel(result_bias.detach().numpy())
    results = np.concatenate((result_weight,result_bias))

    absolute_err = np.absolute( results - actuals )
    fail = np.any(absolute_err > ERR_THRESHOLD)
    if (fail):
        print("FAIL\n")
        verification.dump_results_to_csv([results, actuals, absolute_err],
                                         Path.cwd() / 'gradient_results.csv')
    else:
        print("SUCCESS\n")
        verification.dump_results_to_csv([results, actuals, absolute_err],
                                         Path.cwd() / 'gradient_results.csv')
    return int(fail)


if __name__ == "__main__":
    sys.exit(main())
