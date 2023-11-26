#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import sys
from pathlib import Path
from operator import itemgetter

import numpy as np
import torch
from data.golden_models import golden_model_backward_training, my_golden_model_backward_training
from datetime import datetime

sys.path.append(str(Path(__file__).parent / "../../../util/sim/"))
import verification  # noqa: E402
from elf import Elf  # noqa: E402
from data_utils import (  # noqa: E402
    bytes_to_float,
    bytes_to_struct,
    floating_point_torch_type,
)
import pickle

ERR_THRESHOLD = 1e-6

PRECISION_T = {8: "64", 4: "32", 2: "16", 1: "8"}

NUMPY_T = {"64": np.float64, "32": np.float32, "16": np.float16}

elf = None


def extract_torch_arr(label, prec, dtype, shape: tuple, requires_grad=False):
    numpy_rep = np.array(
        bytes_to_float(elf.get_symbol_contents(label), prec), dtype=NUMPY_T[prec]
    )
    numpy_rep = numpy_rep.reshape(shape)
    return torch.tensor(numpy_rep, requires_grad=requires_grad)

def check_correctness(test, golden,actual):
    golden = golden.detach().numpy().flatten()
    actual = actual.detach().numpy().flatten()
    absolute_err = np.absolute(golden - actual)
    fail = np.any(absolute_err > ERR_THRESHOLD)
    if fail:
        print(f"FAIL: {test} verification failed.")
        verification.dump_results_to_csv(
            [golden, actual, absolute_err],
            Path(__file__).parent / "batchnorm_backward_training_test_results"/f"{test}.csv",
        )
    else:
        print(f"{test} verification passed.")
    return int(fail)

def main():
    global elf
    # Run simulation and get outputs
    args = verification.parse_args()
    raw_results = verification.simulate(
        sim_bin=args.sim_bin,
        snitch_bin=args.snitch_bin,
        symbols_bin=args.symbols_bin,
        log=args.log,
        output_uids=["grad_ifmap_training", "grad_weight_training", "grad_bias_training", "temp"],
    )

    print("Simulation complete. Verifying result...")
    # Extract input operands from ELF file
    if args.symbols_bin:
        elf = Elf(args.symbols_bin)
    else:
        elf = Elf(args.snitch_bin)

    backward_training_layer_struct = {
        "CI": "I",
        "IH": "I",
        "IW": "I",
        # "TILE_CI": "I",
        "ifmap": "I",
        "grad_ofmap": "I",
        "current_mean": "I",
        "current_var": "I",
        "weight": "I",
        "grad_ifmap": "I",
        "grad_weight": "I",
        "grad_bias": "I",
        "eps": "f",
        "dtype": "I",
    }
    layer = bytes_to_struct(elf.get_symbol_contents("backward_training_layer"), backward_training_layer_struct)
    CI, IH, IW= itemgetter("CI", "IH", "IW")(layer)
    eps = layer["eps"]
    prec = PRECISION_T[layer["dtype"]]

    # TODO: support multiple items in batch
    ifmap = extract_torch_arr("ifmap", prec, NUMPY_T[prec], (1, IH, IW, CI), requires_grad=True)
    current_mean = extract_torch_arr("current_mean", prec, NUMPY_T[prec], (CI,))
    current_var = extract_torch_arr("current_var", prec, NUMPY_T[prec], (CI,))
    weight = torch.nn.Parameter(
        extract_torch_arr("weight", prec, NUMPY_T[prec], (CI,))
    )
    bias = torch.nn.Parameter(
        extract_torch_arr("bias", prec, NUMPY_T[prec], (CI,))
    )
    grad_ofmap = extract_torch_arr("grad_ofmap", prec, NUMPY_T[prec], (1, IH, IW, CI))
    # Verify results

    # extract computed results from simulation
    grad_ifmap_actual = torch.from_numpy(
        np.array(
            bytes_to_float(raw_results["grad_ifmap_training"], prec), dtype=NUMPY_T[prec]
        ).reshape((1, IH, IW, CI))
    )
    grad_weight_actual = torch.from_numpy(
        np.array(
            bytes_to_float(raw_results["grad_weight_training"], prec), dtype=NUMPY_T[prec]
        ).reshape((CI,))
    )
    grad_bias_actual = torch.from_numpy(
        np.array(
            bytes_to_float(raw_results["grad_bias_training"], prec), dtype=NUMPY_T[prec]
        ).reshape((CI,))
    )
    temp = torch.from_numpy(
        np.array(
            bytes_to_float(raw_results["temp"], prec), dtype=NUMPY_T[prec]
        ).reshape((8,CI))
    )
    
    # convert from NHWC to NCHW format
    ifmap = ifmap.permute(0, 3, 1, 2)
    ifmap.retain_grad()
    grad_ofmap = grad_ofmap.permute(0,3,1,2)

    # print("All data extracted from simulation and binary. Comparing to golden model. ")
    grad_ifmap_golden, grad_weight_golden, grad_bias_golden = (
        my_golden_model_backward_training(
            ifmap,
            grad_ofmap,
            weight,
            bias,
            current_mean,
            current_var,
            eps,
            floating_point_torch_type(prec),
        )
    )

    bias_fail = check_correctness("grad_bias", grad_bias_golden, grad_bias_actual)
    weight_fail = check_correctness("grad_weight", grad_weight_golden, grad_weight_actual)
    # write out in NHWC format
    ifmap_fail = check_correctness("grad_ifmap", grad_ifmap_golden.permute(0,2,3,1), grad_ifmap_actual)

    return int(bias_fail or weight_fail or ifmap_fail)


if __name__ == "__main__":
    sys.exit(main())
