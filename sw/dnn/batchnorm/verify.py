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
from data.datagen import golden_model

sys.path.append(str(Path(__file__).parent / "../../../util/sim/"))
import verification  # noqa: E402
from elf import Elf  # noqa: E402
from data_utils import (
    bytes_to_float,
    bytes_to_struct,
    floating_point_torch_type,
)  # noqa: E402


ERR_THRESHOLD = 0.001

PRECISION_T = {8: "64", 4: "32", 2: "16", 1: "8"}

NUMPY_T = {"64": np.float64, "32": np.float32, "16": np.float16}

elf = None


def extract_torch_arr(label, prec, dtype, shape: tuple):
    numpy_rep = np.array(
        bytes_to_float(elf.get_symbol_contents(label), prec), dtype=NUMPY_T[prec]
    )
    numpy_rep = numpy_rep.reshape(shape)
    return torch.from_numpy(numpy_rep)


def main():
    global elf
    # Run simulation and get outputs
    args = verification.parse_args()
    raw_results = verification.simulate(
        sim_bin=args.sim_bin,
        snitch_bin=args.snitch_bin,
        symbols_bin=args.symbols_bin,
        log=args.log,
        output_uids=["ofmap"],
    )

    # Extract input operands from ELF file
    if args.symbols_bin:
        elf = Elf(args.symbols_bin)
    else:
        elf = Elf(args.snitch_bin)

    layer_struct = {
        "CI": "I",
        "IH": "I",
        "IW": "I",
        "TILE_CI": "I",
        "ifmap": "I",
        "ofmap": "I",
        "beta": "I",
        "gamma": "I",
        "eps": "f",
        "dtype": "I",
    }
    layer = bytes_to_struct(elf.get_symbol_contents("layer"), layer_struct)
    CI, IH, IW, TILE_CI = itemgetter("CI", "IH", "IW", "TILE_CI")(layer)
    eps = layer["eps"]
    prec = PRECISION_T[layer["dtype"]]

    # TODO: support multiple items in batch
    ifmap = extract_torch_arr("ifmap", prec, NUMPY_T[prec], (1, IH, IW, CI))
    print(f"verify: ifmap shape is {ifmap.shape}")
    running_mean = extract_torch_arr("running_mean", prec, NUMPY_T[prec], (CI))
    running_var = extract_torch_arr("running_var", prec, NUMPY_T[prec], (CI))

    # Verify results

    # extract computed results from simulation
    ofmap_actual = torch.from_numpy(
        np.array(
            bytes_to_float(raw_results["ofmap"], prec), dtype=NUMPY_T[prec]
        ).reshape((1, IH, IW, CI))
    )
    print(f"verify: ofmap is {ofmap_actual}")

    print(f"verify: ofmap shape is {ofmap_actual.shape}")
    # convert from NHWC to NCHW format
    ifmap = ifmap.permute(0, 3, 1, 2)
    ofmap_actual = ofmap_actual.permute(0, 3, 1, 2).detach().numpy().flatten()
    ofmap_golden = (
        golden_model(
            ifmap,
            eps,
            running_mean,
            running_var,
            None,
            None,
            floating_point_torch_type(prec),
        )
        .detach()
        .numpy()
        .flatten()
    )

    absolute_err = np.absolute(ofmap_golden - ofmap_actual)
    fail = np.any(absolute_err > ERR_THRESHOLD)
    if fail:
        verification.dump_results_to_csv(
            [ofmap_golden, ofmap_actual, absolute_err],
            Path.cwd() / "batchnorm_results.csv",
        )

    return int(fail)


if __name__ == "__main__":
    sys.exit(main())
