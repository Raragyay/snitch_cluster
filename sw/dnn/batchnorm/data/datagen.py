#!/usr/bin/env python3
# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Tim Fischer <fischeti@iis.ee.ethz.ch>
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Luca Colagrande <colluca@iis.ee.ethz.ch>

import argparse
import pathlib
import hjson
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
import data_utils  # noqa: E402
from data_utils import (
    emit_license,
    format_struct_definition,
    format_array_definition,
    format_array_declaration,
    format_ifdef_wrapper,
)  # noqa: E402

torch.manual_seed(42)

# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096

PRECISION_T = {"64": "FP64", "32": "FP32", "16": "FP16", "8": "FP8"}


def golden_model(ifmap, eps, running_mean, running_var, weight, bias, dtype):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps, dtype=dtype)
    # TODO: have the randomization for weight and bias
    bn.weight.requires_grad = False
    bn.bias.requires_grad = False
    # print(bn.running_mean.dtype, running_mean.dtype)
    bn.running_mean = running_mean
    bn.running_var = running_var
    # running_mean = torch.randn_like(bn.running_mean, requires_grad=False)
    # running_var = torch.rand_like(bn.running_var, requires_grad=False)
    # gamma = bn.weight / torch.sqrt(running_var + bn.eps)
    # beta = bn.bias - running_mean * bn.weight / torch.sqrt(running_var + bn.eps)
    # ofmap = ifmap * gamma.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)
    
    # print(bn(ifmap))
    bn.eval()
    print(ifmap)
    print(bn(ifmap))
    return bn(ifmap)


def emit_header(**kwargs):
    batch_size = 1
    in_channels = kwargs["input_dim"]["channels"]
    in_height = kwargs["input_dim"]["height"]
    in_width = kwargs["input_dim"]["width"]
    eps = kwargs["eps"]
    tile_ci = kwargs["tile_ci"]
    prec = str(kwargs["prec"])

    # art: going to need to write this prec out somewhere so we can read it back in

    torch_dtype = data_utils.floating_point_torch_type(prec)
    ctype = data_utils.floating_point_ctype(prec)

    eps = kwargs["eps"]
    print(type(eps))

    running_mean = torch.randn(
        in_channels,
        requires_grad=False,
        dtype=torch_dtype,
    )
    running_var = torch.abs(
        torch.randn(
            in_channels,
            requires_grad=False,
            dtype=torch_dtype,
        )
    )
    weight = torch.ones(
        in_channels,
        requires_grad=False,
        dtype=torch_dtype,
    )
    bias = torch.zeros(
        in_channels,
        requires_grad=False,
        dtype=torch_dtype,
    )

    gamma = weight / torch.sqrt(running_var + eps)
    beta = bias - running_mean * gamma

    ifmap = torch.randn(
        batch_size,
        in_channels,
        in_height,
        in_width,
        requires_grad=False,
        dtype=torch_dtype,
    )
    ofmap = golden_model(
        ifmap, eps, running_mean, running_var, weight, bias, torch_dtype
    )
    # consider .detach().numpy()

    # convert from CHW to HWC format
    ifmap = ifmap.permute(0, 2, 3, 1)
    ofmap = ofmap.permute(0, 2, 3, 1)

    batch_size, ih, iw, ci = ifmap.shape

    ifmap_uid = "ifmap"
    ofmap_uid = "ofmap"
    beta_uid = "beta"
    gamma_uid = "gamma"
    running_mean_uid = "running_mean"
    running_var_uid = "running_var"

    layer_cfg = {
        "CI": ci,
        "IH": ih,
        "IW": iw,
        "TILE_CI": tile_ci,
        "ifmap": ifmap_uid,
        "ofmap": ofmap_uid,
        "beta": beta_uid,
        "gamma": gamma_uid,
        "eps": eps,
        "dtype": PRECISION_T[prec],
    }

    data_str = [emit_license()]
    # Array forward declarations
    data_str += [format_array_declaration(ctype, ifmap_uid, ifmap.shape)]
    data_str += [format_array_declaration(ctype, ofmap_uid, ofmap.shape)]
    data_str += [format_array_declaration(ctype, beta_uid, beta.shape)]
    data_str += [format_array_declaration(ctype, gamma_uid, gamma.shape)]
    data_str += [format_array_declaration(ctype, running_mean_uid, running_mean.shape)]
    data_str += [format_array_declaration(ctype, running_var_uid, running_var.shape)]
    # TODO: add in weight and bias once they're randomized
    # Layer struct
    data_str += [format_struct_definition("batchnorm_layer_t", "layer", layer_cfg)]
    # Array definitions
    data_str += [format_array_definition(ctype, ifmap_uid, ifmap)]
    data_str += [format_array_definition(ctype, beta_uid, beta)]
    data_str += [format_array_definition(ctype, gamma_uid, gamma)]
    data_str += [format_array_definition(ctype, running_mean_uid, running_mean)]
    data_str += [format_array_definition(ctype, running_var_uid, running_var)]
    # Golden results for BIST
    result_def = format_array_definition(ctype, "golden", ofmap)
    data_str += [format_ifdef_wrapper("BIST", result_def)]
    data_str = "\n\n".join(data_str)

    return data_str


def main():
    parser = argparse.ArgumentParser(description="Generate data for layernorm kernel")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    parser.add_argument("--section", type=str, help="Section to store matrices in")
    parser.add_argument(
        "output", type=pathlib.Path, help="Path of the output header file"
    )
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    param["section"] = args.section

    # Emit header file
    with open(args.output, "w") as f:
        f.write(emit_header(**param))


if __name__ == "__main__":
    main()
