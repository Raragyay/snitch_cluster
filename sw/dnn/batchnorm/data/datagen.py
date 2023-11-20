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
import data_utils  # noqa: E402
from data_utils import (
    emit_license,
    format_struct_definition,
    format_array_definition,
    format_array_declaration,
    format_ifdef_wrapper,
)  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))

torch.manual_seed(42)

# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096

PRECISION_T = {"64": "FP64", "32": "FP32", "16": "FP16", "8": "FP8"}


def golden_model_eval(ifmap, eps, running_mean, running_var, weight, bias, dtype):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps, dtype=dtype)
    bn.weight = weight
    bn.bias = bias
    bn.running_mean = running_mean
    bn.running_var = running_var
    bn.eval()
    return bn(ifmap)


def golden_model_training(ifmap, eps, momentum, running_mean, running_var, weight, bias, dtype):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps=eps, momentum=momentum, dtype=dtype)
    bn.weight = weight
    bn.bias = bias
    bn.running_mean = running_mean.clone()
    bn.running_var = running_var.clone()
    bn.eval()
    ofmap = bn(ifmap)
    return ofmap, bn.running_mean, bn.running_var


def golden_model_backward(ifmap, grad_ofmap, weight, bias, running_mean, 
                          running_var, eps, dtype) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps=eps, dtype=dtype)
    bn.weight = weight
    bn.bias = bias
    bn.running_mean = running_mean.clone()
    bn.running_var = running_var.clone()
    bn.eval()
    ofmap = bn(ifmap)
    ofmap.retain_grad()
    ofmap.flatten().dot(grad_ofmap.flatten()).backward()
    return ifmap.grad, bn.weight.grad, bn.bias.grad


def golden_model_backward_training(ifmap, grad_ofmap, weight, bias, eps, dtype) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    bn = torch.nn.BatchNorm2d(ci, eps=eps, dtype=dtype)
    bn.weight = weight
    bn.bias = bias
    ofmap = bn(ifmap)
    ofmap.retain_grad()
    ofmap.flatten().dot(grad_ofmap.flatten()).backward()
    return ifmap.grad, bn.weight.grad, bn.bias.grad


def my_golden_model_backward_training(ifmap, grad_ofmap, weight, bias,
                                      current_mean, current_var, eps, dtype) \
        -> (torch.Tensor, torch.Tensor, torch.Tensor):
    n, ci, ih, iw = ifmap.shape
    num_points = n*ih*iw
    invstd = torch.rsqrt(current_var + eps)
    sum = grad_ofmap.sum(dim=(0, 2, 3))
    dotp = torch.zeros(ci)
    for c in range(ci):
        dotp[c] = torch.sum(grad_ofmap[:, c, :, :] * ((ifmap[:, c, :, :] - current_mean[c])))
    k = dotp * invstd * invstd / num_points
    grad_mean = sum / num_points
    dx = torch.zeros(ifmap.shape)
    for c in range(ci):
        dx[:, c, :, :] = (ifmap[:, c, :, :] - current_mean[c]) * k[c]
    grad_ifmap = torch.zeros(ifmap.shape)
    for c in range(ci):
        grad_ifmap[:, c, :, :] = (grad_ofmap[:, c, :, :] - grad_mean[c] - dx[:, c, :, :])\
                               * invstd[c] * weight[c]
    grad_weight = dotp * invstd
    grad_bias = sum
    return grad_ifmap, grad_weight, grad_bias


def emit_header(**kwargs):
    batch_size = 1
    in_channels = kwargs["input_dim"]["channels"]
    in_height = kwargs["input_dim"]["height"]
    in_width = kwargs["input_dim"]["width"]
    eps = kwargs["eps"]
    tile_ci = kwargs["tile_ci"]
    prec = str(kwargs["prec"])

    torch_dtype = data_utils.floating_point_torch_type(prec)
    ctype = data_utils.floating_point_ctype(prec)

    eps = kwargs["eps"]
    print(type(eps))

    running_mean = torch.randn(
        in_channels,
        dtype=torch_dtype,
    )
    running_var = torch.abs(
        torch.randn(
            in_channels,
            dtype=torch_dtype,
        )
    )
    weight = torch.nn.Parameter(
        torch.randn(
            in_channels,
            dtype=torch_dtype,
        )
    )
    bias = torch.nn.Parameter(
        torch.randn(
            in_channels,
            dtype=torch_dtype,
        )
    )

    # Parameters to simplify calculation for core.
    gamma = weight / torch.sqrt(running_var + eps)
    beta = bias - running_mean * gamma

    ifmap = torch.randn(
        batch_size,
        in_channels,
        in_height,
        in_width,
        dtype=torch_dtype,
        requires_grad=True
    )
    current_mean = torch.mean(ifmap, (0, 2, 3))
    current_var = torch.var(ifmap, (0, 2, 3), correction=0)

    with torch.no_grad():
        ofmap = golden_model_eval(
            ifmap, eps, running_mean, running_var, weight, bias, torch_dtype
        )

    grad_ofmap = torch.randn_like(ofmap, dtype=torch_dtype, requires_grad=False)
    grad_ifmap, grad_weight, grad_bias \
        = golden_model_backward(ifmap, grad_ofmap, weight, bias,
                                running_mean, running_var, eps, torch_dtype)
    grad_ifmap_training, grad_weight_training, grad_bias_training \
        = golden_model_backward_training(ifmap, grad_ofmap, weight, bias, eps, torch_dtype)
    print(grad_ifmap_training.shape)

    with torch.no_grad():
        # convert from NCHW to NHWC format
        ifmap = ifmap.permute(0, 2, 3, 1)
        ofmap = ofmap.permute(0, 2, 3, 1)
        grad_ofmap = grad_ofmap.permute(0, 2, 3, 1)
        grad_ifmap = grad_ifmap.permute(0, 2, 3, 1)
        grad_ifmap_training = grad_ifmap_training.permute(0, 2, 3, 1)

        batch_size, ih, iw, ci = ifmap.shape

        ifmap_uid = "ifmap"
        ofmap_uid = "ofmap"
        beta_uid = "beta"
        gamma_uid = "gamma"
        current_mean_uid = "current_mean"
        current_var_uid = "current_var"
        running_mean_uid = "running_mean"
        running_var_uid = "running_var"
        weight_uid = "weight"
        bias_uid = "bias"

        grad_ofmap_uid = "grad_ofmap"
        grad_ifmap_uid = "grad_ifmap"
        grad_weight_uid = "grad_weight"
        grad_bias_uid = "grad_bias"
        grad_ifmap_training_uid = "grad_ifmap_training"
        grad_weight_training_uid = "grad_weight_training"
        grad_bias_training_uid = "grad_bias_training"

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

        backward_layer_cfg = {
            "CI": ci,
            "IH": ih,
            "IW": iw,
            "ifmap": ifmap_uid,
            "grad_ofmap": grad_ofmap_uid,
            "running_mean": running_mean_uid,
            "running_var": running_var_uid,
            "weight": weight_uid,
            "grad_ifmap": grad_ifmap_uid,
            "grad_weight": grad_weight_uid,
            "grad_bias": grad_bias_uid,
            "eps": eps,
            "dtype": PRECISION_T[prec],
        }

        backward_training_layer_cfg = {
            "CI": ci,
            "IH": ih,
            "IW": iw,
            "ifmap": ifmap_uid,
            "grad_ofmap": grad_ofmap_uid,
            "current_mean": current_mean_uid,
            "current_var": current_var_uid,
            "weight": weight_uid,
            "grad_ifmap": grad_ifmap_training_uid,
            "grad_weight": grad_weight_training_uid,
            "grad_bias": grad_bias_training_uid,
            "eps": eps,
            "dtype": PRECISION_T[prec],
        }

        # training_layer_cfg = {
        #     "CI": ci,
        #     "IH": ih,
        #     "IW": iw,
        #     "ifmap": ifmap_uid,
        #     "ofmap": ofmap_uid,
        #     "running_mean": running_mean_uid,
        #     "running_var": running_var_uid,
        #     "weight": weight_uid,
        #     "bias": bias_uid,
        #     "eps": eps,
        #     "momentum": momentum,
        #     "dtype": PRECISION_T[prec]
        # }

        data_str = [emit_license()]
        # Array forward declarations
        data_str += [format_array_declaration(ctype, ifmap_uid, ifmap.shape)]
        data_str += [format_array_declaration(ctype, ofmap_uid, ofmap.shape)]
        data_str += [format_array_declaration(ctype, beta_uid, beta.shape)]
        data_str += [format_array_declaration(ctype, gamma_uid, gamma.shape)]
        data_str += [format_array_declaration(ctype, current_mean_uid, current_mean.shape)]
        data_str += [format_array_declaration(ctype, current_var_uid, current_var.shape)]
        data_str += [format_array_declaration(ctype, running_mean_uid, running_mean.shape)]
        data_str += [format_array_declaration(ctype, running_var_uid, running_var.shape)]
        data_str += [format_array_declaration(ctype, weight_uid, weight.shape)]
        data_str += [format_array_declaration(ctype, bias_uid, bias.shape)]
        data_str += [format_array_declaration(ctype, grad_ifmap_uid, grad_ifmap.shape)]
        data_str += [format_array_declaration(ctype, grad_weight_uid, grad_weight.shape)]
        data_str += [format_array_declaration(ctype, grad_bias_uid, grad_bias.shape)]
        data_str += [format_array_declaration(ctype, grad_ofmap_uid, grad_ofmap.shape)]
        data_str += [format_array_declaration(ctype, grad_ifmap_training_uid,
                                              grad_ifmap_training.shape)]
        data_str += [format_array_declaration(ctype, grad_weight_training_uid,
                                              grad_weight_training.shape)]
        data_str += [format_array_declaration(ctype, grad_bias_training_uid,
                                              grad_bias_training.shape)]
        data_str += [format_array_declaration(ctype, "temp", (8, ci))]
        # Layer struct
        data_str += [format_struct_definition("batchnorm_layer_t", "layer", layer_cfg)]
        data_str += [
            format_struct_definition(
                "batchnorm_backward_layer_t", "backward_layer", backward_layer_cfg
            )
        ]
        data_str += [
            format_struct_definition(
                "batchnorm_backward_training_layer_t", "backward_training_layer",
                backward_training_layer_cfg
            )
        ]
        # Array definitions
        data_str += [format_array_definition(ctype, ifmap_uid, ifmap)]
        data_str += [format_array_definition(ctype, beta_uid, beta)]
        data_str += [format_array_definition(ctype, gamma_uid, gamma)]
        data_str += [format_array_definition(ctype, current_mean_uid, current_mean)]
        data_str += [format_array_definition(ctype, current_var_uid, current_var)]
        data_str += [format_array_definition(ctype, running_mean_uid, running_mean)]
        data_str += [format_array_definition(ctype, running_var_uid, running_var)]
        data_str += [format_array_definition(ctype, weight_uid, weight)]
        data_str += [format_array_definition(ctype, bias_uid, bias)]
        data_str += [format_array_definition(ctype, grad_ofmap_uid, grad_ofmap)]
        # Golden results for BIST
        result_def = format_array_definition(ctype, "golden", ofmap)
        data_str += [format_ifdef_wrapper("BIST", result_def)]
        data_str = "\n\n".join(data_str)

        # No bist for training mode

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
