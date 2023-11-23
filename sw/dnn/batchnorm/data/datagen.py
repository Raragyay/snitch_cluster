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
from enum import Enum

from typing import Dict

from golden_models import (
    golden_model_backward,
    golden_model_backward_training,
    golden_model_forward_eval,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
import data_utils  # noqa: E402
from data_utils import (
    emit_license,
    format_struct_definition,
    format_array_definition,
    format_array_declaration,
    format_scalar_definition,
)  # noqa: E402

torch.manual_seed(42)

# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096

PRECISION_T = {"64": "FP64", "32": "FP32", "16": "FP16", "8": "FP8"}

# base
impl_opt_level_uid = "impl_opt_level"
ifmap_uid = "ifmap"
current_mean_uid = "current_mean"
current_var_uid = "current_var"
running_mean_uid = "running_mean"
running_var_uid = "running_var"
weight_uid = "weight"
bias_uid = "bias"

# backward input
grad_ofmap_uid = "grad_ofmap"

# forward eval output
ofmap_uid = "ofmap"
beta_uid = "beta"
gamma_uid = "gamma"

# backward eval output
grad_ifmap_uid = "grad_ifmap"
grad_weight_uid = "grad_weight"
grad_bias_uid = "grad_bias"

# backward training output
grad_ifmap_training_uid = "grad_ifmap_training"
grad_weight_training_uid = "grad_weight_training"
grad_bias_training_uid = "grad_bias_training"


class BatchNormMode(Enum):
    FORWARD_EVAL = 0
    FORWARD_TRAINING = 1
    BACKWARD_EVAL = 2
    BACKWARD_TRAINING = 3


struct_decls = {
    BatchNormMode.FORWARD_EVAL: ("batchnorm_layer_t", "forward_eval_layer"),
    BatchNormMode.FORWARD_TRAINING: ("batchnorm_training_layer_t", "forward_training_layer"),
    BatchNormMode.BACKWARD_EVAL: ("batchnorm_backward_layer_t", "backward_eval_layer"),
    BatchNormMode.BACKWARD_TRAINING: (
        "batchnorm_backward_training_layer_t",
        "backward_training_layer",
    ),
}


def build_base_kwargs(
    ctype,
    current_mean,
    current_var,
    running_mean,
    running_var,
    weight,
    bias,
    ifmap,
):
    return {
        "ctype": ctype,
        weight_uid: weight,
        bias_uid: bias,
        current_mean_uid: current_mean,
        current_var_uid: current_var,
        running_mean_uid: running_mean,
        running_var_uid: running_var,
        ifmap_uid: ifmap,
    }


def get_declarations(ctype, **map_uid_to_numpy_obj):
    data_str = []
    # Array forward declarations
    for uid, tensor in map_uid_to_numpy_obj.items():
        if tensor.ndim == 4:
            # convert from NCHW to NHWC format
            tensor = tensor.permute(0, 2, 3, 1)
        data_str.append(format_array_declaration(ctype, uid, tensor.shape))

    return data_str


def get_definitions(ctype, **map_uid_to_numpy_obj):
    data_str = []
    with torch.no_grad():
        for uid, tensor in map_uid_to_numpy_obj.items():
            if tensor.ndim == 4:
                # convert from NCHW to NHWC format
                tensor = tensor.permute(0, 2, 3, 1)
            data_str.append(format_array_definition(ctype, uid, tensor))
    return data_str


def get_struct_declaration(batchnorm_mode):
    layer_type, layer_name = struct_decls[batchnorm_mode]
    return f"{layer_type} {layer_name};"


def get_struct_definition(batchnorm_mode, layer_cfg):
    layer_type, layer_name = struct_decls[batchnorm_mode]
    return format_struct_definition(layer_type, layer_name, layer_cfg)


def get_forward_eval_tensors(
    H,
    W,
    C,
    ifmap,
    running_mean,
    running_var,
    weight,
    bias,
    eps,
    prec,
    TILE_CI,
    torch_dtype,
    **kwargs,
):
    with torch.no_grad():
        # Parameters to simplify calculation for core.
        gamma = weight / torch.sqrt(running_var + eps)
        beta = bias - running_mean * gamma
        ofmap = golden_model_forward_eval(
            ifmap, eps, running_mean, running_var, weight, bias, torch_dtype
        )

        layer_cfg = {
            "CI": C,
            "IH": H,
            "IW": W,
            "TILE_CI": TILE_CI,
            "ifmap": ifmap_uid,
            "ofmap": ofmap_uid,
            "beta": beta_uid,
            "gamma": gamma_uid,
            "eps": eps,
            "dtype": PRECISION_T[prec],
        }

        return (
            {gamma_uid: gamma, beta_uid: beta},
            {ofmap_uid: ofmap},
            get_struct_definition(BatchNormMode.FORWARD_EVAL, layer_cfg),
        )


def get_backward_eval_tensors(
    H,
    W,
    C,
    ifmap,
    weight,
    bias,
    running_mean,
    running_var,
    eps,
    prec,
    torch_dtype,
    **kwargs,
):
    # we just need the shape, so just use ifmap instead of recomputing ofmap
    grad_ofmap = torch.randn_like(ifmap, dtype=torch_dtype, requires_grad=False)
    grad_ifmap, grad_weight, grad_bias = golden_model_backward(
        ifmap, grad_ofmap, weight, bias, running_mean, running_var, eps, torch_dtype
    )

    layer_cfg = {
        "CI": C,
        "IH": H,
        "IW": W,
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

    return (
        {
            grad_ofmap_uid: grad_ofmap,
            grad_weight_uid: grad_weight,
            grad_bias_uid: grad_bias,
        },
        {grad_ifmap_uid: grad_ifmap},
        get_struct_definition(BatchNormMode.BACKWARD_EVAL, layer_cfg),
    )


def get_backward_training_tensors(
    C, H, W, eps, prec, ifmap, weight, bias, torch_dtype, **kwargs
):
    # we just need the shape, so just use ifmap instead of recomputing ofmap
    grad_ofmap = torch.randn_like(ifmap, dtype=torch_dtype, requires_grad=False)
    (
        grad_ifmap_training,
        grad_weight_training,
        grad_bias_training,
    ) = golden_model_backward_training(
        ifmap, grad_ofmap, weight, bias, eps, torch_dtype
    )

    layer_cfg = {
        "CI": C,
        "IH": H,
        "IW": W,
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
    return (
        {
            grad_ofmap_uid: grad_ofmap,
            grad_weight_training_uid: grad_weight_training,
            grad_bias_training_uid: grad_bias_training,
        },
        {grad_ifmap_training_uid: grad_ifmap_training},
        get_struct_definition(BatchNormMode.BACKWARD_TRAINING, layer_cfg),
    )


def emit_header(**kwargs):
    N = 1
    C = kwargs["input_dim"]["channels"]
    H = kwargs["input_dim"]["height"]
    W = kwargs["input_dim"]["width"]
    eps = kwargs["eps"]
    tile_ci = kwargs["tile_ci"]
    prec = str(kwargs["prec"])
    impl_opt_level = kwargs["impl_opt_level"]
    is_forward = kwargs["is_forward"]
    is_training = kwargs["is_training"]

    torch_dtype = data_utils.floating_point_torch_type(prec)
    ctype = data_utils.floating_point_ctype(prec)

    eps = kwargs["eps"]

    running_mean = torch.randn(
        C,
        dtype=torch_dtype,
    )
    running_var = torch.abs(
        torch.randn(
            C,
            dtype=torch_dtype,
        )
    )
    weight = torch.nn.Parameter(
        torch.randn(
            C,
            dtype=torch_dtype,
        )
    )
    bias = torch.nn.Parameter(
        torch.randn(
            C,
            dtype=torch_dtype,
        )
    )

    ifmap = torch.randn(
        (N, C, H, W),
        dtype=torch_dtype,
        requires_grad=True,
    )
    current_mean = torch.mean(ifmap, (0, 2, 3))
    current_var = torch.var(ifmap, (0, 2, 3), correction=0)

    data_str = [emit_license()]
    base_tensors = build_base_kwargs(
        ctype,
        current_mean,
        current_var,
        running_mean,
        running_var,
        weight,
        bias,
        ifmap,
    )

    config_params = {
        "TILE_CI": tile_ci,
        "prec": prec,
        "eps": eps,
        "torch_dtype": torch_dtype,
        "N": N,
        "H": H,
        "W": W,
        "C": C,
    }

    mode_specific_inputs: Dict[str, torch.Tensor] = None
    mode_specific_outputs: Dict[str, torch.Tensor] = None
    struct_def: str = None
    if is_forward and is_training:
        raise ValueError("Training model not implemented yet for forward pass.")
    elif is_forward and not is_training:
        (
            mode_specific_inputs,
            mode_specific_outputs,
            struct_def,
        ) = get_forward_eval_tensors(**base_tensors, **config_params)
    elif not is_forward and is_training:
        (
            mode_specific_inputs,
            mode_specific_outputs,
            struct_def,
        ) = get_backward_training_tensors(**base_tensors, **config_params)
    elif not is_forward and not is_training:
        (
            mode_specific_inputs,
            mode_specific_outputs,
            struct_def,
        ) = get_backward_eval_tensors(**base_tensors, **config_params)

    ## BUILD HEADER
    data_str.append(
        format_scalar_definition("impl_opt_level_t", impl_opt_level_uid, impl_opt_level)
    )
    data_str.append(format_scalar_definition("int", "is_forward", int(is_forward)))
    data_str.append(format_scalar_definition("int", "is_training", int(is_training)))
    data_str.extend(
        get_declarations(
            **base_tensors, **mode_specific_inputs, **mode_specific_outputs
        )
    )
    data_str.append(format_array_declaration(ctype, "temp", (8, C)))
    for batchnorm_mode in BatchNormMode:
        data_str.append(get_struct_declaration(batchnorm_mode))
    data_str.append(struct_def)

    data_str.extend(get_definitions(**base_tensors, **mode_specific_inputs))
    return "\n\n".join(data_str)


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
    with args.cfg.open("r") as cfg_file:
        param = hjson.loads(cfg_file.read())
    param["section"] = args.section

    # Emit header file
    with args.output.open("w") as data_header_file:
        data_header_file.write(emit_header(**param))


if __name__ == "__main__":
    main()
