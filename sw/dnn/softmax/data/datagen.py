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
from data_utils import emit_license, \
                       format_struct_definition, format_array_definition, \
                       format_array_declaration, format_ifdef_wrapper  # noqa: E402

torch.manual_seed(42)

# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096

PRECISION_T = {
    '64': 'FP64',
    '32': 'FP32',
    '16': 'FP16',
    '8': 'FP8'
}


def golden_model(ifmap, axis):
    softmax = torch.nn.Softmax(dim=axis)
    return softmax(ifmap)


def emit_header(**kwargs):
    batch_size = kwargs['input_dim']['batch_size']
    seq_len = kwargs['input_dim']['seq_len']
    input_samples = kwargs['input_dim']['input_samples']
    reduce_dim = kwargs['reduce_dim']
    prec = str(kwargs['prec'])

    torch_type = data_utils.floating_point_torch_type(prec)
    ifmap = torch.randn(batch_size, seq_len, input_samples, requires_grad=False, dtype=torch_type)

    ofmap = golden_model(ifmap, reduce_dim)
    ofmap = ofmap.detach().numpy()

    ctype = data_utils.floating_point_ctype(prec)

    ifmap_uid = 'ifmap'
    ofmap_uid = 'ofmap'

    layer_cfg = {
        **kwargs['input_dim'],
        'reduce_dim': reduce_dim,
        'ifmap': ifmap_uid,
        'ofmap': ofmap_uid,
        'dtype': PRECISION_T[prec]
    }

    data_str = [emit_license()]
    data_str += [format_array_declaration(ctype, ifmap_uid, ifmap.shape,
                 alignment=BURST_ALIGNMENT)]
    data_str += [format_array_declaration(ctype, ofmap_uid, ofmap.shape,
                 alignment=BURST_ALIGNMENT)]
    data_str += [format_struct_definition('softmax_layer_t', 'layer', layer_cfg)]
    data_str += [format_array_definition(ctype, ifmap_uid, ifmap,
                 alignment=BURST_ALIGNMENT)]
    result_def = format_array_definition(ctype, 'golden', ofmap, alignment=BURST_ALIGNMENT)
    data_str += [format_ifdef_wrapper('BIST', result_def)]
    data_str = '\n\n'.join(data_str)

    return data_str


def main():

    parser = argparse.ArgumentParser(description='Generate data for layernorm kernel')
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
    parser.add_argument(
        'output',
        type=pathlib.Path,
        help='Path of the output header file')
    args = parser.parse_args()

    # Load param config file
    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    param['section'] = args.section

    # Emit header file
    with open(args.output, 'w') as f:
        f.write(emit_header(**param))


if __name__ == '__main__':
    main()
