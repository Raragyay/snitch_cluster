#!/usr/bin/env python3

# Usage: python3 datagen.py --cfg params.hjson data.h
# (from target/snitch_cluster) python3 ../../sw/onnx/maxpool/data/datagen.py --cfg ../../sw/onnx/maxpool/data/params.hjson ../../sw/onnx/maxpool/data/data.h

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

torch.manual_seed(44)

# AXI splits bursts crossing 4KB address boundaries. To minimize
# the occurrence of these splits the data should be aligned to 4KB
BURST_ALIGNMENT = 4096

PRECISION_T = {
  '64': 'FP64',
  '32': 'FP32',
  '16': 'FP16',
  '8': 'FP8'
}

# ignore indices for now
def golden_model_1d(ifmap, kernel, stride, padding, dilation, ceil_mode):
  max_pool = torch.nn.MaxPool1d(
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilation,
    return_indices=True,
    ceil_mode=ceil_mode
  )
  return max_pool(ifmap)[0]

def golden_model_2d(ifmap, kernel, stride, padding, dilation, ceil_mode):
  max_pool = torch.nn.MaxPool2d(
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilation,
    return_indices=True,
    ceil_mode=ceil_mode
  )
  return max_pool(ifmap)[0]

def golden_model_3d(ifmap, kernel, stride, padding, dilation, ceil_mode):
  max_pool = torch.nn.MaxPool3d(
    kernel_size=kernel,
    stride=stride,
    padding=padding,
    dilation=dilation,
    return_indices=True,
    ceil_mode=ceil_mode
  )
  return max_pool(ifmap)[0]

def gen(bounds):
  return torch.randint(bounds[0], bounds[1] + 1, (1,))[0]

def num_outs(shape):
  prod = 1
  for dim in shape:
    prod *= dim
  return prod

def emit_header(**kwargs):

  batches1 = gen(kwargs["batches1"])
  channels1 = gen(kwargs["channels1"])
  batches2 = gen(kwargs["batches2"])
  channels2 = gen(kwargs["channels2"])
  batches3 = gen(kwargs["batches3"])
  channels3 = gen(kwargs["channels3"])

  dim_obj = kwargs["input_dim"]
  input_dim = [
    gen(dim_obj["height"]),
    gen(dim_obj["width"]),
    gen(dim_obj["depth"])
  ]

  kernel_obj = kwargs["kernel_size"]
  kernel_size = [
    gen(kernel_obj["height"]),
    gen(kernel_obj["width"]),
    gen(kernel_obj["depth"])
  ]

  pads_obj = kwargs["pads"]
  pads = [
    gen(pads_obj["height"]),
    gen(pads_obj["width"]),
    gen(pads_obj["depth"])
  ]

  strides_obj = kwargs["strides"]
  strides = [
    gen(strides_obj["height"]),
    gen(strides_obj["width"]),
    gen(strides_obj["depth"])
  ]

  dilations_obj = kwargs["dilations"]
  dilations = [
    gen(dilations_obj["height"]),
    gen(dilations_obj["width"]),
    gen(dilations_obj["depth"])
  ]

  ceil_mode = torch.rand(1)[0].item() > kwargs["ceil_mode"]

  torch_type = data_utils.floating_point_torch_type("64")
  ctype = data_utils.floating_point_ctype("64")

  ifmap1 = torch.randn(batches1, channels1, input_dim[0], dtype=torch_type)
  ifmap1_uid = "ifmap1"
  ofmap1 = golden_model_1d(
    ifmap1,
    [kernel_size[0]],
    [strides[0]],
    [pads[0]],
    [dilations[0]],
    ceil_mode
  )
  ofmap1_uid = "ofmap1"

  ifmap2 = torch.randn(batches2, channels2, input_dim[0], input_dim[1], dtype=torch_type)
  ifmap2_uid = "ifmap2"
  ofmap2 = golden_model_2d(
    ifmap2,
    [kernel_size[0], kernel_size[1]],
    [strides[0], strides[1]],
    [pads[0], pads[1]],
    [dilations[0], dilations[1]],
    ceil_mode
  )
  ofmap2_uid = "ofmap2"

  ifmap3 = torch.randn(batches3, channels3, *input_dim, dtype=torch_type)
  ifmap3_uid = "ifmap3"
  ofmap3 = golden_model_3d(
    ifmap3,
    kernel_size,
    strides,
    pads,
    dilations,
    ceil_mode
  )
  ofmap3_uid = "ofmap3"
  print(ofmap3.shape)

  data_str = [emit_license()]

# maxpool_attributes test_1d_default_attributes = {
#   .n_dim = 1,
#   .input_shape = {1, 3, 32, -1, -1}, // 3 batches of 32 elements
#   .output_shape = {1, 3, 31, -1, -1},

#   .auto_pad = NOTSET,
#   .ceil_mode = 0,
#   .dilations = {1, -1, -1},
#   .kernel_shape = {2, -1, -1},
#   .pads = {0, 0, -1, -1, -1, -1},
#   .storage_order = 0,
#   .strides = {1, -1, -1}
# };

  attr1 = (
    f"maxpool_attributes attr1 = {{\n"
    f"  .n_dim = 1,\n"
    f"  .input_shape = {{{batches1}, {channels1}, {input_dim[0]}, -1, -1}},\n"
    f"  .output_shape = {{0}},\n" # this gets set by the call
    f"  .auto_pad = NOTSET,\n"
    f"  .ceil_mode = {1 if ceil_mode else 0},\n"
    f"  .dilations = {{{dilations[0]}, -1, -1}},\n"
    f"  .kernel_shape = {{{kernel_size[0]}, -1, -1}},\n"
    f"  .pads = {{{pads[0]}, {pads[0]}, -1, -1, -1, -1}},\n"
    f"  .storage_order = 0,\n"
    f"  .strides = {{{strides[0]}, -1, -1}}\n"
    f"}};"
  )

  attr2 = (
    f"maxpool_attributes attr2 = {{\n"
    f"  .n_dim = 2,\n"
    f"  .input_shape = {{{batches2}, {channels2}, {input_dim[0]}, {input_dim[1]}, -1}},\n"
    f"  .output_shape = {{0}},\n" # this gets set by the call
    f"  .auto_pad = NOTSET,\n"
    f"  .ceil_mode = {1 if ceil_mode else 0},\n"
    f"  .dilations = {{{dilations[0]}, {dilations[1]}, -1}},\n"
    f"  .kernel_shape = {{{kernel_size[0]}, {kernel_size[1]}, -1}},\n"
    f"  .pads = {{{pads[0]}, {pads[1]}, {pads[0]}, {pads[1]}, -1, -1}},\n"
    f"  .storage_order = 0,\n"
    f"  .strides = {{{strides[0]}, {strides[1]}, -1}}\n"
    f"}};"
  )

  attr3 = (
    f"maxpool_attributes attr3 = {{\n"
    f"  .n_dim = 3,\n"
    f"  .input_shape = {{{batches3}, {channels3}, {input_dim[0]}, {input_dim[1]}, {input_dim[2]}}},\n"
    f"  .output_shape = {{0}},\n" # this gets set by the call
    f"  .auto_pad = NOTSET,\n"
    f"  .ceil_mode = {1 if ceil_mode else 0},\n"
    f"  .dilations = {{{dilations[0]}, {dilations[1]}, {dilations[2]}}},\n"
    f"  .kernel_shape = {{{kernel_size[0]}, {kernel_size[1]}, {kernel_size[2]}}},\n"
    f"  .pads = {{{pads[0]}, {pads[1]}, {pads[2]}, {pads[0]}, {pads[1]}, {pads[2]}}},\n"
    f"  .storage_order = 0,\n"
    f"  .strides = {{{strides[0]}, {strides[1]}, {strides[2]}}}\n"
    f"}};"
  )

  data_str.append(attr1)
  data_str.append(attr2)
  data_str.append(attr3)

  data_str.append(format_array_declaration(ctype, ifmap1_uid, (num_outs(ifmap1.shape),)))
  #data_str.append(format_array_declaration(ctype, ofmap1_uid, ofmap1.shape))
  data_str.append(format_array_definition(ctype, ifmap1_uid, ifmap1.reshape((num_outs(ifmap1.shape),))))

  data_str.append(format_array_declaration(ctype, ifmap2_uid, (num_outs(ifmap2.shape),)))
  #data_str.append(format_array_declaration(ctype, ofmap2_uid, ofmap2.shape))
  data_str.append(format_array_definition(ctype, ifmap2_uid, ifmap2.reshape((num_outs(ifmap2.shape),))))

  data_str.append(format_array_declaration(ctype, ifmap3_uid, (num_outs(ifmap3.shape),)))
  #data_str.append(format_array_declaration(ctype, ofmap3_uid, ofmap3.shape))
  # 3 2 4
  # 3 4 2
  # 2 4 3
  # 3 2 4
  data_str.append(format_array_definition(ctype, ifmap3_uid, ifmap3.permute(0, 1, 2, 3, 4).reshape((num_outs(ifmap3.shape),))))

  data_str.append((
    f"{ctype} output_loc1[{num_outs(ofmap1.shape)}] = {{0}};\n"
    f"{ctype} output_loc2[{num_outs(ofmap2.shape)}] = {{0}};\n"
    f"{ctype} output_loc3[{num_outs(ofmap3.shape)}] = {{0}};\n"
  ))

  # data_str.append(format_ifdef_wrapper("BIST", format_array_definition(ctype, "golden1", ofmap1)))
  # data_str.append(format_ifdef_wrapper("BIST", format_array_definition(ctype, "golden2", ofmap2)))
  # data_str.append(format_ifdef_wrapper("BIST", format_array_definition(ctype, "golden3", ofmap3)))

  data_str = "\n\n".join(data_str)

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
