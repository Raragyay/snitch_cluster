#!/usr/bin/env python3

# Usage (from target/snitch_cluster): python3 ../../sw/onnx/maxpool/verify.py bin/snitch_cluster.vlt sw/apps/onnx/maxpool/build/maxpool.elf

import sys
from pathlib import Path
import numpy as np
from torch import from_numpy
from data.datagen import golden_model_1d, golden_model_2d, golden_model_3d

sys.path.append(str(Path(__file__).parent / '../../../util/sim/'))
import verification  # noqa: E402
from elf import Elf  # noqa: E402
from data_utils import bytes_to_float, bytes_to_int, bytes_to_struct  # noqa: E402


ERR_THRESHOLD = 0.001

# .n_dim = 1,
# .input_shape = {1, 1, 10, -1, -1},
# .output_shape = {0},
# .auto_pad = NOTSET,
# .ceil_mode = 0,
# .dilations = {2, -1, -1},
# .kernel_shape = {4, -1, -1},
# .pads = {1, 1, -1, -1, -1, -1},
# .storage_order = 0,
# .strides = {2, -1, -1}

def verify(input_name, attribs_name, output_name, elf=None, raw_results=None, id=0):
  c_actual = np.array(bytes_to_float(raw_results[output_name[0]], prec="64"))
  i_actual = np.array(bytes_to_int(raw_results[output_name[1]], prec="32"))
  attr_map = {
    "n_dim": "i",
    "input_shape": "iiiii",
    "output_shape": "iiiii",
    "auto_pad": "I",
    "ceil_mode": "I",
    "dilations": "iii",
    "kernel_shape": "iii",
    "pads": "iiiiii",
    "storage_order": "I",
    "strides": "iii"
  }
  attribs = bytes_to_struct(elf.get_symbol_contents(attribs_name), attr_map)
  # Truncate to correct size
  inputs = np.array(bytes_to_float(elf.get_symbol_contents(input_name), prec="64"))[0:np.array(attribs["input_shape"]).prod()]

  n_dim = attribs["n_dim"]
  golden_model = None
  if n_dim == 1:
    golden_model = golden_model_1d
  elif n_dim == 2:
    golden_model = golden_model_2d
  elif n_dim == 3:
    golden_model = golden_model_3d

  [c_golden, i_golden] = golden_model(
    from_numpy(inputs.reshape(attribs["input_shape"][0:n_dim + 2])),
    attribs["kernel_shape"][0:n_dim],
    attribs["strides"][0:n_dim],
    attribs["pads"][0:n_dim],
    attribs["dilations"][0:n_dim],
    bool(attribs["ceil_mode"]))
  c_golden = c_golden.flatten().detach().cpu().numpy()
  i_golden = i_golden.flatten().detach().cpu().numpy()

  c_actual = np.resize(c_actual, c_golden.shape) # In case excess memory was allocated for the array in C
  i_actual = np.resize(i_actual, i_golden.shape)

  absolute_err = np.absolute(c_golden - c_actual)
  fail = np.any(absolute_err > ERR_THRESHOLD)
  code = 0
  if fail:
    verification.dump_results_to_csv([c_golden, c_actual, absolute_err], Path.cwd() / f"maxpool_values_{id}.csv")
    code = fail

  fail = np.any(i_actual != i_golden)
  if fail:
    verification.dump_results_to_csv([i_golden, i_actual, np.absolute(i_golden - i_actual)], Path.cwd() / f"maxpool_index_{id}.csv")
    code = fail

  return int(fail)

def main():
  # Run simulation and get outputs
  args = verification.parse_args()
  raw_results = verification.simulate(
    sim_bin=args.sim_bin,
    snitch_bin=args.snitch_bin,
    symbols_bin=args.symbols_bin,
    log=args.log,
    output_uids=["output_loc1", "output_loc2", "output_loc3", "idx_loc1", "idx_loc2", "idx_loc3"]
  )

  # Extract input operands from ELF file
  if args.symbols_bin:
    elf = Elf(args.symbols_bin)
  else:
    elf = Elf(args.snitch_bin)

  ret = verify("ifmap1", "attr1", ["output_loc1", "idx_loc1"], elf=elf, raw_results=raw_results, id=1)
  if ret == 0:
    print("1D good")

  ret = verify("ifmap2", "attr2", ["output_loc2", "idx_loc2"], elf=elf, raw_results=raw_results, id=2)
  if ret == 0:
    print("2D good")

  ret = verify("ifmap3", "attr3", ["output_loc3", "idx_loc3"], elf=elf, raw_results=raw_results, id=3)
  if ret == 0:
    print("3D good")

  return 0


if __name__ == "__main__":
    sys.exit(main())
