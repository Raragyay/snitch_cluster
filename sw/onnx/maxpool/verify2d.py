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

from verify import verify

def main():
  # Run simulation and get outputs
  args = verification.parse_args()
  raw_results = verification.simulate(
    sim_bin=args.sim_bin,
    snitch_bin=args.snitch_bin,
    symbols_bin=args.symbols_bin,
    log=args.log,
    output_uids=["output_loc2", "idx_loc2"]
  )

  # Extract input operands from ELF file
  if args.symbols_bin:
    elf = Elf(args.symbols_bin)
  else:
    elf = Elf(args.snitch_bin)

  ret = verify("ifmap2", "attr2", ["output_loc2", "idx_loc2"], elf=elf, raw_results=raw_results, id=2)
  if ret == 0:
    print("2D good")

  return 0


if __name__ == "__main__":
    sys.exit(main())
