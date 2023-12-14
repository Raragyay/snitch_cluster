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
  parser = verification.get_parser()
  parser.add_argument('--no-index', help='Do not check indexes.', action='store_true')
  args = parser.parse_args()
  raw_results = verification.simulate(
    sim_bin=args.sim_bin,
    snitch_bin=args.snitch_bin,
    symbols_bin=args.symbols_bin,
    log=args.log,
    output_uids=[
      "output_loc1_1", "output_loc2_1", "output_loc3_1", "idx_loc1_1", "idx_loc2_1", "idx_loc3_1",
      "output_loc1_2", "output_loc2_2", "output_loc3_2", "idx_loc1_2", "idx_loc2_2", "idx_loc3_2",
      "output_loc1_3", "output_loc2_3", "output_loc3_3", "idx_loc1_3", "idx_loc2_3", "idx_loc3_3",
      "output_loc1_4", "output_loc2_4", "output_loc3_4", "idx_loc1_4", "idx_loc2_4", "idx_loc3_4"
    ]
  )

  # Extract input operands from ELF file
  if args.symbols_bin:
    elf = Elf(args.symbols_bin)
  else:
    elf = Elf(args.snitch_bin)

  ret = verify("ifmap1_1", "attr1_1", ["output_loc1_1", "idx_loc1_1"], elf=elf, raw_results=raw_results, id="1_1", no_index=args.no_index)
  if ret == 0:
    print("1D good 1")

  ret = verify("ifmap2_1", "attr2_1", ["output_loc2_1", "idx_loc2_1"], elf=elf, raw_results=raw_results, id="2_1", no_index=args.no_index)
  if ret == 0:
    print("2D good 1")

  ret = verify("ifmap3_1", "attr3_1", ["output_loc3_1", "idx_loc3_1"], elf=elf, raw_results=raw_results, id="3_1", no_index=args.no_index)
  if ret == 0:
    print("3D good 1")


  ret = verify("ifmap1_2", "attr1_2", ["output_loc1_2", "idx_loc1_2"], elf=elf, raw_results=raw_results, id="1_2", no_index=args.no_index)
  if ret == 0:
    print("1D good 2")

  ret = verify("ifmap2_2", "attr2_2", ["output_loc2_2", "idx_loc2_2"], elf=elf, raw_results=raw_results, id="2_2", no_index=args.no_index)
  if ret == 0:
    print("2D good 2")

  ret = verify("ifmap3_2", "attr3_2", ["output_loc3_2", "idx_loc3_2"], elf=elf, raw_results=raw_results, id="3_2", no_index=args.no_index)
  if ret == 0:
    print("3D good 2")


  ret = verify("ifmap1_3", "attr1_3", ["output_loc1_3", "idx_loc1_3"], elf=elf, raw_results=raw_results, id="1_3", no_index=args.no_index)
  if ret == 0:
    print("1D good 3")

  ret = verify("ifmap2_3", "attr2_3", ["output_loc2_3", "idx_loc2_3"], elf=elf, raw_results=raw_results, id="2_3", no_index=args.no_index)
  if ret == 0:
    print("2D good 3")

  ret = verify("ifmap3_3", "attr3_3", ["output_loc3_3", "idx_loc3_3"], elf=elf, raw_results=raw_results, id="3_3", no_index=args.no_index)
  if ret == 0:
    print("3D good 3")

  ret = verify("ifmap1_4", "attr1_4", ["output_loc1_4", "idx_loc1_4"], elf=elf, raw_results=raw_results, id="1_4", no_index=args.no_index)
  if ret == 0:
    print("1D good 4")

  ret = verify("ifmap2_4", "attr2_4", ["output_loc2_4", "idx_loc2_4"], elf=elf, raw_results=raw_results, id="2_4", no_index=args.no_index)
  if ret == 0:
    print("2D good 4")

  ret = verify("ifmap3_4", "attr3_4", ["output_loc3_4", "idx_loc3_4"], elf=elf, raw_results=raw_results, id="3_4", no_index=args.no_index)
  if ret == 0:
    print("3D good 4")

  return 0


if __name__ == "__main__":
    sys.exit(main())
