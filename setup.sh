#!/bin/bash

cd target/snitch_cluster

alias mksw="make DEBUG=ON sw"

alias maxpool="bin/snitch_cluster.vlt sw/apps/onnx/maxpool/build/maxpool.elf"

alias verify="python3 ../../sw/onnx/maxpool/verify.py bin/snitch_cluster.vlt sw/apps/onnx/maxpool/build/maxpool.elf"

alias generate="python3 ../../sw/onnx/maxpool/data/datagen.py --cfg ../../sw/onnx/maxpool/data/params.hjson ../../sw/onnx/maxpool/data/data.h"
