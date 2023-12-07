#!/bin/bash

cd target/snitch_cluster

alias mksw="make DEBUG=ON sw"

alias maxpool="bin/snitch_cluster.vlt sw/apps/onnx/maxpool/build/maxpool.elf"

alias verify="python3 ../../sw/onnx/maxpool/verify.py bin/snitch_cluster.vlt sw/apps/onnx/maxpool/build/maxpool.elf; stty echo"

alias verify1="python3 ../../sw/onnx/maxpool/verify1d.py bin/snitch_cluster.vlt sw/apps/onnx/maxpool/build/maxpool.elf; stty echo"

alias verify2="python3 ../../sw/onnx/maxpool/verify2d.py bin/snitch_cluster.vlt sw/apps/onnx/maxpool/build/maxpool.elf; stty echo"

alias verifyall="python3 ../../sw/onnx/maxpool/verifyall.py bin/snitch_cluster.vlt sw/apps/onnx/maxpool/build/maxpool.elf; stty echo"

alias generate="python3 ../../sw/onnx/maxpool/data/datagen.py --cfg ../../sw/onnx/maxpool/data/params.hjson ../../sw/onnx/maxpool/data/data.h"

alias anal="node ../../sw/onnx/maxpool/data/analyze_benchmark.js"

alias maxpoolold="bin/snitch_cluster.vlt sw/apps/dnn/maxpool/build/maxpool.elf"

alias analold="node ../../sw/dnn/maxpool/data/analyze_benchmark.js"
