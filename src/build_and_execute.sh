#!/bin/bash

# Fail on first error.
set -e

current_dir=$(pwd)
cd $current_dir/src

# Run the python script to generate the torchscript model
python model_generation_main.py

# Build and execute pytorch_main.cc
echo ""
echo "Building pytorch_main.cc"
rm -rf build && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
cmake --build . --config Release
echo "Executing pytorch_main.cc"
./pytorch_main ../model.pt
