#!/bin/bash

pip install -r requirements.txt

# shellcheck disable=SC2164
cd nafnet
python3 setup.py develop --no_cuda_ext
cd ../

# onnxruntime-gpu for CUDA version 12.x
# You can refer to this documentation for installing onnxruntime-gpu
# https://onnxruntime.ai/docs/install/#install-onnx-runtime-ort-1
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install onnx
