#!/bin/bash

pip install -r requirements.txt

pip install ultralytics
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install onnx
pip install nvidia-tensorrt==8.4.1.5
pip install easyocr
pip install opencv-python==4.8.0.74
pip install dblur
pip install numpy==1.23.5

