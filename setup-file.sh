#!/bin/bash

# Rebuild the model chunks

# LPR MODEL
cat asset/model/lpr/lpr_accurate_chunks/pt/* > asset/model/lpr/lpr_accurate.pt
cat asset/model/lpr/lpr_accurate_chunks/onnx/* > asset/model/lpr/lpr_accurate.onnx
cat asset/model/lpr/lpr_fast_chunks/pt/* > asset/model/lpr/lpr_fast.pt
cat asset/model/lpr/lpr_fast_chunks/onnx/* > asset/model/lpr/lpr_fast.onnx

# YOLO MODEL
cat asset/model/yolo/yolov8l_chunks/pt/* > asset/model/yolo/yolov8l.pt
cat asset/model/yolo/yolov8l_chunks/onnx/* > asset/model/yolo/yolov8l.onnx
