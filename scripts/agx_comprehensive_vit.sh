#!/usr/bin/env bash

# Supported: resnet18/34/50/101/152, mobilenet_v2/v3_small/v3_large, efficientnet_b0/b1/b2, vgg11/16/19, vit_b_16, deeplabv3_resnet50, fcn_resnet50


#./cli/analyze_comprehensive.py --model vit_b_16 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_b_16.log
./cli/analyze_comprehensive.py --model vit_b_32 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_b_32.log
./cli/analyze_comprehensive.py --model vit_h_14 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_h_14.log
./cli/analyze_comprehensive.py --model vit_l_16 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_l_16.log
./cli/analyze_comprehensive.py --model vit_l_32 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_l_32.log
./cli/analyze_comprehensive.py --model maxvit_t --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/maxvit_t.log

