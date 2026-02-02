#!/usr/bin/env bash

# Supported: resnet18/34/50/101/152, mobilenet_v2/v3_small/v3_large, efficientnet_b0/b1/b2, vgg11/16/19, vit_b_16, deeplabv3_resnet50, fcn_resnet50

./cli/analyze_comprehensive.py --model resnet18 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/resnet18.log
./cli/analyze_comprehensive.py --model resnet34 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/resnet34.log
./cli/analyze_comprehensive.py --model resnet50 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/resnet50.log
./cli/analyze_comprehensive.py --model resnet101 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/resnet101.log
./cli/analyze_comprehensive.py --model resnet152 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/resnet152.log

./cli/analyze_comprehensive.py --model mobilenet_v2 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/mobilenet_v2.log
./cli/analyze_comprehensive.py --model mobilenet_v3_small --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/mobilenet_v3_small.log
./cli/analyze_comprehensive.py --model mobilenet_v3_large --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/mobilenet_v3_large.log

./cli/analyze_comprehensive.py --model efficientnet_b0 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/efficientnet_b0.log
./cli/analyze_comprehensive.py --model efficientnet_b1 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/efficientnet_b1.log
./cli/analyze_comprehensive.py --model efficientnet_b2 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/efficientnet_b2.log

./cli/analyze_comprehensive.py --model vgg11 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vgg11.log
./cli/analyze_comprehensive.py --model vgg16 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vgg16.log
./cli/analyze_comprehensive.py --model vgg19 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vgg19.log

./cli/analyze_comprehensive.py --model vit_b_16 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_b_16.log
./cli/analyze_comprehensive.py --model vit_b_32 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_b_32.log
./cli/analyze_comprehensive.py --model vit_h_14 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_h_14.log
./cli/analyze_comprehensive.py --model vit_l_16 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_l_16.log
./cli/analyze_comprehensive.py --model vit_l_32 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/vit_l_32.log
./cli/analyze_comprehensive.py --model maxvit_t --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/maxvit_t.log

./cli/analyze_comprehensive.py --model deeplabv3_resnet50 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/deeplabv3_resnet50.log

./cli/analyze_comprehensive.py --model fcn_resnet50 --hardware Jetson-Orin-AGX > hardware_registry/boards/jetson_orin_agx/analysis/fcn_resnet50.log
