"""
DNN Model Factory Functions for Graph Characterization.

This module provides factory functions for creating synthetic DNN models
optimized for Embodied AI workloads.

Models are organized by category:
- yolo: Object detection (YOLOv8, YOLOv11)
- segmentation: Semantic segmentation (DeepLabV3+, SegFormer)
- reid: Object re-identification (OSNet, FastReID)
- classic: Classic CNNs (ResNet, MobileNet, EfficientNet)

All models are FX-traceable and include shape propagation for characterization.
"""

__all__ = [
    # YOLO object detection
    "make_yolov8n",
    "make_yolov8s",
    "make_yolov8m",
    "make_yolov11m",

    # Segmentation
    "make_deeplabv3_mobilenet",
    "make_segformer_b0",

    # Re-identification
    "make_osnet",
    "make_fast_reid",
]
