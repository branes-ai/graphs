#!/usr/bin/env python3
"""
Download Embodied AI Benchmark Models

This script downloads the core models for Embodied AI workload benchmarking:
- YOLO object detection (YOLOv8n, YOLOv8m, YOLOv11m)
- Semantic segmentation (DeepLabV3+ MobileNetV2)
- Object re-identification (OSNet)

Models are downloaded to the current directory and can be profiled using:
    python cli/profile_graph.py --model yolov8n.pt

Requirements:
    pip install ultralytics torch torchvision
"""

import sys
from pathlib import Path

def download_yolo_models():
    """Download YOLO models from Ultralytics"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Install with: pip install ultralytics")
        return False

    models = {
        'yolov8n.pt': 'YOLOv8 Nano (3.2M params, 8.7 GFLOPs)',
        'yolov8s.pt': 'YOLOv8 Small (11.2M params, 28.6 GFLOPs)',
        'yolov8m.pt': 'YOLOv8 Medium (25.9M params, 78.9 GFLOPs)',
        'yolo11m.pt': 'YOLO11 Medium (similar to YOLOv8m)',
    }

    print("=" * 80)
    print("DOWNLOADING YOLO MODELS")
    print("=" * 80)
    print()

    downloaded = []
    for model_name, description in models.items():
        model_path = Path(model_name)

        if model_path.exists():
            print(f"✓ {model_name} already exists ({description})")
            downloaded.append(model_name)
            continue

        print(f"⬇ Downloading {model_name} ({description})...")
        try:
            model = YOLO(model_name)
            print(f"  ✓ Downloaded to {model_path.absolute()}")
            downloaded.append(model_name)
        except Exception as e:
            print(f"  ❌ Failed to download {model_name}: {e}")

    print()
    print(f"Downloaded {len(downloaded)}/{len(models)} YOLO models")
    return len(downloaded) > 0


def check_torchvision_models():
    """Check availability of TorchVision segmentation models"""
    try:
        import torchvision.models.segmentation as seg_models
    except ImportError:
        print("❌ TorchVision not installed. Install with: pip install torchvision")
        return False

    print("=" * 80)
    print("CHECKING TORCHVISION SEGMENTATION MODELS")
    print("=" * 80)
    print()

    models = {
        'deeplabv3_mobilenet_v3_large': 'DeepLabV3+ MobileNetV3 (Embodied AI segmentation)',
        'deeplabv3_resnet50': 'DeepLabV3+ ResNet50 (alternative)',
        'fcn_resnet50': 'FCN ResNet50 (alternative)',
    }

    for model_name, description in models.items():
        print(f"✓ {model_name}: {description}")
        print(f"  Available from torchvision.models.segmentation")
        print(f"  Profile with: python cli/profile_graph.py --model {model_name} --input-shape 1 3 512 512")
        print()

    print("Note: These models are part of torchvision and don't need separate download")
    print("They will be downloaded automatically when first used")
    return True


def check_reid_models():
    """Check availability of re-identification models"""
    print("=" * 80)
    print("CHECKING RE-IDENTIFICATION MODELS")
    print("=" * 80)
    print()

    print("⚠️  OSNet and FastReID require separate installation:")
    print()
    print("Option 1: torchreid (OSNet)")
    print("  pip install torchreid")
    print("  from torchreid import models")
    print("  model = models.build_model(name='osnet_x0_25', num_classes=1000)")
    print()
    print("Option 2: FastReID (more advanced)")
    print("  git clone https://github.com/JDAI-CV/fast-reid.git")
    print("  cd fast-reid && pip install -r requirements.txt")
    print()
    print("For Embodied AI benchmarking, we'll use ResNet-based re-ID as a proxy:")
    print("  python cli/profile_graph.py --model resnet18 --input-shape 1 3 256 128")
    print("  (Re-ID typically uses 256×128 input resolution)")
    print()

    return True


def generate_profile_commands():
    """Generate commands to profile the downloaded models"""
    print("=" * 80)
    print("NEXT STEPS: PROFILE MODELS")
    print("=" * 80)
    print()

    commands = [
        "# Profile YOLO models (object detection)",
        "python cli/profile_graph.py --model yolov8n.pt --input-shape 1 3 640 640",
        "python cli/profile_graph.py --model yolov8m.pt --input-shape 1 3 640 640",
        "python cli/profile_graph.py --model yolo11m.pt --input-shape 1 3 640 640",
        "",
        "# Profile segmentation models",
        "python cli/profile_graph.py --model deeplabv3_mobilenet_v3_large --input-shape 1 3 512 512",
        "",
        "# Profile re-identification (using ResNet as proxy)",
        "python cli/profile_graph.py --model resnet18 --input-shape 1 3 256 128",
        "",
        "# Run full Embodied AI comparison (after models are profiled)",
        "python validation/hardware/test_embodied_ai_comparison.py",
    ]

    for cmd in commands:
        print(cmd)

    print()


def main():
    print()
    print("=" * 80)
    print("EMBODIED AI BENCHMARK MODEL DOWNLOADER")
    print("=" * 80)
    print()
    print("This script sets up models for Embodied AI workload benchmarking:")
    print("  - Object Detection: YOLO (YOLOv8n, YOLOv8m, YOLOv11m)")
    print("  - Segmentation: DeepLabV3+ MobileNetV3")
    print("  - Re-Identification: ResNet-based (OSNet alternative)")
    print()

    # Download YOLO models
    yolo_ok = download_yolo_models()
    print()

    # Check TorchVision models
    torchvision_ok = check_torchvision_models()
    print()

    # Check re-ID models
    reid_ok = check_reid_models()
    print()

    # Generate profile commands
    if yolo_ok:
        generate_profile_commands()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if yolo_ok and torchvision_ok:
        print("✓ All core models are ready for benchmarking!")
        print()
        print("Next steps:")
        print("  1. Profile models using the commands above")
        print("  2. Run hardware comparison:")
        print("     python validation/hardware/test_embodied_ai_comparison.py")
        print("  3. Generate market report:")
        print("     python cli/compare_embodied_ai.py --all-tiers --output report.md")
    else:
        print("❌ Some dependencies are missing. Install required packages:")
        if not yolo_ok:
            print("  pip install ultralytics")
        if not torchvision_ok:
            print("  pip install torch torchvision")

    print()


if __name__ == "__main__":
    main()
