"""
YOLOv8 Model Training for Capsule Defect Detection
Standalone script - trains the model with optimized hyperparameters
"""

import torch
from pathlib import Path
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

YOLO_DATASET_ROOT = "yolo_dataset"
MODELS_DIR = "models"
MODEL_NAME = "yolov8m.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 150

# Optimized hyperparameters for defect detection
HYPERPARAMS = {
    'lr0': 0.001,  # Initial learning rate
    'lrf': 0.01,  # Final learning rate factor
    'momentum': 0.937,  # SGD momentum
    'weight_decay': 0.0005,  # Optimizer weight decay
    'warmup_epochs': 3.0,  # Warmup epochs
    'warmup_momentum': 0.8,  # Warmup initial momentum
    'box': 7.5,  # Box loss gain
    'cls': 0.5,  # Class loss gain
    'dfl': 1.5,  # DFL loss gain
    'hsv_h': 0.015,  # HSV-Hue augmentation
    'hsv_s': 0.7,  # HSV-Saturation augmentation
    'hsv_v': 0.4,  # HSV-Value augmentation
    'degrees': 10.0,  # Rotation (+/- deg)
    'translate': 0.1,  # Translation (+/- fraction)
    'scale': 0.5,  # Scaling (+/- gain)
    'shear': 0.0,  # Shear (+/- deg)
    'perspective': 0.0,  # Perspective (+/- fraction)
    'flipud': 0.0,  # Vertical flip probability
    'fliplr': 0.5,  # Horizontal flip probability
    'mosaic': 1.0,  # Mosaic augmentation probability
    'mixup': 0.1,  # Mixup augmentation probability
    'copy_paste': 0.0,  # Copy-paste augmentation probability
}


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model():
    """Train YOLOv8 model"""
    print("=" * 80)
    print("YOLOV8 MODEL TRAINING")
    print("=" * 80)

    # Check if data.yaml exists
    data_yaml = Path(YOLO_DATASET_ROOT) / "data.yaml"
    if not data_yaml.exists():
        print(f"\n❌ ERROR: data.yaml not found at {data_yaml}")
        print("\nPlease run data preparation first:")
        print("  python prepare_data.py")
        print("=" * 80)
        return

    # Load pretrained YOLOv8 model
    print(f"\nLoading pretrained model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # Display model info
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {total_params:,}")

    # Training configuration
    print("\n" + "-" * 80)
    print("TRAINING CONFIGURATION")
    print("-" * 80)
    print(f"  Model:           {MODEL_NAME}")
    print(f"  Image size:      {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Device:          {'GPU' if torch.cuda.is_available() else 'CPU'}")

    if torch.cuda.is_available():
        print(f"  GPU Name:        {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  GPU Memory:      {gpu_mem:.1f} GB")
    else:
        print("\n  ⚠️  WARNING: Training on CPU will be VERY slow!")
        print("     Consider using Google Colab or a GPU-enabled system")

    print("-" * 80)

    # Confirm before starting
    print("\nTraining will take several hours depending on your hardware.")
    print("The model will be saved to: models/capsule_defect_detector/weights/")

    # Train the model
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("=" * 80)
    print()

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='capsule_defect_detector',
        project=MODELS_DIR,
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        **HYPERPARAMS
    )

    # Training completed
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETED!")
    print("=" * 80)

    best_model_path = Path(MODELS_DIR) / "capsule_defect_detector" / "weights" / "best.pt"
    last_model_path = Path(MODELS_DIR) / "capsule_defect_detector" / "weights" / "last.pt"
    results_dir = Path(MODELS_DIR) / "capsule_defect_detector"

    print(f"\nModel files saved:")
    print(f"  Best model:      {best_model_path}")
    print(f"  Last checkpoint: {last_model_path}")
    print(f"  Training plots:  {results_dir}/results.png")
    print(f"  All results:     {results_dir}/")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Evaluate the model:")
    print(f"   python evaluate.py --model {best_model_path}")
    print("\n2. Test on a single image:")
    print(f"   python -c \"from ultralytics import YOLO; YOLO('{best_model_path}').predict('test.jpg', save=True)\"")
    print("\n3. Run web demo (if you have the web app):")
    print("   python web/app.py")
    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Capsule Defect Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training (150 epochs, YOLOv8m)
  python train_model.py

  # Custom epochs
  python train_model.py --epochs 200

  # Different model size
  python train_model.py --model yolov8l.pt

  # Fast training for testing
  python train_model.py --model yolov8n.pt --epochs 50

  # Custom batch size (if GPU memory issues)
  python train_model.py --batch-size 8
        """
    )

    parser.add_argument('--model', type=str, default=None,
                        help=f'YOLOv8 model variant (default: {MODEL_NAME})')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--img-size', type=int, default=None,
                        help=f'Image size (default: {IMG_SIZE})')

    args = parser.parse_args()

    # Update config with command line arguments
    if args.model:
        MODEL_NAME = args.model
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.img_size:
        IMG_SIZE = args.img_size

    # Run training
    train_model()