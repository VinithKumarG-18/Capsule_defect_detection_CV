"""
Dataset Preparation for YOLOv8 Capsule Defect Detection
Standalone script - converts dataset to YOLO format with train/val/test splits
"""

import os
import shutil
import yaml
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

DATASET_ROOT = r"C:\Users\Pc\Documents\Computer_Vision_Projects\ISP_AD\dataset"
YOLO_DATASET_ROOT = "yolo_dataset"
CLASSES = ['good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze']
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


# ============================================================================
# FUNCTIONS
# ============================================================================

def create_directory_structure():
    """Create YOLO dataset directory structure"""
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{YOLO_DATASET_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{YOLO_DATASET_ROOT}/labels/{split}", exist_ok=True)
    print("✓ Created directory structure")


def get_image_files():
    """Collect all image files with their labels"""
    image_data = []
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

    # Good capsules
    good_dir = Path(DATASET_ROOT) / "good_images"
    if good_dir.exists():
        for img_path in good_dir.glob("*.[jp][pn]g"):
            image_data.append({
                'path': str(img_path),
                'class': 'good',
                'class_idx': class_to_idx['good']
            })

    # Defective capsules
    defective_dir = Path(DATASET_ROOT) / "defective_images"
    if defective_dir.exists():
        for defect_type in ['crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze']:
            defect_path = defective_dir / defect_type
            if defect_path.exists():
                for img_path in defect_path.glob("*.[jp][pn]g"):
                    image_data.append({
                        'path': str(img_path),
                        'class': defect_type,
                        'class_idx': class_to_idx[defect_type]
                    })

    return image_data


def create_yolo_annotation(img_path, class_idx):
    """
    Create YOLO annotation (normalized bbox for entire capsule)
    Format: class_id center_x center_y width height (normalized 0-1)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Use entire image as bounding box with small margin
    margin = 0.05
    center_x = 0.5
    center_y = 0.5
    bbox_w = 1.0 - (2 * margin)
    bbox_h = 1.0 - (2 * margin)

    return f"{class_idx} {center_x} {center_y} {bbox_w} {bbox_h}\n"


def create_yaml_config():
    """Create YOLO data configuration file"""
    data_config = {
        'path': os.path.abspath(YOLO_DATASET_ROOT),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASSES),
        'names': CLASSES
    }

    yaml_path = f"{YOLO_DATASET_ROOT}/data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"✓ Created data.yaml at {yaml_path}")


def prepare_dataset():
    """Main dataset preparation pipeline"""
    print("=" * 80)
    print("DATASET PREPARATION FOR YOLO")
    print("=" * 80)

    # Create directory structure
    create_directory_structure()

    # Get all images
    print("\nScanning for images...")
    image_data = get_image_files()
    print(f"✓ Total images found: {len(image_data)}")

    if len(image_data) == 0:
        print("\n❌ ERROR: No images found!")
        print(f"Please check if DATASET_ROOT path is correct: {DATASET_ROOT}")
        print("\nExpected structure:")
        print("  dataset/")
        print("    ├── good_images/")
        print("    │   ├── image1.jpg")
        print("    │   └── ...")
        print("    └── defective_images/")
        print("        ├── crack/")
        print("        ├── faulty_imprint/")
        print("        ├── poke/")
        print("        ├── scratch/")
        print("        └── squeeze/")
        return

    # Print class distribution
    class_counts = defaultdict(int)
    for item in image_data:
        class_counts[item['class']] += 1

    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:20s}: {count:4d} images")

    # Split dataset: train/val/test
    print("\nSplitting dataset...")
    train_data, temp_data = train_test_split(
        image_data,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42,
        stratify=[d['class'] for d in image_data]
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42,
        stratify=[d['class'] for d in temp_data]
    )

    print(f"\n✓ Dataset split:")
    print(f"    Train: {len(train_data)} images ({len(train_data) / len(image_data) * 100:.1f}%)")
    print(f"    Val:   {len(val_data)} images ({len(val_data) / len(image_data) * 100:.1f}%)")
    print(f"    Test:  {len(test_data)} images ({len(test_data) / len(image_data) * 100:.1f}%)")

    # Copy images and create annotations
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split_name, split_data in splits.items():
        print(f"\n  Processing {split_name} split...")
        for idx, item in enumerate(split_data):
            # Copy image
            src_path = item['path']
            img_name = f"{split_name}_{idx:04d}.jpg"
            dst_img_path = f"{YOLO_DATASET_ROOT}/images/{split_name}/{img_name}"
            shutil.copy2(src_path, dst_img_path)

            # Create annotation
            annotation = create_yolo_annotation(src_path, item['class_idx'])
            if annotation:
                label_path = f"{YOLO_DATASET_ROOT}/labels/{split_name}/{img_name.replace('.jpg', '.txt')}"
                with open(label_path, 'w') as f:
                    f.write(annotation)

        print(f"    ✓ Processed {len(split_data)} images")

    # Create data.yaml
    create_yaml_config()

    print("\n" + "=" * 80)
    print("✓ DATASET PREPARATION COMPLETED!")
    print("=" * 80)
    print(f"\nDataset location: {os.path.abspath(YOLO_DATASET_ROOT)}")
    print("\nNext step: Run training")
    print("  python train_model.py")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    prepare_dataset()