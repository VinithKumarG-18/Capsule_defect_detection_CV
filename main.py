"""
YOLOv8 Pharmaceutical Capsule Defect Detection System
Complete implementation from data preparation to production deployment
Target Accuracy: 85-90%
"""

import os
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from collections import defaultdict
import json
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the entire pipeline"""

    # Paths
    DATASET_ROOT = r"C:\Users\Pc\Documents\Computer_Vision_Projects\ISP_AD\dataset"
    YOLO_DATASET_ROOT = "yolo_dataset"
    RESULTS_DIR = "results"
    MODELS_DIR = "models"

    # Classes
    CLASSES = ['good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze']

    # Dataset split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.20
    TEST_RATIO = 0.10

    # YOLOv8 Model Configuration
    MODEL_NAME = "yolov8m.pt"  # Medium model - best balance for defect detection
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

    # Inference configuration
    CONF_THRESHOLD = 0.35  # Confidence threshold
    IOU_THRESHOLD = 0.45  # NMS IoU threshold
    MAX_DETECTIONS = 1  # Max detections per image (1 capsule)


# ============================================================================
# PART 1: DATASET PREPARATION
# ============================================================================

class DatasetPreparator:
    """Prepares dataset in YOLO format with annotations"""

    def __init__(self, config):
        self.config = config
        self.class_to_idx = {cls: idx for idx, cls in enumerate(config.CLASSES)}

    def create_directory_structure(self):
        """Create YOLO dataset directory structure"""
        for split in ['train', 'val', 'test']:
            os.makedirs(f"{self.config.YOLO_DATASET_ROOT}/images/{split}", exist_ok=True)
            os.makedirs(f"{self.config.YOLO_DATASET_ROOT}/labels/{split}", exist_ok=True)

    def get_image_files(self):
        """Collect all image files with their labels"""
        image_data = []

        # Good capsules
        good_dir = Path(self.config.DATASET_ROOT) / "good_images"
        if good_dir.exists():
            for img_path in good_dir.glob("*.[jp][pn]g"):
                image_data.append({
                    'path': str(img_path),
                    'class': 'good',
                    'class_idx': self.class_to_idx['good']
                })

        # Defective capsules
        defective_dir = Path(self.config.DATASET_ROOT) / "defective_images"
        if defective_dir.exists():
            for defect_type in ['crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze']:
                defect_path = defective_dir / defect_type
                if defect_path.exists():
                    for img_path in defect_path.glob("*.[jp][pn]g"):
                        image_data.append({
                            'path': str(img_path),
                            'class': defect_type,
                            'class_idx': self.class_to_idx[defect_type]
                        })

        return image_data

    def create_yolo_annotation(self, img_path, class_idx):
        """
        Create YOLO annotation (normalized bbox for entire capsule)
        Format: class_id center_x center_y width height (normalized 0-1)
        """
        img = cv2.imread(img_path)
        if img is None:
            return None

        h, w = img.shape[:2]

        # Use entire image as bounding box (capsule fills most of the image)
        # Add small margin (5%) for better detection
        margin = 0.05
        center_x = 0.5
        center_y = 0.5
        bbox_w = 1.0 - (2 * margin)
        bbox_h = 1.0 - (2 * margin)

        return f"{class_idx} {center_x} {center_y} {bbox_w} {bbox_h}\n"

    def prepare_dataset(self):
        """Main dataset preparation pipeline"""
        print("=" * 80)
        print("DATASET PREPARATION")
        print("=" * 80)

        # Create directory structure
        self.create_directory_structure()

        # Get all images
        image_data = self.get_image_files()
        print(f"\nTotal images found: {len(image_data)}")

        # Print class distribution
        class_counts = defaultdict(int)
        for item in image_data:
            class_counts[item['class']] += 1

        print("\nClass distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls:20s}: {count:4d} images")

        # Split dataset: train/val/test
        train_data, temp_data = train_test_split(
            image_data,
            test_size=(self.config.VAL_RATIO + self.config.TEST_RATIO),
            random_state=42,
            stratify=[d['class'] for d in image_data]
        )

        val_data, test_data = train_test_split(
            temp_data,
            test_size=self.config.TEST_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO),
            random_state=42,
            stratify=[d['class'] for d in temp_data]
        )

        print(f"\nDataset split:")
        print(f"  Train: {len(train_data)} images ({len(train_data) / len(image_data) * 100:.1f}%)")
        print(f"  Val:   {len(val_data)} images ({len(val_data) / len(image_data) * 100:.1f}%)")
        print(f"  Test:  {len(test_data)} images ({len(test_data) / len(image_data) * 100:.1f}%)")

        # Copy images and create annotations
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        for split_name, split_data in splits.items():
            print(f"\nProcessing {split_name} split...")
            for idx, item in enumerate(split_data):
                # Copy image
                src_path = item['path']
                img_name = f"{split_name}_{idx:04d}.jpg"
                dst_img_path = f"{self.config.YOLO_DATASET_ROOT}/images/{split_name}/{img_name}"
                shutil.copy2(src_path, dst_img_path)

                # Create annotation
                annotation = self.create_yolo_annotation(src_path, item['class_idx'])
                if annotation:
                    label_path = f"{self.config.YOLO_DATASET_ROOT}/labels/{split_name}/{img_name.replace('.jpg', '.txt')}"
                    with open(label_path, 'w') as f:
                        f.write(annotation)

        # Create data.yaml
        self.create_yaml_config()

        print("\n✓ Dataset preparation completed successfully!")
        return len(train_data), len(val_data), len(test_data)

    def create_yaml_config(self):
        """Create YOLO data configuration file"""
        data_config = {
            'path': os.path.abspath(self.config.YOLO_DATASET_ROOT),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.config.CLASSES),
            'names': self.config.CLASSES
        }

        yaml_path = f"{self.config.YOLO_DATASET_ROOT}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"\n✓ Created data.yaml at {yaml_path}")


# ============================================================================
# PART 2: MODEL TRAINING
# ============================================================================

class ModelTrainer:
    """Train YOLOv8 model with optimized hyperparameters"""

    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self):
        """Train YOLOv8 model"""
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)

        # Load pretrained YOLOv8 model
        print(f"\nLoading {self.config.MODEL_NAME}...")
        self.model = YOLO(self.config.MODEL_NAME)

        # Display model info
        print(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")

        # Training configuration
        print("\nTraining configuration:")
        print(f"  Image size: {self.config.IMG_SIZE}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Epochs: {self.config.EPOCHS}")
        print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

        # Train the model
        print("\nStarting training...")
        results = self.model.train(
            data=f"{self.config.YOLO_DATASET_ROOT}/data.yaml",
            epochs=self.config.EPOCHS,
            imgsz=self.config.IMG_SIZE,
            batch=self.config.BATCH_SIZE,
            name='capsule_defect_detector',
            project=self.config.MODELS_DIR,
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
            # Hyperparameters
            **self.config.HYPERPARAMS
        )

        print("\n✓ Training completed!")

        # Save best model
        best_model_path = f"{self.config.MODELS_DIR}/capsule_defect_detector/weights/best.pt"
        print(f"\n✓ Best model saved at: {best_model_path}")

        return results


# ============================================================================
# PART 3: MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation with detailed metrics"""

    def __init__(self, config, model_path):
        self.config = config
        self.model = YOLO(model_path)
        self.results_dir = Path(config.RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)

    def evaluate(self):
        """Run comprehensive evaluation"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)

        # Validate on test set
        print("\nEvaluating on test set...")
        metrics = self.model.val(
            data=f"{self.config.YOLO_DATASET_ROOT}/data.yaml",
            split='test',
            imgsz=self.config.IMG_SIZE,
            batch=self.config.BATCH_SIZE,
            conf=self.config.CONF_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=True
        )

        # Extract metrics
        results = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'per_class_AP50': {},
            'per_class_AP50-95': {}
        }

        # Per-class metrics
        for idx, class_name in enumerate(self.config.CLASSES):
            results['per_class_AP50'][class_name] = float(metrics.box.maps50[idx])
            results['per_class_AP50-95'][class_name] = float(metrics.box.maps[idx])

        # Print results
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"\nOverall Metrics:")
        print(f"  mAP@0.5:      {results['mAP50']:.3f} ({results['mAP50'] * 100:.1f}%)")
        print(f"  mAP@0.5:0.95: {results['mAP50-95']:.3f} ({results['mAP50-95'] * 100:.1f}%)")
        print(f"  Precision:    {results['precision']:.3f} ({results['precision'] * 100:.1f}%)")
        print(f"  Recall:       {results['recall']:.3f} ({results['recall'] * 100:.1f}%)")

        print(f"\nPer-Class AP@0.5:")
        for class_name, ap in results['per_class_AP50'].items():
            print(f"  {class_name:20s}: {ap:.3f} ({ap * 100:.1f}%)")

        # Save results
        results_file = self.results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Check if target accuracy achieved
        target_achieved = results['mAP50'] >= 0.85
        print(
            f"\n{'✓' if target_achieved else '✗'} Target accuracy (85-90%): {'ACHIEVED' if target_achieved else 'NOT ACHIEVED'}")

        return results

    def visualize_predictions(self, num_samples=10):
        """Visualize predictions on test images"""
        print(f"\nGenerating prediction visualizations...")

        test_img_dir = Path(self.config.YOLO_DATASET_ROOT) / "images" / "test"
        test_images = list(test_img_dir.glob("*.jpg"))[:num_samples]

        vis_dir = self.results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for img_path in test_images:
            results = self.model.predict(
                source=str(img_path),
                conf=self.config.CONF_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                max_det=self.config.MAX_DETECTIONS,
                save=False,
                verbose=False
            )

            # Save annotated image
            annotated = results[0].plot()
            cv2.imwrite(str(vis_dir / img_path.name), annotated)

        print(f"✓ Visualizations saved to {vis_dir}")


# ============================================================================
# PART 4: PRODUCTION INFERENCE SYSTEM
# ============================================================================

class CapsuleDefectDetector:
    """Production-ready inference system"""

    def __init__(self, model_path, config):
        self.model = YOLO(model_path)
        self.config = config
        self.class_names = config.CLASSES

    def detect(self, image_path):
        """
        Detect defects in a single capsule image
        Returns: dict with detection results
        """
        results = self.model.predict(
            source=image_path,
            conf=self.config.CONF_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=False
        )[0]

        # Parse results
        detection = {
            'image_path': image_path,
            'detected': False,
            'class': None,
            'confidence': 0.0,
            'bbox': None,
            'status': 'No detection'
        }

        if len(results.boxes) > 0:
            box = results.boxes[0]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()

            detection.update({
                'detected': True,
                'class': self.class_names[class_id],
                'confidence': confidence,
                'bbox': bbox,
                'status': 'GOOD' if self.class_names[class_id] == 'good' else 'DEFECTIVE'
            })

        return detection

    def detect_batch(self, image_paths):
        """Batch detection for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.detect(img_path)
            results.append(result)
        return results

    def visualize_detection(self, image_path, output_path=None):
        """Visualize detection with bounding box and label"""
        detection = self.detect(image_path)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if detection['detected']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box
            color = (0, 255, 0) if detection['class'] == 'good' else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            # Add label
            label = f"{detection['class']} {detection['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return img, detection


# ============================================================================
# PART 5: WEB INTERFACE (Flask)
# ============================================================================

class WebInterface:
    """Flask-based web interface for capsule defect detection"""

    @staticmethod
    def create_app(model_path, config):
        """Create Flask application"""
        from flask import Flask, render_template_string, request, jsonify
        import base64
        from io import BytesIO

        app = Flask(__name__)
        detector = CapsuleDefectDetector(model_path, config)

        HTML_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Capsule Defect Detection System</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    padding: 40px;
                }
                h1 {
                    color: #667eea;
                    text-align: center;
                    margin-bottom: 10px;
                    font-size: 2.5em;
                }
                .subtitle {
                    text-align: center;
                    color: #666;
                    margin-bottom: 40px;
                }
                .upload-section {
                    border: 3px dashed #667eea;
                    border-radius: 15px;
                    padding: 40px;
                    text-align: center;
                    background: #f8f9ff;
                    margin-bottom: 30px;
                    cursor: pointer;
                    transition: all 0.3s;
                }
                .upload-section:hover {
                    background: #e8ebff;
                    border-color: #764ba2;
                }
                .upload-btn {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px 40px;
                    border: none;
                    border-radius: 30px;
                    font-size: 1.1em;
                    cursor: pointer;
                    transition: transform 0.2s;
                }
                .upload-btn:hover {
                    transform: scale(1.05);
                }
                .results {
                    display: none;
                    margin-top: 30px;
                }
                .result-card {
                    background: #f8f9ff;
                    border-radius: 15px;
                    padding: 30px;
                    margin-top: 20px;
                }
                .status-good {
                    color: #10b981;
                    font-size: 1.5em;
                    font-weight: bold;
                }
                .status-defect {
                    color: #ef4444;
                    font-size: 1.5em;
                    font-weight: bold;
                }
                .metric {
                    display: inline-block;
                    margin: 10px 20px;
                    padding: 15px 30px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .metric-label {
                    color: #666;
                    font-size: 0.9em;
                }
                .metric-value {
                    color: #667eea;
                    font-size: 1.8em;
                    font-weight: bold;
                }
                img {
                    max-width: 100%;
                    border-radius: 10px;
                    margin-top: 20px;
                }
                .loading {
                    display: none;
                    text-align: center;
                    padding: 20px;
                }
                .spinner {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #667eea;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 Capsule Defect Detection System</h1>
                <p class="subtitle">YOLOv8-powered Quality Control | 6 Defect Classes</p>

                <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                    <h2>📤 Upload Capsule Image</h2>
                    <p style="margin: 20px 0; color: #666;">Click to select or drag and drop</p>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
                    <button class="upload-btn">Choose Image</button>
                </div>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 20px; color: #667eea;">Analyzing capsule...</p>
                </div>

                <div class="results" id="results">
                    <div class="result-card">
                        <h2>Detection Results</h2>
                        <div id="statusDisplay"></div>
                        <div id="metricsDisplay"></div>
                        <div id="imageDisplay"></div>
                    </div>
                </div>
            </div>

            <script>
                function uploadImage() {
                    const fileInput = document.getElementById('fileInput');
                    const file = fileInput.files[0];

                    if (!file) return;

                    const formData = new FormData();
                    formData.append('image', file);

                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('results').style.display = 'none';

                    fetch('/detect', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('loading').style.display = 'none';
                        displayResults(data);
                    })
                    .catch(error => {
                        document.getElementById('loading').style.display = 'none';
                        alert('Error: ' + error);
                    });
                }

                function displayResults(data) {
                    const statusClass = data.status === 'GOOD' ? 'status-good' : 'status-defect';
                    const statusIcon = data.status === 'GOOD' ? '✅' : '❌';

                    document.getElementById('statusDisplay').innerHTML = `
                        <p class="${statusClass}">${statusIcon} ${data.status}</p>
                    `;

                    document.getElementById('metricsDisplay').innerHTML = `
                        <div class="metric">
                            <div class="metric-label">Class</div>
                            <div class="metric-value">${data.class}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value">${(data.confidence * 100).toFixed(1)}%</div>
                        </div>
                    `;

                    document.getElementById('imageDisplay').innerHTML = `
                        <img src="data:image/jpeg;base64,${data.image}" alt="Detection Result">
                    `;

                    document.getElementById('results').style.display = 'block';
                }
            </script>
        </body>
        </html>
        """

        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)

        @app.route('/detect', methods=['POST'])
        def detect():
            if 'image' not in request.files:
                return jsonify({'error': 'No image uploaded'}), 400

            file = request.files['image']

            # Save temporarily
            temp_path = 'temp_upload.jpg'
            file.save(temp_path)

            # Run detection
            img, detection = detector.visualize_detection(temp_path)

            # Convert to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Clean up
            os.remove(temp_path)

            return jsonify({
                'status': detection['status'],
                'class': detection['class'] or 'Unknown',
                'confidence': detection['confidence'],
                'image': img_base64
            })

        return app


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    print("=" * 80)
    print("YOLOv8 PHARMACEUTICAL CAPSULE DEFECT DETECTION SYSTEM")
    print("=" * 80)
    print(f"\nModel: {Config.MODEL_NAME}")
    print(f"Target Accuracy: 85-90%")
    print(f"Classes: {', '.join(Config.CLASSES)}")

    # Step 1: Prepare dataset
    preparator = DatasetPreparator(Config)
    train_size, val_size, test_size = preparator.prepare_dataset()

    # Step 2: Train model
    trainer = ModelTrainer(Config)
    trainer.train()

    # Step 3: Evaluate model
    best_model_path = f"{Config.MODELS_DIR}/capsule_defect_detector/weights/best.pt"
    evaluator = ModelEvaluator(Config, best_model_path)
    results = evaluator.evaluate()
    evaluator.visualize_predictions(num_samples=10)

    # Step 4: Production inference demo
    print("\n" + "=" * 80)
    print("PRODUCTION INFERENCE SYSTEM")
    print("=" * 80)

    detector = CapsuleDefectDetector(best_model_path, Config)

    # Test on sample images
    test_img_dir = Path(Config.YOLO_DATASET_ROOT) / "images" / "test"
    test_images = list(test_img_dir.glob("*.jpg"))[:5]

    print("\nRunning inference on sample images...")
    for img_path in test_images:
        detection = detector.detect(str(img_path))
        print(f"\n{img_path.name}:")
        print(f"  Status: {detection['status']}")
        print(f"  Class: {detection['class']}")
        print(f"  Confidence: {detection['confidence']:.2%}")

    # Step 5: Launch web interface
    print("\n" + "=" * 80)
    print("WEB INTERFACE")
    print("=" * 80)
    print("\nTo launch the web interface, run:")
    print("  python capsule_detection.py --web")
    print("\nOr use the detector programmatically:")
    print("  detector = CapsuleDefectDetector('models/capsule_defect_detector/weights/best.pt', Config)")
    print("  result = detector.detect('path/to/image.jpg')")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n✓ Model saved: {best_model_path}")
    print(f"✓ Accuracy: {results['mAP50'] * 100:.1f}%")
    print(f"✓ Ready for production deployment")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        # Launch web interface
        print("Launching web interface...")
        best_model_path = f"{Config.MODELS_DIR}/capsule_defect_detector/weights/best.pt"
        app = WebInterface.create_app(best_model_path, Config)
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        # Run full pipeline
        main()