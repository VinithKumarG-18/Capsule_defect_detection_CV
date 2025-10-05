"""
Fixed Model Evaluation Script - Complete Version
Evaluates trained YOLOv8 model with comprehensive metrics
Fixed: UTF-8 encoding for emoji support on Windows
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class CapsuleModelEvaluator:
    """
    Complete evaluation system for capsule defect detection model
    Fixed to work with current Ultralytics API and Windows encoding
    """

    def __init__(self, model_path, config):
        """Initialize evaluator"""
        self.model_path = model_path
        self.config = config
        self.model = YOLO(model_path)
        self.results_dir = Path(config.RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)

        print(f"Loaded model from: {model_path}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")

    def evaluate_on_test_set(self):
        """Run validation on test set - FIXED for current Ultralytics API"""
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)

        # Run validation
        metrics = self.model.val(
            data=f"{self.config.YOLO_DATASET_ROOT}/data.yaml",
            split='test',
            imgsz=self.config.IMG_SIZE,
            batch=self.config.BATCH_SIZE,
            conf=self.config.CONF_THRESHOLD,
            iou=self.config.IOU_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=True,
            save_json=False,
            plots=True
        )

        # Extract metrics using CORRECT attributes
        results = {
            'model_path': str(self.model_path),
            'evaluation_date': datetime.now().isoformat(),
            'dataset_info': {
                'test_images': len(list(Path(self.config.YOLO_DATASET_ROOT).glob('images/test/*.jpg'))),
                'classes': self.config.CLASSES
            },
            'overall_metrics': {
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
                'f1_score': float(2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6))
            },
            'per_class_metrics': {}
        }

        # Per-class metrics using CORRECT attributes
        for idx, class_name in enumerate(self.config.CLASSES):
            results['per_class_metrics'][class_name] = {
                'AP50': float(metrics.box.ap50[idx]),
                'AP50-95': float(metrics.box.ap[idx]),
                'precision': float(metrics.box.p[idx]) if len(metrics.box.p) > idx else 0.0,
                'recall': float(metrics.box.r[idx]) if len(metrics.box.r) > idx else 0.0,
            }

        return results, metrics

    def print_results(self, results):
        """Print formatted evaluation results"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        # Overall metrics
        print("\nOverall Performance:")
        print(f"  mAP@0.5:      {results['overall_metrics']['mAP50']:.3f} ({results['overall_metrics']['mAP50']*100:.1f}%)")
        print(f"  mAP@0.5:0.95: {results['overall_metrics']['mAP50-95']:.3f} ({results['overall_metrics']['mAP50-95']*100:.1f}%)")
        print(f"  Precision:    {results['overall_metrics']['precision']:.3f} ({results['overall_metrics']['precision']*100:.1f}%)")
        print(f"  Recall:       {results['overall_metrics']['recall']:.3f} ({results['overall_metrics']['recall']*100:.1f}%)")
        print(f"  F1 Score:     {results['overall_metrics']['f1_score']:.3f} ({results['overall_metrics']['f1_score']*100:.1f}%)")

        # Per-class metrics
        print("\nPer-Class Performance (AP@0.5):")
        for class_name, metrics in results['per_class_metrics'].items():
            status = "[GOOD]" if metrics['AP50'] >= 0.85 else "[WARN]" if metrics['AP50'] >= 0.70 else "[FAIL]"
            print(f"  {status} {class_name:20s}: {metrics['AP50']:.3f} ({metrics['AP50']*100:.1f}%)")

        # Target assessment
        print("\nTarget Accuracy Assessment:")
        target_met = results['overall_metrics']['mAP50'] >= 0.85
        if target_met:
            print(f"  [PASS] TARGET ACHIEVED! {results['overall_metrics']['mAP50']*100:.1f}% >= 85%")
        else:
            gap = (0.85 - results['overall_metrics']['mAP50']) * 100
            print(f"  [FAIL] Target not met. Gap: {gap:.1f}% points")

        # Problem classes
        problem_classes = [
            name for name, m in results['per_class_metrics'].items()
            if m['AP50'] < 0.70
        ]

        if problem_classes:
            print("\nClasses Needing Attention:")
            for cls in problem_classes:
                ap = results['per_class_metrics'][cls]['AP50']
                print(f"  - {cls}: {ap*100:.1f}% (needs improvement)")

    def create_confusion_matrix(self, save_path=None):
        """Generate confusion matrix visualization"""
        print("\nGenerating confusion matrix...")

        # Run predictions on test set
        test_dir = Path(self.config.YOLO_DATASET_ROOT) / "images" / "test"
        test_images = list(test_dir.glob("*.jpg"))

        # Initialize confusion matrix
        n_classes = len(self.config.CLASSES)
        confusion = np.zeros((n_classes, n_classes), dtype=int)

        # Get ground truth and predictions
        for img_path in test_images:
            # Get ground truth label
            label_path = str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt')
            label_path = label_path.replace('\\images\\', '\\labels\\')

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    gt_class = int(f.readline().split()[0])

                # Get prediction
                results = self.model.predict(
                    source=str(img_path),
                    conf=self.config.CONF_THRESHOLD,
                    verbose=False
                )[0]

                if len(results.boxes) > 0:
                    pred_class = int(results.boxes[0].cls[0])
                else:
                    pred_class = 0  # Default to 'good' if no detection

                confusion[gt_class][pred_class] += 1

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config.CLASSES,
            yticklabels=self.config.CLASSES,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path is None:
            save_path = self.results_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to: {save_path}")

        return confusion

    def visualize_predictions(self, num_samples=10):
        """Create visual comparison of predictions"""
        print(f"\nGenerating prediction visualizations...")

        test_img_dir = Path(self.config.YOLO_DATASET_ROOT) / "images" / "test"
        test_images = list(test_img_dir.glob("*.jpg"))[:num_samples]

        vis_dir = self.results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for img_path in test_images:
            # Run prediction
            results = self.model.predict(
                source=str(img_path),
                conf=self.config.CONF_THRESHOLD,
                iou=self.config.IOU_THRESHOLD,
                max_det=self.config.MAX_DETECTIONS,
                save=False,
                verbose=False
            )[0]

            # Plot and save
            annotated = results.plot()
            output_path = vis_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(output_path), annotated)

        print(f"{len(test_images)} visualizations saved to: {vis_dir}")

    def generate_classification_report(self, results):
        """Generate detailed classification report - FIXED: UTF-8 encoding"""
        report_path = self.results_dir / f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # FIX: Use UTF-8 encoding to support all Unicode characters
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CAPSULE DEFECT DETECTION - CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Model: {self.model_path}\n")
            f.write(f"Evaluation Date: {results['evaluation_date']}\n")
            f.write(f"Test Images: {results['dataset_info']['test_images']}\n\n")

            f.write("="*80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("="*80 + "\n")
            for metric, value in results['overall_metrics'].items():
                f.write(f"{metric:20s}: {value:.4f} ({value*100:.2f}%)\n")

            f.write("\n" + "="*80 + "\n")
            f.write("PER-CLASS METRICS\n")
            f.write("="*80 + "\n\n")

            f.write(f"{'Class':<20} {'AP@0.5':<12} {'AP@0.5:0.95':<15} {'Precision':<12} {'Recall':<12}\n")
            f.write("-"*80 + "\n")

            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"{class_name:<20} ")
                f.write(f"{metrics['AP50']:.3f} ({metrics['AP50']*100:.1f}%)  ")
                f.write(f"{metrics['AP50-95']:.3f} ({metrics['AP50-95']*100:.1f}%)  ")
                f.write(f"{metrics['precision']:.3f}        ")
                f.write(f"{metrics['recall']:.3f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")

            # Generate recommendations
            if results['overall_metrics']['mAP50'] >= 0.85:
                f.write("[PASS] Model meets target accuracy (85-90%)\n")
                f.write("   - Ready for production deployment\n")
                f.write("   - Consider A/B testing in real environment\n")
                f.write("   - Monitor performance on new data\n\n")
            else:
                f.write("[ACTION NEEDED] Model below target accuracy\n")
                f.write("   - Collect more training data for weak classes\n")
                f.write("   - Try larger model (YOLOv8l or YOLOv8x)\n")
                f.write("   - Increase training epochs to 200-300\n")
                f.write("   - Apply stronger data augmentation\n\n")

            # Class-specific recommendations
            problem_classes = [
                (name, m['AP50']) for name, m in results['per_class_metrics'].items()
                if m['AP50'] < 0.70
            ]

            if problem_classes:
                f.write("Classes requiring immediate attention:\n\n")
                for cls, ap in problem_classes:
                    f.write(f"  {cls.upper()} - {ap*100:.1f}% AP\n")

                    if ap == 0.0:
                        f.write(f"    [CRITICAL] Model cannot detect this class at all!\n")
                        f.write(f"    -> Collect 100+ high-quality examples\n")
                        f.write(f"    -> Verify annotation quality\n")
                        f.write(f"    -> Check if class is visually distinct\n")
                    elif ap < 0.50:
                        f.write(f"    [URGENT] Very poor performance\n")
                        f.write(f"    -> Collect 50-100 more examples\n")
                        f.write(f"    -> Review existing annotations\n")
                    else:
                        f.write(f"    [MODERATE] Needs improvement\n")
                        f.write(f"    -> Collect 30-50 more examples\n")

                    f.write(f"    -> Ensure diverse lighting/angles\n")
                    f.write(f"    -> Check for class imbalance\n\n")

            # Additional analysis
            f.write("="*80 + "\n")
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("="*80 + "\n\n")

            # Calculate class balance
            total_ap = sum(m['AP50'] for m in results['per_class_metrics'].values())
            avg_ap = total_ap / len(results['per_class_metrics'])

            f.write(f"Average Class AP: {avg_ap:.3f} ({avg_ap*100:.1f}%)\n")
            f.write(f"Overall mAP:      {results['overall_metrics']['mAP50']:.3f} ({results['overall_metrics']['mAP50']*100:.1f}%)\n")
            f.write(f"Std Deviation:    {np.std([m['AP50'] for m in results['per_class_metrics'].values()]):.3f}\n\n")

            # Identify strengths
            strong_classes = [
                (name, m['AP50']) for name, m in results['per_class_metrics'].items()
                if m['AP50'] >= 0.90
            ]

            if strong_classes:
                f.write("Strong performing classes:\n")
                for cls, ap in strong_classes:
                    f.write(f"  - {cls}: {ap*100:.1f}%\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write("NEXT STEPS\n")
            f.write("="*80 + "\n\n")

            if results['overall_metrics']['mAP50'] >= 0.85:
                f.write("1. Deploy model to staging environment\n")
                f.write("2. Run inference on real production samples\n")
                f.write("3. Monitor false positive/negative rates\n")
                f.write("4. Collect edge cases for continuous improvement\n")
            else:
                f.write("1. Focus on improving weak classes (see above)\n")
                f.write("2. Collect more diverse training data\n")
                f.write("3. Consider transfer learning from similar domain\n")
                f.write("4. Re-train with improved dataset\n")
                f.write("5. Validate on larger test set before deployment\n")

        print(f"Classification report saved to: {report_path}")

    def generate_detailed_analysis(self, results):
        """Generate detailed performance analysis with charts"""
        analysis_dir = self.results_dir / "detailed_analysis"
        analysis_dir.mkdir(exist_ok=True)

        # 1. Per-class AP comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        classes = list(results['per_class_metrics'].keys())
        ap50_values = [results['per_class_metrics'][c]['AP50'] for c in classes]
        ap50_95_values = [results['per_class_metrics'][c]['AP50-95'] for c in classes]

        # AP@0.5 bar chart
        colors = ['green' if v >= 0.85 else 'orange' if v >= 0.70 else 'red' for v in ap50_values]
        ax1.barh(classes, ap50_values, color=colors, alpha=0.7)
        ax1.axvline(x=0.85, color='green', linestyle='--', label='Target (85%)')
        ax1.axvline(x=0.70, color='orange', linestyle='--', label='Warning (70%)')
        ax1.set_xlabel('AP@0.5', fontsize=12)
        ax1.set_title('Per-Class Performance (AP@0.5)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)

        # AP@0.5:0.95 bar chart
        colors2 = ['green' if v >= 0.70 else 'orange' if v >= 0.50 else 'red' for v in ap50_95_values]
        ax2.barh(classes, ap50_95_values, color=colors2, alpha=0.7)
        ax2.set_xlabel('AP@0.5:0.95', fontsize=12)
        ax2.set_title('Per-Class Performance (AP@0.5:0.95)', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(analysis_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Overall metrics radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        metrics_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1 Score']
        metrics_values = [
            results['overall_metrics']['mAP50'],
            results['overall_metrics']['mAP50-95'],
            results['overall_metrics']['precision'],
            results['overall_metrics']['recall'],
            results['overall_metrics']['f1_score']
        ]

        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        metrics_values += metrics_values[:1]
        angles += angles[:1]

        ax.plot(angles, metrics_values, 'o-', linewidth=2, color='#667eea')
        ax.fill(angles, metrics_values, alpha=0.25, color='#667eea')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(analysis_dir / 'overall_metrics_radar.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Detailed analysis charts saved to: {analysis_dir}")

    def run_complete_evaluation(self):
        """Run all evaluation steps"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)

        # 1. Evaluate on test set
        results, metrics = self.evaluate_on_test_set()

        # 2. Print results
        self.print_results(results)

        # 3. Save JSON results
        json_path = self.results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:  # UTF-8 encoding
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

        # 4. Generate confusion matrix
        self.create_confusion_matrix()

        # 5. Visualize predictions
        self.visualize_predictions(num_samples=15)

        # 6. Generate classification report
        self.generate_classification_report(results)

        # 7. Generate detailed analysis
        self.generate_detailed_analysis(results)

        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)

        return results


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration matching your training setup"""
    YOLO_DATASET_ROOT = "yolo_dataset"
    RESULTS_DIR = "evaluation_results"
    MODELS_DIR = "models"
    CLASSES = ['good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze']
    IMG_SIZE = 640
    BATCH_SIZE = 16
    CONF_THRESHOLD = 0.35
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 1


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Capsule Defect Detection Model')
    parser.add_argument(
        '--model',
        type=str,
        default='models/capsule_defect_detector/weights/best.pt',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.35,
        help='Confidence threshold'
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("\nAvailable models:")
        model_dir = Path('models/capsule_defect_detector/weights')
        if model_dir.exists():
            for model_file in model_dir.glob('*.pt'):
                print(f"  - {model_file}")
        return

    # Update config with custom threshold
    Config.CONF_THRESHOLD = args.conf

    # Create evaluator
    evaluator = CapsuleModelEvaluator(args.model, Config)

    # Run complete evaluation
    results = evaluator.run_complete_evaluation()

    # Final summary
    print(f"\nFinal Summary:")
    print(f"  mAP@0.5: {results['overall_metrics']['mAP50']*100:.1f}%")
    print(f"  Target (85-90%): {'[PASS]' if results['overall_metrics']['mAP50'] >= 0.85 else '[FAIL]'}")
    print(f"\nAll results saved to: evaluation_results/")


if __name__ == "__main__":
    main()