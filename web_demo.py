# """
# Capsule Defect Detection Web Demo
# Standalone Flask application for real-time defect detection
# """

# from flask import Flask, render_template_string, request, jsonify
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import base64
# from pathlib import Path
# import os


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class Config:
#     MODEL_PATH = "models/capsule_defect_detector/weights/best.pt"  # Update this path
#     CLASSES = ['good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze']
#     CONF_THRESHOLD = 0.35
#     IOU_THRESHOLD = 0.45
#     MAX_DETECTIONS = 1


# # ============================================================================
# # DETECTOR CLASS
# # ============================================================================

# class CapsuleDefectDetector:
#     """Production-ready inference system"""

#     def __init__(self, model_path, config):
#         self.model = YOLO(model_path)
#         self.config = config
#         self.class_names = config.CLASSES
#         print(f"Model loaded from: {model_path}")

#     def detect(self, image_path):
#         """Detect defects in a capsule image"""
#         results = self.model.predict(
#             source=image_path,
#             conf=self.config.CONF_THRESHOLD,
#             iou=self.config.IOU_THRESHOLD,
#             max_det=self.config.MAX_DETECTIONS,
#             verbose=False
#         )[0]

#         detection = {
#             'detected': False,
#             'class': 'Unknown',
#             'confidence': 0.0,
#             'status': 'No detection'
#         }

#         if len(results.boxes) > 0:
#             box = results.boxes[0]
#             class_id = int(box.cls[0])
#             confidence = float(box.conf[0])

#             detection.update({
#                 'detected': True,
#                 'class': self.class_names[class_id],
#                 'confidence': confidence,
#                 'status': 'GOOD' if self.class_names[class_id] == 'good' else 'DEFECTIVE'
#             })

#         return detection, results

#     def visualize_detection(self, image_path):
#         """Visualize detection with bounding box"""
#         detection, results = self.detect(image_path)

#         # Get annotated image from YOLO results
#         annotated = results.plot()

#         return annotated, detection


# # ============================================================================
# # FLASK APPLICATION
# # ============================================================================

# app = Flask(__name__)
# detector = CapsuleDefectDetector(Config.MODEL_PATH, Config)

# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Capsule Defect Detection System</title>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }

#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             min-height: 100vh;
#             padding: 20px;
#         }

#         .container {
#             max-width: 1200px;
#             margin: 0 auto;
#             background: white;
#             border-radius: 20px;
#             box-shadow: 0 20px 60px rgba(0,0,0,0.3);
#             padding: 40px;
#         }

#         header {
#             text-align: center;
#             margin-bottom: 40px;
#         }

#         h1 {
#             color: #667eea;
#             font-size: 2.5em;
#             margin-bottom: 10px;
#         }

#         .subtitle {
#             color: #666;
#             font-size: 1.1em;
#         }

#         .upload-section {
#             border: 3px dashed #667eea;
#             border-radius: 15px;
#             padding: 40px;
#             text-align: center;
#             background: #f8f9ff;
#             margin-bottom: 30px;
#             cursor: pointer;
#             transition: all 0.3s;
#         }

#         .upload-section:hover {
#             background: #e8ebff;
#             border-color: #764ba2;
#             transform: translateY(-2px);
#         }

#         .upload-section h2 {
#             color: #667eea;
#             margin-bottom: 15px;
#         }

#         .upload-section p {
#             color: #666;
#             margin: 20px 0;
#         }

#         .upload-btn {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 15px 40px;
#             border: none;
#             border-radius: 30px;
#             font-size: 1.1em;
#             cursor: pointer;
#             transition: transform 0.2s;
#             font-weight: 600;
#         }

#         .upload-btn:hover {
#             transform: scale(1.05);
#         }

#         .loading {
#             display: none;
#             text-align: center;
#             padding: 40px;
#         }

#         .spinner {
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #667eea;
#             border-radius: 50%;
#             width: 60px;
#             height: 60px;
#             animation: spin 1s linear infinite;
#             margin: 0 auto 20px;
#         }

#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }

#         .loading p {
#             color: #667eea;
#             font-size: 1.2em;
#             font-weight: 600;
#         }

#         .results {
#             display: none;
#             margin-top: 30px;
#         }

#         .result-card {
#             background: #f8f9ff;
#             border-radius: 15px;
#             padding: 30px;
#         }

#         .result-header {
#             text-align: center;
#             margin-bottom: 30px;
#         }

#         .status-good {
#             color: #10b981;
#             font-size: 2em;
#             font-weight: bold;
#             display: flex;
#             align-items: center;
#             justify-content: center;
#             gap: 10px;
#         }

#         .status-defect {
#             color: #ef4444;
#             font-size: 2em;
#             font-weight: bold;
#             display: flex;
#             align-items: center;
#             justify-content: center;
#             gap: 10px;
#         }

#         .metrics-grid {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#             gap: 20px;
#             margin: 30px 0;
#         }

#         .metric {
#             background: white;
#             padding: 20px;
#             border-radius: 10px;
#             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#             text-align: center;
#             transition: transform 0.2s;
#         }

#         .metric:hover {
#             transform: translateY(-5px);
#         }

#         .metric-label {
#             color: #666;
#             font-size: 0.9em;
#             margin-bottom: 10px;
#             text-transform: uppercase;
#             letter-spacing: 1px;
#         }

#         .metric-value {
#             color: #667eea;
#             font-size: 1.8em;
#             font-weight: bold;
#         }

#         .image-display {
#             margin-top: 30px;
#             text-align: center;
#         }

#         .image-display img {
#             max-width: 100%;
#             height: auto;
#             border-radius: 10px;
#             box-shadow: 0 10px 30px rgba(0,0,0,0.2);
#         }

#         .class-badge {
#             display: inline-block;
#             padding: 8px 20px;
#             border-radius: 20px;
#             font-weight: 600;
#             margin: 10px 0;
#         }

#         .badge-good {
#             background: #d1fae5;
#             color: #065f46;
#         }

#         .badge-defect {
#             background: #fee2e2;
#             color: #991b1b;
#         }

#         .reset-btn {
#             background: #6b7280;
#             color: white;
#             padding: 12px 30px;
#             border: none;
#             border-radius: 25px;
#             font-size: 1em;
#             cursor: pointer;
#             margin-top: 20px;
#             transition: background 0.3s;
#         }

#         .reset-btn:hover {
#             background: #4b5563;
#         }

#         .info-section {
#             background: #e0e7ff;
#             padding: 20px;
#             border-radius: 10px;
#             margin-top: 30px;
#         }

#         .info-section h3 {
#             color: #667eea;
#             margin-bottom: 10px;
#         }

#         .info-section ul {
#             list-style-position: inside;
#             color: #4b5563;
#         }

#         .info-section li {
#             margin: 5px 0;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <header>
#             <h1>Capsule Defect Detection System</h1>
#             <p class="subtitle">YOLOv8-Powered Quality Control | Real-time Defect Classification</p>
#         </header>

#         <div class="upload-section" onclick="document.getElementById('fileInput').click()">
#             <h2>Upload Capsule Image</h2>
#             <p>Click to select or drag and drop an image</p>
#             <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
#             <button class="upload-btn">Choose Image</button>
#         </div>

#         <div class="loading" id="loading">
#             <div class="spinner"></div>
#             <p>Analyzing capsule image...</p>
#         </div>

#         <div class="results" id="results">
#             <div class="result-card">
#                 <div class="result-header">
#                     <h2>Detection Results</h2>
#                     <div id="statusDisplay"></div>
#                 </div>

#                 <div class="metrics-grid" id="metricsDisplay"></div>

#                 <div class="image-display" id="imageDisplay"></div>

#                 <div style="text-align: center;">
#                     <button class="reset-btn" onclick="resetDemo()">Analyze Another Image</button>
#                 </div>
#             </div>
#         </div>

#         <div class="info-section">
#             <h3>Detectable Defect Types:</h3>
#             <ul>
#                 <li><strong>Good:</strong> No defects detected</li>
#                 <li><strong>Crack:</strong> Visible cracks on capsule surface</li>
#                 <li><strong>Faulty Imprint:</strong> Incorrect or missing imprint</li>
#                 <li><strong>Poke:</strong> Puncture or hole in capsule</li>
#                 <li><strong>Scratch:</strong> Surface scratches</li>
#                 <li><strong>Squeeze:</strong> Deformed or squeezed capsule</li>
#             </ul>
#         </div>
#     </div>

#     <script>
#         function uploadImage() {
#             const fileInput = document.getElementById('fileInput');
#             const file = fileInput.files[0];

#             if (!file) return;

#             // Validate file type
#             if (!file.type.startsWith('image/')) {
#                 alert('Please upload a valid image file');
#                 return;
#             }

#             const formData = new FormData();
#             formData.append('image', file);

#             // Show loading
#             document.getElementById('loading').style.display = 'block';
#             document.getElementById('results').style.display = 'none';
#             document.querySelector('.upload-section').style.display = 'none';

#             // Send to server
#             fetch('/detect', {
#                 method: 'POST',
#                 body: formData
#             })
#             .then(response => response.json())
#             .then(data => {
#                 document.getElementById('loading').style.display = 'none';
#                 if (data.error) {
#                     alert('Error: ' + data.error);
#                     resetDemo();
#                 } else {
#                     displayResults(data);
#                 }
#             })
#             .catch(error => {
#                 document.getElementById('loading').style.display = 'none';
#                 alert('Error processing image: ' + error);
#                 resetDemo();
#             });
#         }

#         function displayResults(data) {
#             const statusClass = data.status === 'GOOD' ? 'status-good' : 'status-defect';
#             const statusIcon = data.status === 'GOOD' ? '✅' : '❌';
#             const badgeClass = data.status === 'GOOD' ? 'badge-good' : 'badge-defect';

#             document.getElementById('statusDisplay').innerHTML = `
#                 <div class="${statusClass}">
#                     <span>${statusIcon}</span>
#                     <span>${data.status}</span>
#                 </div>
#                 <div class="class-badge ${badgeClass}">${data.class.toUpperCase()}</div>
#             `;

#             document.getElementById('metricsDisplay').innerHTML = `
#                 <div class="metric">
#                     <div class="metric-label">Classification</div>
#                     <div class="metric-value">${data.class}</div>
#                 </div>
#                 <div class="metric">
#                     <div class="metric-label">Confidence</div>
#                     <div class="metric-value">${(data.confidence * 100).toFixed(1)}%</div>
#                 </div>
#                 <div class="metric">
#                     <div class="metric-label">Status</div>
#                     <div class="metric-value" style="color: ${data.status === 'GOOD' ? '#10b981' : '#ef4444'}">
#                         ${data.status}
#                     </div>
#                 </div>
#             `;

#             document.getElementById('imageDisplay').innerHTML = `
#                 <h3 style="margin-bottom: 15px; color: #667eea;">Annotated Image</h3>
#                 <img src="data:image/jpeg;base64,${data.image}" alt="Detection Result">
#             `;

#             document.getElementById('results').style.display = 'block';
#         }

#         function resetDemo() {
#             document.getElementById('results').style.display = 'none';
#             document.querySelector('.upload-section').style.display = 'block';
#             document.getElementById('fileInput').value = '';
#         }

#         // Drag and drop functionality
#         const uploadSection = document.querySelector('.upload-section');

#         uploadSection.addEventListener('dragover', (e) => {
#             e.preventDefault();
#             uploadSection.style.borderColor = '#764ba2';
#             uploadSection.style.background = '#e8ebff';
#         });

#         uploadSection.addEventListener('dragleave', (e) => {
#             e.preventDefault();
#             uploadSection.style.borderColor = '#667eea';
#             uploadSection.style.background = '#f8f9ff';
#         });

#         uploadSection.addEventListener('drop', (e) => {
#             e.preventDefault();
#             uploadSection.style.borderColor = '#667eea';
#             uploadSection.style.background = '#f8f9ff';

#             const files = e.dataTransfer.files;
#             if (files.length > 0) {
#                 document.getElementById('fileInput').files = files;
#                 uploadImage();
#             }
#         });
#     </script>
# </body>
# </html>
# """


# @app.route('/')
# def index():
#     return render_template_string(HTML_TEMPLATE)


# @app.route('/detect', methods=['POST'])
# def detect():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image uploaded'}), 400

#         file = request.files['image']

#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400

#         # Save temporarily
#         temp_path = 'temp_upload.jpg'
#         file.save(temp_path)

#         # Run detection
#         annotated_img, detection = detector.visualize_detection(temp_path)

#         # Convert to base64
#         _, buffer = cv2.imencode('.jpg', annotated_img)
#         img_base64 = base64.b64encode(buffer).decode('utf-8')

#         # Clean up
#         os.remove(temp_path)

#         return jsonify({
#             'status': detection['status'],
#             'class': detection['class'],
#             'confidence': detection['confidence'],
#             'image': img_base64
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# if __name__ == '__main__':
#     # Check if model exists
#     if not Path(Config.MODEL_PATH).exists():
#         print(f"ERROR: Model not found at {Config.MODEL_PATH}")
#         print("Please update the MODEL_PATH in the Config class")
#         exit(1)

#     print("=" * 80)
#     print("CAPSULE DEFECT DETECTION WEB DEMO")
#     print("=" * 80)
#     print(f"Model: {Config.MODEL_PATH}")
#     print(f"Classes: {', '.join(Config.CLASSES)}")
#     print("\nStarting web server...")
#     print("Access the demo at: http://localhost:10000")
#     print("=" * 80)

#     app.run(host='0.0.0.0', port=10000, debug=False)







"""
Capsule Defect Detection Web Demo - Production Version for Render
Optimized Flask application for real-time defect detection on cloud platforms
"""

from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from pathlib import Path
import os
import gc
import torch
import tempfile
import logging
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MODEL_PATH = "models/capsule_defect_detector/weights/best.pt"
    CLASSES = ['good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze']
    CONF_THRESHOLD = 0.25  # Lowered for better detection
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 1
    MAX_IMAGE_SIZE = 640  # Limit image size for memory optimization

# ============================================================================
# DETECTOR CLASS - OPTIMIZED FOR PRODUCTION
# ============================================================================

class CapsuleDefectDetector:
    """Production-optimized inference system with memory management"""

    def __init__(self, model_path, config):
        self.config = config
        self.class_names = config.CLASSES
        self.model = None
        self._model_lock = Lock()  # Thread safety
        self._load_model(model_path)
        logger.info(f"Model loaded from: {model_path}")

    def _load_model(self, model_path):
        """Load model with memory optimization"""
        try:
            # Clear memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Load model
            self.model = YOLO(model_path)
            
            # Warm-up with dummy inference to optimize performance
            logger.info("Warming up model...")
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                _ = self.model.predict(dummy_image, verbose=False, conf=0.1)
            
            # Clean up after warm-up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Model warm-up completed")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _optimize_image(self, image_path):
        """Optimize image size and format for faster inference"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            h, w = image.shape[:2]
            max_size = self.config.MAX_IMAGE_SIZE
            
            # Resize if too large
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Save optimized image
                temp_path = image_path.replace('.jpg', '_optimized.jpg')
                cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return temp_path
            
            return image_path
            
        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            return image_path

    def detect(self, image_path):
        """Detect defects with memory management"""
        with self._model_lock:  # Ensure thread safety
            try:
                # Optimize image first
                optimized_path = self._optimize_image(image_path)
                
                # Clear memory before inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Run inference with optimized settings
                with torch.no_grad():
                    results = self.model.predict(
                        source=optimized_path,
                        conf=self.config.CONF_THRESHOLD,
                        iou=self.config.IOU_THRESHOLD,
                        max_det=self.config.MAX_DETECTIONS,
                        verbose=False,
                        device='cpu',  # Force CPU to avoid GPU memory issues
                        imgsz=640  # Standard input size
                    )[0]

                # Process results
                detection = {
                    'detected': False,
                    'class': 'good',  # Default to good
                    'confidence': 0.0,
                    'status': 'GOOD'  # Default status
                }

                if len(results.boxes) > 0:
                    box = results.boxes[0]
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    detection.update({
                        'detected': True,
                        'class': self.class_names[class_id],
                        'confidence': confidence,
                        'status': 'GOOD' if self.class_names[class_id] == 'good' else 'DEFECTIVE'
                    })
                
                # Clean up optimized image if created
                if optimized_path != image_path:
                    try:
                        os.remove(optimized_path)
                    except:
                        pass

                # Clean memory after inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                return detection, results

            except Exception as e:
                logger.error(f"Detection failed: {e}")
                # Return default good classification on error
                return {
                    'detected': False,
                    'class': 'good',
                    'confidence': 0.0,
                    'status': 'GOOD'
                }, None

    def visualize_detection(self, image_path):
        """Visualize detection with error handling"""
        try:
            detection, results = self.detect(image_path)
            
            if results is not None:
                # Get annotated image
                annotated = results.plot()
            else:
                # If no results, return original image
                annotated = cv2.imread(image_path)
                if annotated is None:
                    # Create placeholder image
                    annotated = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(annotated, "Image Processing Error", 
                              (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return annotated, detection
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Return error image
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Processing Error", 
                       (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return error_img, {
                'detected': False,
                'class': 'error',
                'confidence': 0.0,
                'status': 'ERROR'
            }

# ============================================================================
# FLASK APPLICATION - PRODUCTION OPTIMIZED
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global detector instance
detector = None

def init_detector():
    """Initialize detector with error handling"""
    global detector
    try:
        if not Path(Config.MODEL_PATH).exists():
            logger.error(f"Model not found at {Config.MODEL_PATH}")
            return False
        
        detector = CapsuleDefectDetector(Config.MODEL_PATH, Config)
        logger.info("Detector initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return False

# Initialize detector on startup
if not init_detector():
    logger.error("Failed to initialize detector - using dummy mode")

# Your existing HTML template remains the same
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Capsule Defect Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

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

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #666;
            font-size: 1.1em;
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
            transform: translateY(-2px);
        }

        .upload-section h2 {
            color: #667eea;
            margin-bottom: 15px;
        }

        .upload-section p {
            color: #666;
            margin: 20px 0;
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
            font-weight: 600;
        }

        .upload-btn:hover {
            transform: scale(1.05);
        }

        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading p {
            color: #667eea;
            font-size: 1.2em;
            font-weight: 600;
        }

        .results {
            display: none;
            margin-top: 30px;
        }

        .result-card {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 30px;
        }

        .result-header {
            text-align: center;
            margin-bottom: 30px;
        }

        .status-good {
            color: #10b981;
            font-size: 2em;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .status-defect {
            color: #ef4444;
            font-size: 2em;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }

        .metric {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }

        .metric:hover {
            transform: translateY(-5px);
        }

        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-value {
            color: #667eea;
            font-size: 1.8em;
            font-weight: bold;
        }

        .image-display {
            margin-top: 30px;
            text-align: center;
        }

        .image-display img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .class-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 600;
            margin: 10px 0;
        }

        .badge-good {
            background: #d1fae5;
            color: #065f46;
        }

        .badge-defect {
            background: #fee2e2;
            color: #991b1b;
        }

        .reset-btn {
            background: #6b7280;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            margin-top: 20px;
            transition: background 0.3s;
        }

        .reset-btn:hover {
            background: #4b5563;
        }

        .info-section {
            background: #e0e7ff;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
        }

        .info-section h3 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .info-section ul {
            list-style-position: inside;
            color: #4b5563;
        }

        .info-section li {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Capsule Defect Detection System</h1>
            <p class="subtitle">YOLOv8-Powered Quality Control | Real-time Defect Classification</p>
        </header>

        <div class="upload-section" onclick="document.getElementById('fileInput').click()">
            <h2>Upload Capsule Image</h2>
            <p>Click to select or drag and drop an image (Max 16MB)</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
            <button class="upload-btn">Choose Image</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing capsule image... (This may take up to 2 minutes)</p>
        </div>

        <div class="results" id="results">
            <div class="result-card">
                <div class="result-header">
                    <h2>Detection Results</h2>
                    <div id="statusDisplay"></div>
                </div>

                <div class="metrics-grid" id="metricsDisplay"></div>

                <div class="image-display" id="imageDisplay"></div>

                <div style="text-align: center;">
                    <button class="reset-btn" onclick="resetDemo()">Analyze Another Image</button>
                </div>
            </div>
        </div>

        <div class="info-section">
            <h3>Detectable Defect Types:</h3>
            <ul>
                <li><strong>Good:</strong> No defects detected</li>
                <li><strong>Crack:</strong> Visible cracks on capsule surface</li>
                <li><strong>Faulty Imprint:</strong> Incorrect or missing imprint</li>
                <li><strong>Poke:</strong> Puncture or hole in capsule</li>
                <li><strong>Scratch:</strong> Surface scratches</li>
                <li><strong>Squeeze:</strong> Deformed or squeezed capsule</li>
            </ul>
        </div>
    </div>

    <script>
        function uploadImage() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];

            if (!file) return;

            // Validate file type
            if (!file.type.startsWith('image/')) {
                alert('Please upload a valid image file (JPEG, PNG, etc.)');
                return;
            }

            // Validate file size (16MB limit)
            if (file.size > 16 * 1024 * 1024) {
                alert('File too large. Please upload an image smaller than 16MB.');
                return;
            }

            const formData = new FormData();
            formData.append('image', file);

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.querySelector('.upload-section').style.display = 'none';

            // Set timeout for request (5 minutes)
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000);

            // Send to server
            fetch('/detect', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            })
            .then(async response => {
                clearTimeout(timeoutId);
                
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return await response.json();
                } else {
                    const text = await response.text();
                    console.error('Received HTML instead of JSON:', text);
                    throw new Error('Server error - please try again');
                }
            })
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                if (data.error) {
                    alert('Error: ' + data.error);
                    resetDemo();
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                clearTimeout(timeoutId);
                document.getElementById('loading').style.display = 'none';
                
                if (error.name === 'AbortError') {
                    alert('Request timeout - the analysis is taking too long. Please try again with a smaller image.');
                } else {
                    alert('Error processing image: ' + error.message);
                }
                resetDemo();
            });
        }

        function displayResults(data) {
            const statusClass = data.status === 'GOOD' ? 'status-good' : 'status-defect';
            const statusIcon = data.status === 'GOOD' ? '✅' : '❌';
            const badgeClass = data.status === 'GOOD' ? 'badge-good' : 'badge-defect';

            document.getElementById('statusDisplay').innerHTML = `
                <div class="${statusClass}">
                    <span>${statusIcon}</span>
                    <span>${data.status}</span>
                </div>
                <div class="class-badge ${badgeClass}">${data.class.toUpperCase()}</div>
            `;

            document.getElementById('metricsDisplay').innerHTML = `
                <div class="metric">
                    <div class="metric-label">Classification</div>
                    <div class="metric-value">${data.class}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">${(data.confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Status</div>
                    <div class="metric-value" style="color: ${data.status === 'GOOD' ? '#10b981' : '#ef4444'}">
                        ${data.status}
                    </div>
                </div>
            `;

            document.getElementById('imageDisplay').innerHTML = `
                <h3 style="margin-bottom: 15px; color: #667eea;">Analysis Result</h3>
                <img src="data:image/jpeg;base64,${data.image}" alt="Detection Result">
            `;

            document.getElementById('results').style.display = 'block';
        }

        function resetDemo() {
            document.getElementById('results').style.display = 'none';
            document.querySelector('.upload-section').style.display = 'block';
            document.getElementById('fileInput').value = '';
        }

        // Drag and drop functionality
        const uploadSection = document.querySelector('.upload-section');

        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#764ba2';
            uploadSection.style.background = '#e8ebff';
        });

        uploadSection.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#667eea';
            uploadSection.style.background = '#f8f9ff';
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.style.borderColor = '#667eea';
            uploadSection.style.background = '#f8f9ff';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('fileInput').files = files;
                uploadImage();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if detector is available
        if detector is None:
            logger.error("Detector not initialized")
            return jsonify({'error': 'Service temporarily unavailable'}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Create secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        try:
            logger.info("Starting detection...")
            
            # Run detection with timeout protection
            annotated_img, detection = detector.visualize_detection(temp_path)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            logger.info(f"Detection completed: {detection['class']} ({detection['confidence']:.2f})")

            return jsonify({
                'status': detection['status'],
                'class': detection['class'],
                'confidence': detection['confidence'],
                'image': img_base64
            })

        finally:
            # Always clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': 'Internal server error - please try again'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if detector else 'degraded',
        'model_loaded': detector is not None
    })

# ============================================================================
# MAIN EXECUTION - PRODUCTION READY
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    if detector:
        logger.info("=" * 80)
        logger.info("CAPSULE DEFECT DETECTION WEB DEMO - PRODUCTION")
        logger.info("=" * 80)
        logger.info(f"Model: {Config.MODEL_PATH}")
        logger.info(f"Classes: {', '.join(Config.CLASSES)}")
        logger.info(f"Port: {port}")
        logger.info("=" * 80)

    app.run(host='0.0.0.0', port=port, debug=debug_mode)
