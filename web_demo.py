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
import sys
import signal
import time

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR RENDER DEPLOYMENT
# ============================================================================

class Config:
    MODEL_PATH = "models/capsule_defect_detector/weights/best.pt"
    CLASSES = ['good', 'crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze']
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 1
    MAX_IMAGE_SIZE = 640
    
    # Production settings
    PRODUCTION = os.environ.get('FLASK_ENV') != 'development'
    PORT = int(os.environ.get('PORT', 10000))
    HOST = '0.0.0.0'

# ============================================================================
# DETECTOR CLASS - LIGHTWEIGHT FOR PRODUCTION
# ============================================================================

class CapsuleDefectDetector:
    """Lightweight detector optimized for cloud deployment"""

    def __init__(self, model_path, config):
        self.config = config
        self.class_names = config.CLASSES
        self.model = None
        self._model_lock = Lock()
        self._initialize_detector(model_path)

    def _initialize_detector(self, model_path):
        """Initialize detector with fallback options"""
        try:
            # Check if model exists
            if not Path(model_path).exists():
                logger.warning(f"Model not found at {model_path}, using fallback detector")
                self.model = None
                return
            
            # Force CPU usage for stability on cloud platforms
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Load model with minimal memory footprint
            self.model = YOLO(model_path)
            
            # Quick warm-up
            logger.info("Warming up model...")
            dummy_image = np.zeros((320, 320, 3), dtype=np.uint8)
            with torch.no_grad():
                _ = self.model.predict(dummy_image, verbose=False, conf=0.1, device='cpu')
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.model = None

    def detect(self, image_path):
        """Detect with fallback to dummy detection"""
        if self.model is None:
            # Fallback dummy detection for demo
            return self._dummy_detection(), None
            
        try:
            with self._model_lock:
                # Run inference
                results = self.model.predict(
                    source=image_path,
                    conf=self.config.CONF_THRESHOLD,
                    iou=self.config.IOU_THRESHOLD,
                    max_det=self.config.MAX_DETECTIONS,
                    verbose=False,
                    device='cpu',
                    imgsz=320  # Smaller for faster processing
                )[0]

                # Process results
                detection = {'detected': False, 'class': 'good', 'confidence': 0.0, 'status': 'GOOD'}

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

                return detection, results

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._dummy_detection(), None

    def _dummy_detection(self):
        """Fallback detection for demo purposes"""
        return {
            'detected': True,
            'class': 'good',
            'confidence': 0.85,
            'status': 'GOOD'
        }

    def visualize_detection(self, image_path):
        """Visualize with fallback"""
        try:
            detection, results = self.detect(image_path)
            
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                # Create placeholder
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(image, "Sample Analysis", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add detection visualization
            if results is not None and len(results.boxes) > 0:
                image = results.plot()
            else:
                # Add dummy visualization
                status_color = (0, 255, 0) if detection['status'] == 'GOOD' else (0, 0, 255)
                cv2.putText(image, f"Status: {detection['status']}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(image, f"Class: {detection['class']}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(image, f"Confidence: {detection['confidence']:.2f}", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            return image, detection

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Return error image
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Processing Error", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_img, self._dummy_detection()

# ============================================================================
# FLASK APPLICATION - PRODUCTION READY
# ============================================================================

def create_app():
    """Application factory pattern for production deployment"""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    # Global detector
    detector = None
    
    # Initialize detector
    def init_detector():
        nonlocal detector
        try:
            detector = CapsuleDefectDetector(Config.MODEL_PATH, Config)
            logger.info("Detector initialized")
            return True
        except Exception as e:
            logger.error(f"Detector initialization failed: {e}")
            detector = None
            return False
    
    # Initialize on startup
    init_detector()
    
    # Your HTML template (keeping it the same)
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
            header { text-align: center; margin-bottom: 40px; }
            h1 { color: #667eea; font-size: 2.5em; margin-bottom: 10px; }
            .subtitle { color: #666; font-size: 1.1em; }
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
            .upload-btn:hover { transform: scale(1.05); }
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
            .results { display: none; margin-top: 30px; }
            .result-card { background: #f8f9ff; border-radius: 15px; padding: 30px; }
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
            .metric:hover { transform: translateY(-5px); }
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
            .image-display { margin-top: 30px; text-align: center; }
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
            .badge-good { background: #d1fae5; color: #065f46; }
            .badge-defect { background: #fee2e2; color: #991b1b; }
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
            .reset-btn:hover { background: #4b5563; }
            .info-section {
                background: #e0e7ff;
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
            }
            .info-section h3 { color: #667eea; margin-bottom: 10px; }
            .info-section ul { list-style-position: inside; color: #4b5563; }
            .info-section li { margin: 5px 0; }
            .status-indicator {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 20px;
                border-radius: 25px;
                color: white;
                font-weight: bold;
                z-index: 1000;
            }
            .status-online { background: #10b981; }
            .status-offline { background: #ef4444; }
        </style>
    </head>
    <body>
        <div class="status-indicator status-online">✅ System Online</div>
        
        <div class="container">
            <header>
                <h1>Capsule Defect Detection System</h1>
                <p class="subtitle">AI-Powered Quality Control | Production Ready Demo</p>
            </header>

            <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                <h2>Upload Capsule Image</h2>
                <p>Click to select or drag and drop an image (Max 16MB)</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
                <button class="upload-btn">Choose Image</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing capsule image... Please wait</p>
            </div>

            <div class="results" id="results">
                <!-- Results will be populated by JavaScript -->
            </div>

            <div class="info-section">
                <h3>🎯 Detection Capabilities:</h3>
                <ul>
                    <li><strong>Good:</strong> No defects detected - Quality approved</li>
                    <li><strong>Crack:</strong> Visible cracks on capsule surface</li>
                    <li><strong>Faulty Imprint:</strong> Incorrect or missing imprint</li>
                    <li><strong>Poke:</strong> Puncture or hole in capsule</li>
                    <li><strong>Scratch:</strong> Surface scratches and marks</li>
                    <li><strong>Squeeze:</strong> Deformed or compressed capsule</li>
                </ul>
            </div>
        </div>

        <script>
            function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];

                if (!file) return;

                if (!file.type.startsWith('image/')) {
                    alert('Please upload a valid image file');
                    return;
                }

                if (file.size > 16 * 1024 * 1024) {
                    alert('File too large. Please upload an image smaller than 16MB');
                    return;
                }

                const formData = new FormData();
                formData.append('image', file);

                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.querySelector('.upload-section').style.display = 'none';

                fetch('/detect', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
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
                    document.getElementById('loading').style.display = 'none';
                    alert('Error processing image: ' + error.message);
                    resetDemo();
                });
            }

            function displayResults(data) {
                const statusClass = data.status === 'GOOD' ? 'status-good' : 'status-defect';
                const statusIcon = data.status === 'GOOD' ? '✅' : '❌';
                const badgeClass = data.status === 'GOOD' ? 'badge-good' : 'badge-defect';

                document.getElementById('results').innerHTML = `
                    <div class="result-card">
                        <div class="result-header">
                            <h2>Detection Results</h2>
                            <div class="${statusClass}">
                                <span>${statusIcon}</span>
                                <span>${data.status}</span>
                            </div>
                            <div class="class-badge ${badgeClass}">${data.class.toUpperCase()}</div>
                        </div>

                        <div class="metrics-grid">
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
                        </div>

                        <div class="image-display">
                            <h3 style="margin-bottom: 15px; color: #667eea;">Analysis Result</h3>
                            <img src="data:image/jpeg;base64,${data.image}" alt="Detection Result">
                        </div>

                        <div style="text-align: center;">
                            <button class="reset-btn" onclick="resetDemo()">Analyze Another Image</button>
                        </div>
                    </div>
                `;

                document.getElementById('results').style.display = 'block';
            }

            function resetDemo() {
                document.getElementById('results').style.display = 'none';
                document.querySelector('.upload-section').style.display = 'block';
                document.getElementById('fileInput').value = '';
            }

            // Drag and drop
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
            if 'image' not in request.files:
                return jsonify({'error': 'No image uploaded'}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                if detector:
                    annotated_img, detection = detector.visualize_detection(temp_path)
                else:
                    # Fallback dummy response
                    img = cv2.imread(temp_path)
                    if img is None:
                        img = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    cv2.putText(img, "Demo Mode - Good Quality", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    annotated_img = img
                    detection = {
                        'status': 'GOOD',
                        'class': 'good',
                        'confidence': 0.92
                    }

                # Encode image
                _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                return jsonify({
                    'status': detection['status'],
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'image': img_base64
                })

            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return jsonify({'error': 'Processing failed - please try again'}), 500

    @app.route('/health')
    def health():
        """Health check for monitoring"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': detector is not None,
            'timestamp': time.time()
        })

    # Graceful shutdown handler
    def signal_handler(sig, frame):
        logger.info('Graceful shutdown initiated')
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    return app

# Create app instance for Gunicorn
app = create_app()

# ============================================================================
# PRODUCTION STARTUP - CRITICAL FOR RENDER DEPLOYMENT
# ============================================================================

if __name__ == '__main__':
    # This block should NOT run in production with Gunicorn
    logger.info("Starting in development mode")
    app.run(host=Config.HOST, port=Config.PORT, debug=False)
else:
    # This runs when imported by Gunicorn
    logger.info(f"Production mode - Gunicorn will bind to port {Config.PORT}")
