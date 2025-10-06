💊 Capsule Defect Detection (CV)

A **YOLOv8-based computer vision system** for automated **pharmaceutical capsule quality control**.

---

🧠 Overview
This project aims to detect and classify **defects in pharmaceutical capsules** using deep learning.  
The system can identify **six types of capsule conditions**:
- ✅ Good
- ⚡ Crack
- 🏷️ Faulty Imprint
- 🕳️ Poke
- ✴️ Scratch
- 💢 Squeeze Damage

---

🚀 Features
- **6-Class Detection:** Classifies capsules into good and five defect categories  
- **YOLOv8 Integration:** Leverages state-of-the-art object detection  
- **Flask Web Interface:** Simple UI for real-time image upload and testing  
- **Real-time Processing:** Fast and efficient inference pipeline  
- **Comprehensive Evaluation:** Provides detailed metrics and visualizations  

---

⚙️ Quick Start

1️⃣ Install Dependencies

pip install ultralytics opencv-python flask numpy scikit-learn matplotlib


2️⃣ Run the System
python web_demo.py

3️⃣ Access the Web Interface

Open your browser and go to:

http://localhost:5000

🗂️ Dataset Structure:

Your dataset should follow this structure:

dataset/
├── good_images/
└── defective_images/
    ├── crack/
    ├── faulty_imprint/
    ├── poke/
    ├── scratch/
    └── squeeze/

🧩 Project Workflow:
Stage	Description
Data Preparation	Converts dataset into YOLOv8-compatible format
Model Training	Trains YOLOv8 with optimized hyperparameters
Evaluation	Tests model with metrics and confusion matrix
Web Interface	Enables real-time defect detection through a simple UI

📈 Performance:
🎯 Target Accuracy: 85–90%
⚡ Inference Speed: Real-time on local GPU/CPU
🧩 Deployment Ready: Flask web app for seamless integration

🧑‍💻 Author:

Vinith Kumar G
📎 GitHub: @VinithKumarG-18
📎 Linkedin: www.linkedin.com/in/vinithkumarg
