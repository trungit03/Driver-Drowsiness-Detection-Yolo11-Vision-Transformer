# Driver-Drowsiness-Detection-YOLO11-ViT

This project implements a real-time drowsiness detection system. The system monitors facial states to detect signs of drowsiness and provides immediate alerts to prevent accidents.

## Dataset

Before running the application, you need to prepare your dataset:

Train YOLO11:
<br>
**Dataset Link:** [Yolo_data](https://universe.roboflow.com/karthik-madhvan/drowsiness-detection-xsriz/dataset/1)
<br><br>
Train ViT:
<br>
**Dataset Link:** [ViT_data](https://www.kaggle.com/datasets/trungngm/drowsy-3-classes-yolo)
## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/trungit03/Driver-Drowsiness-Detection-Yolo11-Vision-Transformer.git
```

### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  
```

**Using conda:**
```bash
conda create -n tf python=3.9
conda activate tf
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Model Training

Training Platform: Kaggle with GPU T4 x2

For YOLO
```bash
yolo train data=data.yaml model=yolov11s.pt epochs=100 batch=64 close_mosaic=20 imgsz=640 pretrained=True device='0,1'
```
For ViT
```
Follow vit_train.ipynb
```

## Usage

For YOLO
### Method 1: Run Flask Web Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Use the web interface to upload videos or start webcam detection

### Method 2: Run Direct Detection Script

**For Video Detection:**
```bash
python detector.py --source "path/to/your/video.mp4" --output "results/"
```
or
```bash
python detector.py --source "path/to/your/video.mp4" 
```

**For Webcam Detection:**
```bash
python detector.py
```
<hr>

**ViT**
```bash
python yolo_vit.py --model models_yolo11/best.pt --vit_model best_vit_drowsiness_model.pth --source 0
```

## Features

- Real-time drowsiness detection from webcam
- Video file processing capabilities
- Web-based interface for easy interaction
- Audio alerts when drowsiness is detected
- Configurable detection sensitivity

- Make sure the Flask port (5000) is not occupied by other applications
