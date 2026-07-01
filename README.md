#  Vision Guard: Night Driving Guidance System

An AI-powered **Night Driving Guidance System** that improves road visibility and driving safety in low-light conditions using **Deep Learning**, **Computer Vision**, and **Semantic Segmentation**. The system detects drivable road areas, vehicles, and pedestrians in real time and provides visual guidance to assist drivers during night driving.

---

##  Project Overview

Driving at night is significantly more challenging due to poor visibility, glare, and reduced reaction time. This project leverages an **Enhanced U-Net** segmentation model to analyze night-time road scenes and generate real-time guidance overlays.

The system identifies:
-  Drivable road area
-  Vehicles
-  Pedestrians

The processed output is displayed through a **Flask-based web application**, making the system easy to use and demonstrate.

---

##  Features

-  Night-time road scene analysis
-  Enhanced U-Net deep learning model
-  Vehicle detection
-  Pedestrian detection
-  Road segmentation
-  Real-time inference
-  Flask web interface
-  High-quality segmentation visualization

---

##  Project Structure

```
NightDrivingGS/
│
├── checkpoints/
├── dataset/
├── models/
│   └── enhanced_unet_model.py
├── static/
├── templates/
│   └── index.html
│
├── Enhanced_app.py
├── Enhanced_Inference.py
├── Train.py
├── Evaluate.py
├── Dataset_loader.py
├── Datacheck.py
├── Prepare_dataset.py
├── config.py
├── losses.py
├── requirements.txt
└── README.md
```

---

##  Model Architecture

The project uses an **Enhanced U-Net** architecture with improved feature extraction for accurate semantic segmentation in challenging night-time environments.

### Segmentation Classes

| Class | Description |
|--------|-------------|
| Background | Non-relevant pixels |
| Road | Drivable road surface |
| Vehicle | Cars, buses, trucks, motorcycles |
| Pedestrian | Human detection |

---

##  Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Flask
- Pillow
- Matplotlib
- Scikit-learn

---

##  Dataset

This project is trained on the **BDD100K Night Driving Dataset**.

Dataset includes:
- Night driving images
- Pixel-wise segmentation masks
- Urban road scenes
- Various weather and lighting conditions

> **Note:** The dataset is not included in this repository because of its large size.

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/aryanIngale/Vision-Guard-Night-Driving-Guidance-System.git
```

Move into the project directory:

```bash
cd Vision-Guard-Night-Driving-Guidance-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Running the Application

Start the Flask application:

```bash
python Enhanced_app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

##  Training

Train the model:

```bash
python Train.py
```

---

##  Evaluation

Evaluate model performance:

```bash
python Evaluate.py
```

---

##  Results

The model performs semantic segmentation on night driving scenes by highlighting:

- Drivable road
- Vehicles
- Pedestrians
- Guidance overlay for safer navigation

> Add screenshots of your application and model outputs here.

Example:

```
Input Image

↓

Road Segmentation

↓

Vehicle & Pedestrian Detection

↓

Final Guidance Output
```

---

##  Future Enhancements

- YOLO-based object detection
- Lane detection
- Traffic sign recognition
- Depth estimation
- Collision warning system
- GPS navigation integration
- Real-time video optimization
- Edge deployment on NVIDIA Jetson

---

##  Learning Outcomes

This project strengthened knowledge in:

- Deep Learning
- Computer Vision
- Semantic Segmentation
- PyTorch
- Flask
- Image Processing
- Model Training & Evaluation
- Deployment of AI applications

---

##  Author

**Aryan Ingale**

CSE AIML Student

GitHub: https://github.com/aryanIngale

---

##  If you found this project useful

Please consider giving this repository a on GitHub.
