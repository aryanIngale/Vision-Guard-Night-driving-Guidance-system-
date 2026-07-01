"""
config.py — Enhanced Night Guidance System Configuration
Optimized for multi-class segmentation and better detection
"""

import os


# Flask Server Settings

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
MAX_CONTENT_MB = 500


# Directory Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, CHECKPOINTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# Model Architecture Settings

IN_CHANNELS = 3        # RGB input
OUT_CHANNELS = 4       # Background, Road, Vehicles, Pedestrians
IMAGE_SIZE = 256       # Input image size (256x256)
DROPOUT_P = 0.3        # Dropout probability at bottleneck
FEATURES = 64          # Base feature count in U-Net

# Enhanced model settings
USE_ATTENTION = True   # Enable attention gates (highly recommended)
USE_SPP = False        # Spatial Pyramid Pooling (optional)


# Training Settings

EPOCHS = 10
BATCH_SIZE = 8         # Adjust based on GPU memory
LEARNING_RATE = 1e-4
LR_MIN = 1e-6
PATIENCE = 10          # Early stopping patience
RESUME = False         # Resume from checkpoint


# Data Loading

NUM_WORKERS = 4        # DataLoader workers

# Dataset paths
TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_MASK_DIR = "dataset/train/masks"
VAL_IMG_DIR = "dataset/val/images"
VAL_MASK_DIR = "dataset/val/masks"

# BDD100K class mapping
ROAD_TRAIN_ID = 0      # Road surface class ID


# Data Augmentation Settings

AUG_HFLIP = True       # Horizontal flip
AUG_VFLIP = False      # Vertical flip (usually False for driving)
AUG_ROTATE90 = False   # 90-degree rotation (usually False for driving)
AUG_COLOR_JITTER = True  # Color jittering
AUG_BLUR = True        # Gaussian blur


# Model Paths

BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")


# Inference Settings

CONF_THRESHOLD = 0.75  # Detection confidence threshold
                      
# Night enhancement
NIGHT_ENHANCE = True   # Enable CLAHE enhancement
CLAHE_CLIP = 2.0       # CLAHE clip limit (1.5-3.0)
CLAHE_GRID = (8, 8)    # CLAHE tile grid size

# Visualization settings
SHOW_DANGER_ZONES = True  # Highlight danger zones
SHOW_DISTANCE = True   # Show distance estimates
SHOW_TTC = False       # Show time-to-collision (optional)


# Color Scheme (BGR format for OpenCV)

COLOR_ROAD = (46, 125, 79)       # Green - safe road surface
COLOR_ROAD_CAUTION = (0, 165, 255)  # Orange - caution zone
COLOR_CAR = (140, 90, 58)         # Blue - vehicles
COLOR_PERSON = (43, 57, 192)      # Red - pedestrians
COLOR_WARNING = (0, 128, 255)     # Orange - warnings
COLOR_CRITICAL = (0, 0, 255)      # Red - critical danger

# Transparency levels (0.0-1.0)
ROAD_ALPHA = 0.40          # Road overlay opacity
VEHICLE_ALPHA = 0.30       # Vehicle overlay opacity
PEDESTRIAN_ALPHA = 0.40    # Pedestrian overlay opacity (higher = more visible)
DANGER_ALPHA = 0.15        # Danger zone tint


# Detection Thresholds

# Minimum pixels for valid detection (filters noise)
MIN_VEHICLE_PIXELS = 100       # Minimum pixels for vehicle detection
MIN_PEDESTRIAN_PIXELS = 300    # Minimum pixels for pedestrian detection (INCREASED to reduce clutter)
DANGER_PIXEL_THRESHOLD = 500   # Minimum pixels for danger alert

# Distance estimation parameters
CAMERA_HEIGHT_M = 1.2          # Camera height above ground (meters)
CAMERA_ANGLE_DEG = 10          # Camera tilt angle (degrees)
MIN_DISTANCE_M = 2.0           # Minimum measurable distance
MAX_DISTANCE_M = 50.0          # Maximum measurable distance

# Time-to-collision thresholds
TTC_CRITICAL_S = 2.0      # Critical warning (<2s)
TTC_WARNING_S = 4.0       # Warning (<4s)


# Video Processing

DEFAULT_FPS = 25.0             # Fallback FPS if metadata invalid
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N frames

# Video encoding
USE_FFMPEG = True              # Use ffmpeg for re-encoding
FFMPEG_CODEC = "libx264"       # H.264 codec
FFMPEG_PIXEL_FORMAT = "yuv420p"  # Compatible pixel format
FFMPEG_MOVFLAGS = "faststart"    # Enable streaming


# Performance Settings

USE_GPU = True                


# Logging

LOG_LEVEL = "INFO"             # DEBUG, INFO, WARNING, ERROR
LOG_FILE = os.path.join(BASE_DIR, "night_guidance.log")
LOG_DETECTIONS = True          # Log all detections to file


# Safety Settings

ENABLE_DANGER_ALERTS = True    # Enable danger alert system
ALERT_COOLDOWN_S = 3.0         # Minimum time between alerts
MAX_ALERTS_PER_VIDEO = 100     # Prevent spam


# Debug Settings

DEBUG_MODE = False             # Enable debug visualizations
SAVE_INTERMEDIATE = False      # Save intermediate processing steps
PROFILE_PERFORMANCE = False    # Profile inference speed


# Visualization Settings

# Post-processing
MORPHOLOGY_KERNEL_SIZE = 3     # Kernel for morphological ops
APPLY_MORPHOLOGY = True        # Clean up masks with morphology

# HUD (Heads-Up Display)
SHOW_HUD = True                # Show statistics overlay
HUD_FONT_SCALE = 0.5           # Font size for HUD text
HUD_THICKNESS = 1              # Text thickness


# Multi-Class Settings

CLASS_NAMES = [
    "Background",
    "Road",
    "Vehicle",
    "Pedestrian"
]

# Class weights for training
# Higher weight = more importance during training
CLASS_WEIGHTS = [0.2, 2.0, 2.0, 1.0]  # Background, Road, Vehicle, Pedestrian

# BDD100K class ID mappings
VEHICLE_IDS = [13, 14, 15, 16]  # car, truck, bus, train
PERSON_IDS = [11, 12]            # person, rider


# Helper Functions


def get_device():
    """Get computation device"""
    import torch
    if USE_GPU and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("Enhanced Night Guidance System - Configuration")
    print("=" * 80)
    print(f"Model Architecture:")
    print(f"  - Input Channels:     {IN_CHANNELS}")
    print(f"  - Output Classes:     {OUT_CHANNELS} {CLASS_NAMES}")
    print(f"  - Image Size:         {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Attention Gates:    {'Enabled' if USE_ATTENTION else 'Disabled'}")
    print(f"  - Dropout:            {DROPOUT_P}")
    print()
    print(f"Training Settings:")
    print(f"  - Epochs:             {EPOCHS}")
    print(f"  - Batch Size:         {BATCH_SIZE}")
    print(f"  - Learning Rate:      {LEARNING_RATE}")
    print(f"  - Patience:           {PATIENCE}")
    print()
    print(f"Inference Settings:")
    print(f"  - Confidence Thresh:  {CONF_THRESHOLD}")
    print(f"  - Night Enhancement:  {'Enabled' if NIGHT_ENHANCE else 'Disabled'}")
    print(f"  - CLAHE Clip:         {CLAHE_CLIP}")
    print(f"  - Min Vehicle Pixels: {MIN_VEHICLE_PIXELS}")
    print(f"  - Min Person Pixels:  {MIN_PEDESTRIAN_PIXELS}")
    print()
    print(f"Paths:")
    print(f"  - Model:              {BEST_MODEL_PATH}")
    print(f"  - Uploads:            {UPLOAD_DIR}")
    print(f"  - Outputs:            {OUTPUT_DIR}")
    print()
    print(f"Device:                 {get_device()}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()