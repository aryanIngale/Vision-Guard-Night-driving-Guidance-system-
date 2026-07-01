"""
Enhanced_Inference.py — Night Guidance System
Multi-class detection without arrows

"""

import os
import sys
import cv2
import time
import torch
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        IMAGE_SIZE, CONF_THRESHOLD,
        COLOR_ROAD, COLOR_CAR, COLOR_PERSON, COLOR_WARNING, COLOR_CRITICAL,
        NIGHT_ENHANCE, CLAHE_CLIP, CLAHE_GRID,
        ROAD_ALPHA, VEHICLE_ALPHA, PEDESTRIAN_ALPHA,
        MIN_VEHICLE_PIXELS, MIN_PEDESTRIAN_PIXELS,
    )
except ImportError:
    # Fallback defaults
    IMAGE_SIZE = 256
    CONF_THRESHOLD = 0.85
    COLOR_ROAD = (46, 125, 79)
    COLOR_CAR = (140, 90, 58)
    COLOR_PERSON = (43, 57, 192)
    COLOR_WARNING = (0, 128, 255)
    COLOR_CRITICAL = (0, 0, 255)
    NIGHT_ENHANCE = True
    CLAHE_CLIP = 2.0
    CLAHE_GRID = (8, 8)
    ROAD_ALPHA = 0.30
    VEHICLE_ALPHA = 0.25
    PEDESTRIAN_ALPHA = 0.40
    MIN_VEHICLE_PIXELS = 300
    MIN_PEDESTRIAN_PIXELS = 200


# Model Loading


_model_cache = {}

def load_model(path, device=None, use_enhanced=True):
    """Load model with caching"""
    global _model_cache
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (path, device, use_enhanced)
    
    if key not in _model_cache:
        if use_enhanced:
            try:
                from enhanced_unet_model import EnhancedUNet
                model = EnhancedUNet(3, 4, features=64, dropout_p=0.3, use_attention=True).to(device)
                print("[Enhanced Inference] Using EnhancedUNet with attention gates")
            except ImportError:
                print("[Enhanced Inference] EnhancedUNet not found, falling back to standard UNet")
                from unet_model import UNet
                model = UNet(3, 4, dropout_p=0.3).to(device)
        else:
            from unet_model import UNet
            model = UNet(3, 4, dropout_p=0.3).to(device)
        
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            state = ckpt.get("model", ckpt)
            
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as e:
                print(f"[Enhanced Inference] Loading with strict=False due to: {e}")
                model.load_state_dict(state, strict=False)
            
            print(f"[Enhanced Inference] Model loaded from {path}")
        else:
            print(f"[Enhanced Inference] WARNING: {path} not found — using random weights")
        
        model.eval()
        _model_cache[key] = (model, device)
    
    return _model_cache[key]



# Image Processing


def enhance_night(frame_bgr, clip=CLAHE_CLIP):
    """Apply CLAHE enhancement for night scenes"""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def estimate_distance(mask, img_height):
    """Estimate distance based on vertical position in frame"""
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    
    # Use bottom-most point
    max_y = coords[:, 0].max()
    
    # Normalize to 0-1 (bottom = 0, top = 1)
    norm = max_y / img_height
    
    # Exponential mapping: closer objects appear lower
    # 2m at bottom, 50m at top
    distance = 2 + (50 - 2) * (1 - norm) ** 2
    return round(distance, 1)



# Detection Filtering


def filter_pedestrian_detections(person_mask, min_pixels=MIN_PEDESTRIAN_PIXELS):
    """
    Filter pedestrian detections to reduce clutter
    Returns list of valid detections with masks and bboxes
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        person_mask.astype(np.uint8), connectivity=8
    )
    
    detections = []
    
    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_pixels:
            continue
        
        # Extract bounding box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Extract component mask
        component_mask = (labels == i).astype(np.uint8)
        
        detections.append({
            'mask': component_mask,
            'bbox': (x, y, w, h),
            'area': area,
            'centroid': centroids[i]
        })
    
    return detections



# Main Processing


def process_frame(model, device, frame_bgr,
                  threshold=CONF_THRESHOLD,
                  night_enhance=NIGHT_ENHANCE,
                  clahe_clip=CLAHE_CLIP,
                  show_arrows=True,  # Kept for API compatibility but not used
                  show_danger_zones=True):
    """
    Process a single frame
    Returns overlay and statistics
    """
    
    t_start = time.time()
    h_orig, w_orig = frame_bgr.shape[:2]
    
    # Night enhancement
    if night_enhance:
        frame_bgr = enhance_night(frame_bgr, clahe_clip)
    
    # Prepare input
    img = cv2.resize(frame_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    
    # Resize masks back to original resolution
    road_prob = cv2.resize(probs[1], (w_orig, h_orig))
    vehicle_prob = cv2.resize(probs[2], (w_orig, h_orig))
    person_prob = cv2.resize(probs[3], (w_orig, h_orig))
    
    # Apply threshold
    road_mask = (road_prob > threshold).astype(np.uint8)
    vehicle_mask = (vehicle_prob > threshold).astype(np.uint8)
    person_mask = (person_prob > threshold).astype(np.uint8)
    
    latency_ms = (time.time() - t_start) * 1000
    
    # Create overlay
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]
    
    # Draw road
    road_overlay = overlay.copy()
    road_overlay[road_mask == 1] = COLOR_ROAD
    overlay = cv2.addWeighted(overlay, 1 - ROAD_ALPHA, road_overlay, ROAD_ALPHA, 0)
    
    # Detect and draw vehicles
    num_vehicles = 0
    vehicle_distances = []
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        vehicle_mask, connectivity=8
    )
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < MIN_VEHICLE_PIXELS:
            continue
        
        num_vehicles += 1
        
        # Extract bounding box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Component mask
        mask = (labels == i).astype(np.uint8)
        
        # Estimate distance
        dist = estimate_distance(mask, h)
        
        # Determine color
        color = COLOR_WARNING if (dist and dist < 10) else COLOR_CAR
        
        if dist:
            vehicle_distances.append(dist)
        
        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), color, 2)
        
        # Draw label
        text = f"CAR {dist}m" if dist else "Vehicle"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        cv2.rectangle(overlay, (x, y - text_h - 8), (x + text_w + 8, y), color, -1)
        cv2.putText(overlay, text, (x + 4, y - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Fill with semi-transparent color
        vehicle_overlay = overlay.copy()
        vehicle_overlay[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1 - VEHICLE_ALPHA, vehicle_overlay, VEHICLE_ALPHA, 0)
    
    # Detect and draw pedestrians
    num_pedestrians = 0
    pedestrian_distances = []
    danger_detected = False
    
    # Use improved filtering
    pedestrian_detections = filter_pedestrian_detections(person_mask, MIN_PEDESTRIAN_PIXELS)
    
    for detection in pedestrian_detections:
        mask = detection['mask']
        x, y, w_box, h_box = detection['bbox']
        
        num_pedestrians += 1
        
        # Estimate distance
        dist = estimate_distance(mask, h)
        
        # Determine danger level
        color = COLOR_WARNING
        label = "Person"
        
        if dist and dist < 5:
            color = COLOR_CRITICAL
            label = "Danger"
            danger_detected = True
        elif dist and dist < 10:
            color = COLOR_WARNING
        
        if dist:
            pedestrian_distances.append(dist)
        
        # Draw bounding box with rounded corners for better visibility
        cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), color, 3)
        
        # Draw label with background
        text = f"{label} {dist}m" if dist else label
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(overlay, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        cv2.putText(overlay, text, (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Fill with semi-transparent color
        person_overlay = overlay.copy()
        person_overlay[mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1 - PEDESTRIAN_ALPHA, person_overlay, PEDESTRIAN_ALPHA, 0)
    
    # Danger Zone Alert
    if show_danger_zones and danger_detected:
        # Draw danger warning
        # Smaller text
        warning_text = "DANGER - Go Slow !"

        font_scale = 0.6
        thickness = 2

        (tw, th), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Get frame width
        h, w = overlay.shape[:2]

        # Position (top-right with margin)
        x1 = w - tw - 20
        y1 = 20
        x2 = w - 10
        y2 = y1 + th + 10

        # Draw background (slightly transparent)
        danger_overlay = overlay.copy()
        cv2.rectangle(danger_overlay, (x1 - 10, y1), (x2, y2), (0, 0, 0), -1)

        overlay = cv2.addWeighted(danger_overlay, 0.6, overlay, 0.4, 0)

        # Border
        cv2.rectangle(overlay, (x1 - 10, y1), (x2, y2), COLOR_CRITICAL, 1)

        # Text
        cv2.putText(overlay, warning_text, (x1, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_CRITICAL, thickness, cv2.LINE_AA)
                
        # Add red tint to top of frame
        danger_tint = overlay.copy()
        danger_tint[0:h//4, :] = COLOR_CRITICAL
        overlay = cv2.addWeighted(overlay, 0.85, danger_tint, 0.15, 0)
    
    # HUD / Stats Display
    # Background for HUD
    hud_height = 100
    hud_width = 180
    hud_overlay = overlay.copy()

    cv2.rectangle(hud_overlay, (5, 5), (hud_width, hud_height), (0, 0, 0), -1)
    # Blend (0.0 = fully transparent, 1.0 = solid)
    alpha = 0.5
    overlay = cv2.addWeighted(hud_overlay, alpha, overlay, 1 - alpha, 0)
    cv2.rectangle(overlay, (5, 5), (hud_width, hud_height), (255, 255, 255), 1)
    
    def put_hud(text, pos, color=(255, 255, 255)):
        cv2.putText(overlay, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    road_pct = (road_mask.sum() / (h * w)) * 100
    
    put_hud(f"Latency: {latency_ms:.1f}ms", (10, 20))
    put_hud(f"Road: {road_pct:.1f}%", (10, 35))
    put_hud(f"Vehicles: {num_vehicles}", (10, 50), COLOR_CAR)
    put_hud(f"Pedestrians: {num_pedestrians}", (10, 65), COLOR_PERSON)

    if danger_detected:
        put_hud("STATUS: DANGER", (10, 80), COLOR_CRITICAL)
    else:
        put_hud("STATUS: SAFE", (10, 80), (0, 255, 0))
    
    # Return Results
    stats = {
        "latency_ms": round(latency_ms, 2),
        "road_pct": round(road_pct, 2),
        "num_vehicles": num_vehicles,
        "num_pedestrians": num_pedestrians,
        "danger_detected": danger_detected,
        "vehicle_distances": vehicle_distances,
        "pedestrian_distances": pedestrian_distances,
        "closest_vehicle": min(vehicle_distances) if vehicle_distances else None,
        "closest_pedestrian": min(pedestrian_distances) if pedestrian_distances else None,
    }
    
    return overlay, stats


def process_video(model, device, input_path, output_path,
                  threshold=CONF_THRESHOLD, night_enhance=NIGHT_ENHANCE,
                  clahe_clip=CLAHE_CLIP, progress_cb=None,
                  show_arrows=True, show_danger_zones=True):  # Kept for API compatibility
    """
    Process entire video file
    
    Returns:
        Dictionary with aggregated statistics and output file path
    """
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output
    temp_path = output_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
    
    # Aggregate stats
    agg = {
        "frames": 0,
        "total_vehicles": 0,
        "total_pedestrians": 0,
        "danger_count": 0,
        "avg_latency_ms": 0.0,
        "avg_road_pct": 0.0,
    }
    
    frame_idx = 0
    
    print(f"\n[Enhanced Inference] Processing video: {input_path}")
    print(f"  Resolution: {w}x{h} @ {fps:.1f} FPS")
    print(f"  Total frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        overlay, stats = process_frame(
            model, device, frame,
            threshold=threshold,
            night_enhance=night_enhance,
            clahe_clip=clahe_clip,
            show_arrows=show_arrows,
            show_danger_zones=show_danger_zones,
        )
        
        writer.write(overlay)
        frame_idx += 1
        
        # Aggregate
        agg["avg_latency_ms"] += stats["latency_ms"]
        agg["avg_road_pct"] += stats["road_pct"]
        agg["total_vehicles"] += stats["num_vehicles"]
        agg["total_pedestrians"] += stats["num_pedestrians"]
        if stats["danger_detected"]:
            agg["danger_count"] += 1
        
        # Progress callback
        if progress_cb and frame_idx % 10 == 0:
            progress = (frame_idx / max(total_frames, 1)) * 100
            progress_cb(progress)
    
    cap.release()
    writer.release()
    
    # Re-encode with ffmpeg for browser compatibility
    print(f"\n[Enhanced Inference] Re-encoding with ffmpeg...")
    ret = os.system(
        f'ffmpeg -y -i "{temp_path}" -vcodec libx264 -pix_fmt yuv420p '
        f'-movflags faststart "{output_path}" -loglevel error'
    )
    
    if ret == 0 and os.path.exists(output_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass
        final_name = os.path.basename(output_path)
        print(f"[Enhanced Inference] Output saved: {output_path}")
    else:
        print("[Enhanced Inference] ⚠ ffmpeg failed, using raw output")
        final_name = os.path.basename(temp_path)
    
    # Compute averages
    if frame_idx > 0:
        agg["avg_latency_ms"] = round(agg["avg_latency_ms"] / frame_idx, 2)
        agg["avg_road_pct"] = round(agg["avg_road_pct"] / frame_idx, 2)
    
    agg["frames"] = frame_idx
    agg["output_file"] = final_name
    agg["fps"] = fps
    agg["resolution"] = f"{w}x{h}"
    
    print(f"\n[Enhanced Inference] Processing complete!")
    print(f"  Frames: {frame_idx}")
    print(f"  Vehicles detected: {agg['total_vehicles']}")
    print(f"  Pedestrians detected: {agg['total_pedestrians']}")
    print(f"  Danger events: {agg['danger_count']}")
    
    return agg


if __name__ == "__main__":
    print("Enhanced Inference - Module loaded successfully")
    print("\nFeatures:")
    print("  Multi-class segmentation (Road, Vehicles, Pedestrians)")
    print("  Night vision enhancement (CLAHE)")
    print("  Distance estimation")
    print("  Danger zone detection")