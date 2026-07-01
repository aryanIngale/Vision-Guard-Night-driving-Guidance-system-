"""
evaluate.py  —  Night Guidance System
Evaluate a trained model on the validation set.

Run:
    python src/evaluate.py

Outputs:
  - IoU, Dice, Pixel Accuracy (overall)
  - Per-class breakdown (road vs background)
  - Saves a few sample overlay images to checkpoints/eval_samples/
"""

import os
import sys

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_BACKUP       import (
    BEST_MODEL_PATH, CHECKPOINTS_DIR,
    VAL_IMG_DIR, VAL_MASK_DIR,
    IN_CHANNELS, OUT_CHANNELS, IMAGE_SIZE, DROPOUT_P,
    CONF_THRESHOLD, COLOR_ROAD,
)
from unet_model    import UNet
from dataset_loader import BDDDataset
from losses        import iou_score, dice_score, pixel_accuracy


EVAL_SAMPLE_DIR = os.path.join(CHECKPOINTS_DIR, "eval_samples")
os.makedirs(EVAL_SAMPLE_DIR, exist_ok=True)
NUM_SAMPLES_TO_SAVE = 10



# Load model


def load_model(path, device):
    model = UNet(IN_CHANNELS, OUT_CHANNELS, dropout_p=DROPOUT_P).to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model



# Save sample overlay


def save_sample(img_tensor, mask_tensor, pred_tensor, idx):
    """Save side-by-side: original | ground-truth | prediction."""

    def t2np(t):
        return t.squeeze().cpu().numpy()

    img  = (t2np(img_tensor).transpose(1, 2, 0) * 255).astype(np.uint8)
    img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gt   = (t2np(mask_tensor) * 255).astype(np.uint8)
    pred = (t2np(pred_tensor) * 255).astype(np.uint8)

    # Colour overlays
    gt_color   = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_color = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    green_mask = np.zeros_like(img)
    green_mask[gt > 127]   = COLOR_ROAD
    gt_color   = cv2.addWeighted(img, 0.6, green_mask, 0.4, 0)

    pred_overlay = np.zeros_like(img)
    pred_overlay[pred > 127] = COLOR_ROAD
    pred_color = cv2.addWeighted(img, 0.6, pred_overlay, 0.4, 0)

    combined = np.concatenate([img, gt_color, pred_color], axis=1)

    # Labels
    for i, label in enumerate(["Original", "Ground Truth", "Prediction"]):
        x = i * IMAGE_SIZE + 8
        cv2.putText(combined, label, (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = os.path.join(EVAL_SAMPLE_DIR, f"sample_{idx:03d}.png")
    cv2.imwrite(out_path, combined)



# Main evaluation


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Night Guidance System — Evaluation")
    print(f"  Device : {device}")
    print(f"  Model  : {BEST_MODEL_PATH}")
    print(f"{'='*55}\n")

    if not os.path.exists(BEST_MODEL_PATH):
        print("ERROR: No trained model found. Run src/train.py first.")
        sys.exit(1)

    model  = load_model(BEST_MODEL_PATH, device)
    val_ds = BDDDataset(VAL_IMG_DIR, VAL_MASK_DIR, augment=False)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    total_iou  = 0.0
    total_dice = 0.0
    total_acc  = 0.0
    saved      = 0
    n_batches  = len(loader)

    with torch.no_grad():
        for b_idx, (imgs, masks) in enumerate(tqdm(loader, desc="Evaluating")):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            preds  = (torch.sigmoid(logits) > CONF_THRESHOLD).float()

            total_iou  += iou_score(logits, masks, CONF_THRESHOLD)
            total_dice += dice_score(logits, masks, CONF_THRESHOLD)
            total_acc  += pixel_accuracy(logits, masks, CONF_THRESHOLD)

            # Save samples from first few batches
            if saved < NUM_SAMPLES_TO_SAVE:
                for i in range(min(imgs.size(0), NUM_SAMPLES_TO_SAVE - saved)):
                    save_sample(imgs[i], masks[i], preds[i], saved)
                    saved += 1

    iou  = total_iou  / n_batches
    dice = total_dice / n_batches
    acc  = total_acc  / n_batches

    print(f"\n{'─'*40}")
    print(f"  Val IoU          : {iou:.4f}  ({iou*100:.2f}%)")
    print(f"  Val Dice / F1    : {dice:.4f}  ({dice*100:.2f}%)")
    print(f"  Val Pixel Acc    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'─'*40}")
    print(f"  {NUM_SAMPLES_TO_SAVE} sample overlays → {EVAL_SAMPLE_DIR}")
    print()


if __name__ == "__main__":
    main()