"""
Train.py — Night Guidance System
Multi-Class Training Pipeline with Enhanced Metrics

Features:
  - Multi-class segmentation (Background, Road, Vehicles, Pedestrians)
  - Class-weighted loss
  - Comprehensive metrics (IoU, Dice, Accuracy per class)
  - Mixed precision training
  - Early stopping
  - Checkpoint management
"""

import os
import sys
import time

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_BACKUP import (
    CHECKPOINTS_DIR, BEST_MODEL_PATH,
    IN_CHANNELS, OUT_CHANNELS, IMAGE_SIZE,
    DROPOUT_P, EPOCHS, LEARNING_RATE, LR_MIN, PATIENCE, RESUME,
    USE_ATTENTION,
)

from Dataset_loader import get_loaders
from losses import CombinedLoss, compute_iou, compute_dice, compute_pixel_accuracy

# Try to use enhanced model, fallback to standard
try:
    from enhanced_unet_model import EnhancedUNet
    USE_ENHANCED = True
    print("[Train] Using EnhancedUNet with attention gates")
except ImportError:
    from unet_model import UNet
    USE_ENHANCED = False
    print("[Train] Using standard UNet")



# Checkpoint Management


def save_checkpoint(state, path):
    """Save checkpoint to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer, scheduler):
    """Load checkpoint from disk"""
    ckpt = torch.load(path, map_location="cpu")
    
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        best_iou = ckpt.get("best_iou", 0.0)
    else:
        model.load_state_dict(ckpt)
        start_epoch = 0
        best_iou = 0.0
    
    print(f"  ✓ Resumed from epoch {start_epoch} (best IoU = {best_iou:.4f})")
    return start_epoch, best_iou



# Training Loop


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    
    pbar = tqdm(loader, desc="  Train", leave=False)
    
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Compute metrics
        with torch.no_grad():
            iou, _ = compute_iou(logits, masks, num_classes=OUT_CHANNELS, ignore_background=True)
            dice, _ = compute_dice(logits, masks, num_classes=OUT_CHANNELS, ignore_background=True)
        
        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}',
        })
    
    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n



# Validation Loop


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=4):
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    
    # Per-class metrics
    class_ious = [0.0] * (num_classes - 1)  # Exclude background
    class_dices = [0.0] * (num_classes - 1)
    
    pbar = tqdm(loader, desc="  Val  ", leave=False)
    
    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, masks)
        
        # Compute metrics
        iou, per_class_iou = compute_iou(logits, masks, num_classes=num_classes, ignore_background=True)
        dice, per_class_dice = compute_dice(logits, masks, num_classes=num_classes, ignore_background=True)
        acc = compute_pixel_accuracy(logits, masks)
        
        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        total_acc += acc
        
        # Accumulate per-class metrics
        for i in range(len(per_class_iou)):
            class_ious[i] += per_class_iou[i]
            class_dices[i] += per_class_dice[i]
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{iou:.4f}',
        })
    
    n = len(loader)
    
    # Average per-class metrics
    class_ious = [x / n for x in class_ious]
    class_dices = [x / n for x in class_dices]
    
    return (
        total_loss / n,
        total_iou / n,
        total_dice / n,
        total_acc / n,
        class_ious,
        class_dices,
    )



# Main Training Function


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'=' * 80}")
    print(f"  Night Guidance System — Multi-Class Training")
    print(f"{'=' * 80}")
    print(f"  Device: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {mem:.1f} GB")
    
    print(f"  Model: {'EnhancedUNet' if USE_ENHANCED else 'Standard UNet'}")
    print(f"  Output Classes: {OUT_CHANNELS} (Background, Road, Vehicles, Pedestrians)")
    print(f"{'=' * 80}\n")
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader = get_loaders()
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}\n")
    
    # Create model
    if USE_ENHANCED:
        model = EnhancedUNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            features=64,
            dropout_p=DROPOUT_P,
            use_attention=USE_ATTENTION,
        ).to(device)
    else:
        from unet_model import UNet
        model = UNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            dropout_p=DROPOUT_P,
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})\n")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    
    # Multi-class weighted loss
    # Background=0.2, Road=1.0, Vehicles=2.0, Pedestrians=2.0
    class_weights = torch.tensor([0.2, 1.0, 2.0, 2.0]).to(device)
    criterion = CombinedLoss(
        num_classes=OUT_CHANNELS,
        ce_weight=class_weights,
        alpha=0.5,
        dice_weight=[0.2, 1.0, 2.0, 2.0],
    )
    
    scaler = GradScaler()
    
    # Resume if requested
    start_epoch = 0
    best_iou = 0.0
    patience_counter = 0
    
    if RESUME and os.path.exists(BEST_MODEL_PATH):
        start_epoch, best_iou = load_checkpoint(
            BEST_MODEL_PATH, model, optimizer, scheduler
        )
    
    # Training loop
    print(f"Training for up to {EPOCHS} epochs (patience={PATIENCE})\n")
    
    class_names = ["Road", "Vehicle", "Pedestrian"]
    
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        
        print(f"{'─' * 80}")
        print(f"Epoch [{epoch + 1:03d}/{EPOCHS}]  LR={lr:.2e}")
        print(f"{'─' * 80}")
        
        # Train
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        
        # Validate
        val_loss, val_iou, val_dice, val_acc, class_ious, class_dices = validate(
            model, val_loader, criterion, device, num_classes=OUT_CHANNELS
        )
        
        # Step scheduler
        scheduler.step()
        
        elapsed = time.time() - t0
        
        # Print results
        print(f"\n  Train Loss: {train_loss:.4f}  |  IoU: {train_iou:.4f}  |  Dice: {train_dice:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  |  IoU: {val_iou:.4f}  |  Dice: {val_dice:.4f}  |  Acc: {val_acc:.4f}")
        
        print(f"\n  Per-Class IoU:")
        for i, name in enumerate(class_names):
            print(f"    {name:12s}: {class_ious[i]:.4f}")
        
        print(f"\n  Per-Class Dice:")
        for i, name in enumerate(class_names):
            print(f"    {name:12s}: {class_dices[i]:.4f}")
        
        print(f"\n  Epoch time: {elapsed:.0f}s")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            
            print(f"\n  ✨ New best IoU: {best_iou:.4f}")
            
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_iou": best_iou,
                    "config": {
                        "in_channels": IN_CHANNELS,
                        "out_channels": OUT_CHANNELS,
                        "image_size": IMAGE_SIZE,
                        "dropout_p": DROPOUT_P,
                        "use_enhanced": USE_ENHANCED,
                        "use_attention": USE_ATTENTION,
                    },
                    "class_metrics": {
                        "class_ious": class_ious,
                        "class_dices": class_dices,
                    }
                },
                BEST_MODEL_PATH,
            )
        else:
            patience_counter += 1
            print(f"\n  No improvement ({patience_counter}/{PATIENCE})")
        
        print()
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n{'=' * 80}")
            print(f"  Early stopping triggered after {epoch + 1} epochs")
            print(f"{'=' * 80}\n")
            break
    
    # Training complete
    print(f"\n{'=' * 80}")
    print(f"  Training Complete!")
    print(f"  Best Validation IoU: {best_iou:.4f}")
    print(f"  Model saved to: {BEST_MODEL_PATH}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()