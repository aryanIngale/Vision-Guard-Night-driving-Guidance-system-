"""
losses.py — Night Guidance System
Loss functions and evaluation metrics for multi-class segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# Multi-Class Dice Loss


class MultiClassDiceLoss(nn.Module):
    """
    Dice loss for multi-class segmentation
    
    Args:
        num_classes: Number of classes (including background)
        weights: Class weights (None for equal weighting)
        smooth: Smoothing factor
    """
    
    def __init__(self, num_classes=4, weights=None, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
        if weights is None:
            self.weights = torch.ones(num_classes)
        else:
            self.weights = torch.tensor(weights)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw model outputs
            targets: (B, H, W) - class indices
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Move weights to same device
        weights = self.weights.to(probs.device)
        
        # Compute dice for each class
        dice_scores = []
        
        for c in range(self.num_classes):
            prob_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            
            # Flatten
            prob_flat = prob_c.contiguous().view(-1)
            target_flat = target_c.contiguous().view(-1)
            
            intersection = (prob_flat * target_flat).sum()
            union = prob_flat.sum() + target_flat.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice * weights[c])
        
        # Average weighted dice scores
        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        return dice_loss



# Combined Cross-Entropy + Dice Loss


class CombinedLoss(nn.Module):
    """
    Combines Cross-Entropy and Dice loss for multi-class segmentation
    
    Args:
        num_classes: Number of classes
        ce_weight: Class weights for CE loss
        alpha: Balance between CE and Dice (0.5 = equal)
        dice_weight: Class weights for Dice loss
    """
    
    def __init__(self, num_classes=4, ce_weight=None, alpha=0.5, dice_weight=None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = MultiClassDiceLoss(num_classes=num_classes, weights=dice_weight)
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * ce_loss + (1.0 - self.alpha) * dice_loss



# Focal Loss (for class imbalance)


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class segmentation
    Helps with hard examples and class imbalance
    
    Args:
        alpha: Class weights
        gamma: Focusing parameter (2.0 is standard)
    """
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W)
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Convert targets to one-hot
        num_classes = logits.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Cross-entropy
        ce = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probability of true class
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        loss = focal_weight * ce
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = (alpha.view(1, -1, 1, 1) * targets_one_hot).sum(dim=1)
            loss = alpha_t * loss
        
        return loss.mean()



# Metrics


def compute_iou(logits, targets, num_classes=4, ignore_background=True):
    """
    Compute mean IoU across classes
    
    Args:
        logits: (B, C, H, W)
        targets: (B, H, W)
        num_classes: Number of classes
        ignore_background: Whether to exclude class 0 from mean
    
    Returns:
        mean_iou: float
        per_class_iou: list of floats
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        
        ious = []
        start_class = 1 if ignore_background else 0
        
        for c in range(start_class, num_classes):
            pred_mask = (preds == c)
            target_mask = (targets == c)
            
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = (intersection / union).item()
            
            ious.append(iou)
        
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        
        return mean_iou, ious


def compute_dice(logits, targets, num_classes=4, ignore_background=True):
    """
    Compute mean Dice coefficient across classes
    
    Args:
        logits: (B, C, H, W)
        targets: (B, H, W)
        num_classes: Number of classes
        ignore_background: Whether to exclude class 0 from mean
    
    Returns:
        mean_dice: float
        per_class_dice: list of floats
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        
        dice_scores = []
        start_class = 1 if ignore_background else 0
        
        for c in range(start_class, num_classes):
            pred_mask = (preds == c).float()
            target_mask = (targets == c).float()
            
            intersection = (pred_mask * target_mask).sum()
            dice = (2.0 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-6)
            
            dice_scores.append(dice.item())
        
        mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
        
        return mean_dice, dice_scores


def compute_pixel_accuracy(logits, targets):
    """
    Compute overall pixel accuracy
    
    Args:
        logits: (B, C, H, W)
        targets: (B, H, W)
    
    Returns:
        accuracy: float
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).float().sum()
        total = targets.numel()
        return (correct / total).item()


def compute_class_accuracy(logits, targets, num_classes=4):
    """
    Compute per-class accuracy
    
    Args:
        logits: (B, C, H, W)
        targets: (B, H, W)
        num_classes: Number of classes
    
    Returns:
        per_class_acc: list of floats
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        
        accuracies = []
        
        for c in range(num_classes):
            mask = (targets == c)
            if mask.sum() == 0:
                accuracies.append(0.0)
                continue
            
            correct = ((preds == c) & mask).float().sum()
            total = mask.float().sum()
            acc = (correct / total).item()
            accuracies.append(acc)
        
        return accuracies



# Evaluation Summary


def evaluate_metrics(logits, targets, num_classes=4, class_names=None):
    """
    Compute all metrics and return as dictionary
    
    Args:
        logits: (B, C, H, W)
        targets: (B, H, W)
        num_classes: Number of classes
        class_names: Optional list of class names
    
    Returns:
        metrics: Dictionary with all computed metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    mean_iou, per_class_iou = compute_iou(logits, targets, num_classes)
    mean_dice, per_class_dice = compute_dice(logits, targets, num_classes)
    pixel_acc = compute_pixel_accuracy(logits, targets)
    class_acc = compute_class_accuracy(logits, targets, num_classes)
    
    metrics = {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "pixel_accuracy": pixel_acc,
    }
    
    # Add per-class metrics
    for i, name in enumerate(class_names):
        if i > 0:  # Skip background
            metrics[f"iou_{name}"] = per_class_iou[i-1] if i-1 < len(per_class_iou) else 0.0
            metrics[f"dice_{name}"] = per_class_dice[i-1] if i-1 < len(per_class_dice) else 0.0
        metrics[f"acc_{name}"] = class_acc[i]
    
    return metrics



# Binary compatibility (for backward compat)


class DiceLoss(nn.Module):
    """Binary Dice loss (backward compatibility)"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Binary BCE + Dice loss (backward compatibility)"""
    
    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    """Binary IoU (backward compatibility)"""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        inter = (preds * targets).sum()
        union = preds.sum() + targets.sum() - inter
        return ((inter + eps) / (union + eps)).item()


def dice_score(logits, targets, threshold=0.5, eps=1e-6):
    """Binary Dice (backward compatibility)"""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        inter = (preds * targets).sum()
        return ((2.0 * inter + eps) / (preds.sum() + targets.sum() + eps)).item()


def pixel_accuracy(logits, targets, threshold=0.5):
    """Binary pixel accuracy (backward compatibility)"""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        correct = (preds == targets).float().sum()
        total = targets.numel()
        return (correct / total).item()


if __name__ == "__main__":
    # Test multi-class losses
    batch_size = 2
    num_classes = 4
    h, w = 256, 256
    
    logits = torch.randn(batch_size, num_classes, h, w)
    targets = torch.randint(0, num_classes, (batch_size, h, w))
    
    # Test losses
    print("Testing Multi-Class Losses:")
    
    ce_weights = torch.tensor([0.2, 1.0, 2.0, 2.0])
    dice_weights = [0.2, 1.0, 2.0, 2.0]
    
    combined_loss = CombinedLoss(num_classes=num_classes, ce_weight=ce_weights, dice_weight=dice_weights)
    loss = combined_loss(logits, targets)
    print(f"Combined Loss: {loss.item():.4f}")
    
    focal_loss = FocalLoss(alpha=ce_weights, gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test metrics
    print("\nTesting Metrics:")
    metrics = evaluate_metrics(logits, targets, num_classes=4, 
                              class_names=["Background", "Road", "Vehicle", "Pedestrian"])
    
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")