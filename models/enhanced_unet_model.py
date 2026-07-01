"""
enhanced_unet_model.py — Night Guidance System
Enhanced U-Net with Attention Gates and Multi-Class Support

Features:
  - Attention gates for better feature focus
  - Multi-class segmentation (Road, Vehicles, Pedestrians)
  - Improved skip connections
  - Better feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate for feature refinement"""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv with residual connection"""
    
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout_p)
        
        # Residual connection
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x):
        x = self.pool(x)
        identity = self.residual(x)
        out = self.conv(x)
        return out + identity


class Up(nn.Module):
    """Upsample → Attention → concat skip → DoubleConv"""
    
    def __init__(self, in_ch, out_ch, use_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(F_g=in_ch // 2, F_l=in_ch // 2, F_int=out_ch)
        
        self.conv = DoubleConv(in_ch, out_ch)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Match spatial dimensions
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        
        # Apply attention gate
        if self.use_attention:
            skip = self.attention(g=x, x=skip)
        
        # Concatenate
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class EnhancedUNet(nn.Module):
    """
    Enhanced U-Net for multi-class segmentation
    
    Args:
        in_channels: Input channels (3 for RGB)
        out_channels: Output channels (4 for background, road, vehicles, pedestrians)
        features: Base feature count
        dropout_p: Dropout probability
        use_attention: Enable attention gates
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        features=64,
        dropout_p=0.3,
        use_attention=True,
    ):
        super().__init__()
        
        f = features
        
        # Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2, dropout_p=0.1)
        self.down2 = Down(f * 2, f * 4, dropout_p=0.2)
        self.down3 = Down(f * 4, f * 8, dropout_p=0.2)
        
        # Bottleneck with heavy dropout
        self.bottleneck = nn.Sequential(
            Down(f * 8, f * 16, dropout_p=0.3),
            nn.Dropout2d(p=dropout_p),
        )
        
        # Decoder with attention
        self.up1 = Up(f * 16, f * 8, use_attention=use_attention)
        self.up2 = Up(f * 8, f * 4, use_attention=use_attention)
        self.up3 = Up(f * 4, f * 2, use_attention=use_attention)
        self.up4 = Up(f * 2, f, use_attention=use_attention)
        
        # Output layer
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        return self.outc(x)


if __name__ == "__main__":
    # Test model
    model = EnhancedUNet(3, 4, features=64, dropout_p=0.3, use_attention=True)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {total:,} (trainable: {trainable:,})")