import torch
import torch.nn as nn
from models.blocks import ConvBlock, C3, SPPF

class CustomYOLO(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # --- 1. BACKBONE (Feature Extraction) ---
        # Input: 3x640x640
        self.stem = ConvBlock(3, 32, kernel_size=3, stride=2)  # P1/2 (320x320)
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2),       # P2/4 (160x160)
            C3(64, 64, n=1)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2),      # P3/8 (80x80)
            C3(128, 128, n=2)
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2),     # P4/16 (40x40)
            C3(256, 256, n=3)
        )
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=2),     # P5/32 (20x20)
            C3(512, 512, n=1),
            SPPF(512, 512, k=5)
        )

        # --- 2. NECK (PANet Multi-Scale Fusion) ---
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_c3_p4 = C3(512 + 256, 256, n=1, shortcut=False)
        self.neck_c3_p3 = C3(256 + 128, 128, n=1, shortcut=False)
        
        self.down_p3 = ConvBlock(128, 128, kernel_size=3, stride=2)
        self.neck_c3_down_p4 = C3(128 + 256, 256, n=1, shortcut=False)
        self.down_p4 = ConvBlock(256, 256, kernel_size=3, stride=2)
        self.neck_c3_down_p5 = C3(256 + 512, 512, n=1, shortcut=False)

        # --- 3. DETECTION HEADS (P3, P4, P5 scales) ---
        # Har scale par bounding boxes (4 coords) + classes predict hongi
        out_dim = 4 + self.num_classes
        self.head_p3 = nn.Conv2d(128, out_dim, kernel_size=1)
        self.head_p4 = nn.Conv2d(256, out_dim, kernel_size=1)
        self.head_p5 = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, x):
        # Backbone Forward Pass
        p1 = self.stem(x)
        p2 = self.stage1(p1)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        # Neck Top-Down Pass
        p5_up = self.up(p5)
        p4_cat = torch.cat([p5_up, p4], dim=1)
        p4_out = self.neck_c3_p4(p4_cat)

        p4_up = self.up(p4_out)
        p3_cat = torch.cat([p4_up, p3], dim=1)
        p3_out = self.neck_c3_p3(p3_cat)

        # Neck Bottom-Up Pass
        p3_down = self.down_p3(p3_out)
        p4_bottom_cat = torch.cat([p3_down, p4_out], dim=1)
        p4_final = self.neck_c3_down_p4(p4_bottom_cat)

        p4_down = self.down_p4(p4_final)
        p5_bottom_cat = torch.cat([p4_down, p5], dim=1)
        p5_final = self.neck_c3_down_p5(p5_bottom_cat)

        # Multi-scale outputs
        out3 = self.head_p3(p3_out)     # Small objects (80x80 grid)
        out4 = self.head_p4(p4_final)   # Medium objects (40x40 grid)
        out5 = self.head_p5(p5_final)   # Large objects (20x20 grid)

        return [out3, out4, out5]