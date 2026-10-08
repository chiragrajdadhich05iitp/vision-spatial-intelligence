import torch
from models.yolo import CustomYOLO

if __name__ == "__main__":
    print("Initializing Custom YOLO Architecture from Scratch...")
    model = CustomYOLO(num_classes=80)
    
    # Batch of 2 images with 3 channels, 640x640 resolution
    dummy_input = torch.randn(2, 3, 640, 640)
    outputs = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f} Million")
    print("Multi-Scale Head Output Shapes:")
    for idx, out in enumerate(outputs):
        print(f" Scale P{idx+3} Tensor Shape: {out.shape}")