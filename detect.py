import os
import torch
import numpy as np
from models.yolo import CustomYOLO

def predict_with_scratch_model():
    print("=" * 60)
    print("🧠 RUNNING INFERENCE WITH CUSTOM TRAINED YOLO WEIGHTS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join("weights", "custom_yolo_trained.pt")

    if not os.path.exists(weights_path):
        print(f"[!] Error: Weights file nahi mili at {weights_path}")
        return

    # 1. Architecture initialize karo
    model = CustomYOLO(num_classes=80).to(device)

    # 2. Apne trained weights load karo
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[*] Successfully loaded trained checkpoint from: {weights_path}")
    print(f"[*] Model trained for {checkpoint['epochs']} epochs with final loss: {checkpoint['loss']:.4f}")

    # 3. Input tensor (Sample frame)
    sample_input = torch.rand(1, 3, 640, 640).to(device)

    # 4. Forward Pass Prediction
    with torch.no_grad():
        preds = model(sample_input)

    print("\n[+] Inference Output Summary:")
    print(f" -> P3 Small Object Head Shape: {preds[0].shape}")
    print(f" -> P4 Medium Object Head Shape: {preds[1].shape}")
    print(f" -> P5 Large Object Head Shape: {preds[2].shape}")
    print("=" * 60)
    print("✅ Model inference successfully verified with your own trained weights!")
    print("=" * 60)

if __name__ == "__main__":
    predict_with_scratch_model()