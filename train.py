import os
import time
import torch
import torch.optim as optim
from models.yolo import CustomYOLO
from utils.loss import ComputeYOLOLoss
from utils.dataset import build_dataloader

def train_scratch_yolo():
    print("=" * 60)
    print("🚀 STARTING CUSTOM YOLO DEEP LEARNING TRAINING PIPELINE")
    print("=" * 60)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    # 2. Hyperparameters
    epochs = 3
    batch_size = 4
    learning_rate = 1e-3

    # 3. Model, Loss, Optimizer Initialization
    model = CustomYOLO(num_classes=80).to(device)
    criterion = ComputeYOLOLoss(num_classes=80).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. DataLoader
    train_loader = build_dataloader(batch_size=batch_size)
    print(f"[*] Loaded Dataset: {len(train_loader.dataset)} samples ({len(train_loader)} batches per epoch)")

    # 5. Training Loop
    os.makedirs("weights", exist_ok=True)
    model.train()

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        print(f"\n--- Epoch [{epoch}/{epochs}] ---")

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            preds = model(imgs)

            # Compute Loss
            loss = criterion(preds, targets)

            # Backward pass & Optimizer step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 4 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] | Current Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"--> Epoch {epoch} Completed | Average Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    total_duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Training Finished in {total_duration:.2f} seconds!")

    # 6. Save trained model weights
    save_path = os.path.join("weights", "custom_yolo_trained.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": epochs,
        "loss": avg_loss
    }, save_path)
    print(f" Trained weights saved to: {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_scratch_yolo()