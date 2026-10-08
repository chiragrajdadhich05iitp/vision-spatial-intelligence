#Custom YOLO Deep Learning Architecture from Scratch 🚀

A modular, ground-up implementation of a YOLO-style object detection network written in pure **PyTorch**. This project implements custom convolutional backbones, multi-scale feature aggregation (PANet neck), and anchor-free detection heads without relying on external high-level computer vision wrappers.

---

## 🏗️ Architecture Overview

- **Backbone:** Modified Darknet/CSP structure with 4 progressive downsampling stages (P1 through P5), residual bottlenecks, and Spatial Pyramid Pooling - Fast (SPPF) for wide receptive field capture.
- **Neck:** Feature Pyramid Network (FPN) + Path Aggregation Network (PANet) with top-down and bottom-up feature fusion layers.
- **Detection Head:** Tri-scale decoupled detection heads mapping feature maps across 80x80 (P3), 40x40 (P4), and 20x20 (P5) grids.
- **Total Parameters:** ~7.16M trainable weights.

---

## 📂 Repository Structure

```text
custom-yolo-from-scratch/
│
├── models/
│   ├── blocks.py        # ConvBlock, Bottleneck, C3, and SPPF modules
│   └── yolo.py          # Unified multi-scale YOLO model architecture
├── utils/
│   └── loss.py          # Box IoU & Multi-objective Loss computation
├── detect.py            # Model forward-pass & inference validation
├── test_architecture.py # Tensor shape verification pipeline
├── main.py              # OpenCV real-time visual tracking application
└── README.md
⚡ Quickstart
1. Verify Architecture Shapes
Bash
python test_architecture.py
2. Run Forward Inference
Bash
python detect.py
3. Run Real-Time MOT Tracking & Line Counter
Bash
python main.py

---

### Step 3: GitHub Par Push Karo

1. Apne browser mein GitHub par jao aur ek nayi repository create karo (naam de do: `custom-yolo-scratch` ya `vision-tracker-project`, repository ko **Public** rakhna aur *Add README* unchecked rakhna).
2. Terminal mein ye commands execute karo[cite: 10]:

```powershell
git add .
git commit -m "feat: complete scratch yolo architecture and inference pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main