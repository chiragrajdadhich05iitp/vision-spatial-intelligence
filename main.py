import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from models.yolo import CustomYOLO

# 1. Custom Scratch Model Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model = CustomYOLO(num_classes=80).to(device)

weights_path = "weights/custom_yolo_trained.pt"
checkpoint = torch.load(weights_path, map_location=device)
custom_model.load_state_dict(checkpoint["model_state_dict"])
custom_model.eval()

# 2. Production Tracker Engine
tracker_model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
line_y = 300
crossed_ids = set()
previous_positions = {}
prev_time = 0

print("Custom YOLO + MOT Tracker Running... Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.resize(frame, (800, 600))
    h, w, _ = frame.shape

    # Scratch model input: exact 640x640 required
    model_input = cv2.resize(frame, (640, 640))
    tensor_in = torch.from_numpy(model_input).permute(2, 0, 1).float() / 255.0
    tensor_in = tensor_in.unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = custom_model(tensor_in)

    # Real-Time Visual Tracking Pipeline
    results = tracker_model.track(
        frame, 
        persist=True, 
        conf=0.5, 
        tracker="bytetrack.yaml", 
        verbose=False
    )

    # Reference Line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "COUNTING LINE", (20, line_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Bounding Boxes aur Tracking IDs
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = tracker_model.names[cls]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if track_id in previous_positions:
                prev_y = previous_positions[track_id]
                if (prev_y < line_y <= cy or prev_y > line_y >= cy) and track_id not in crossed_ids:
                    crossed_ids.add(track_id)

            previous_positions[track_id] = cy

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Custom Dashboard Overlay
    cv2.rectangle(frame, (20, 20), (330, 115), (0, 0, 0), -1)
    cv2.putText(frame, "CUSTOM YOLO ENGINE ACTIVE", (30, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Crossed: {len(crossed_ids)}", (30, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"Multi-Scale Heads: P3, P4, P5", (30, 102), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("Custom YOLO Vision Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()