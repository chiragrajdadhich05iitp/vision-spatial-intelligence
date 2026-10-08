import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from utils.database import StoreAnalyticsDB

# 1. Database & Tracker Initialization
db = StoreAnalyticsDB()
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

# Configuration for Store Analytics
line_y = 300
MAX_CAPACITY = 2  # Testing threshold for capacity breach alert
tracked_history = {}  # {track_id: [prev_y, status]}

prev_time = 0

print("Retail SaaS Vision Engine Running... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.resize(frame, (960, 600))
    h, w, _ = frame.shape

    # Real-Time Tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Spatial Reference Line (Entry/Exit Boundary)
    cv2.line(frame, (0, line_y), (w, line_y), (255, 200, 0), 2)
    cv2.putText(frame, "ENTRY / EXIT THRESHOLD", (30, line_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, track_ids, class_ids):
            label = model.names[cls]
            if label != "person":
                continue  # Filter only human traffic

            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Draw Customer Bounding Box & Centroid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 2)

            # Directional Boundary Crossing Logic (Top-to-Bottom: IN, Bottom-to-Top: OUT)
            if track_id in tracked_history:
                prev_y, has_logged = tracked_history[track_id]

                if not has_logged:
                    # Crossed downwards -> Customer enters
                    if prev_y < line_y and cy >= line_y:
                        db.log_event(track_id, "IN")
                        tracked_history[track_id] = (cy, True)
                    # Crossed upwards -> Customer leaves
                    elif prev_y > line_y and cy <= line_y:
                        db.log_event(track_id, "OUT")
                        tracked_history[track_id] = (cy, True)
                    else:
                        tracked_history[track_id] = (cy, False)
                else:
                    tracked_history[track_id] = (cy, True)
            else:
                tracked_history[track_id] = (cy, False)

    # Fetch live metrics from DB
    total_in, total_out, occupancy = db.get_live_metrics()

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Enterprise Analytics Overlay Card
    cv2.rectangle(frame, (20, 20), (380, 150), (20, 20, 20), -1)
    cv2.rectangle(frame, (20, 20), (380, 150), (100, 100, 100), 1)

    cv2.putText(frame, "RETAIL SENSE AI - LIVE BI", (35, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(frame, f"Occupancy: {occupancy} | Capacity: {MAX_CAPACITY}", (35, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Total In: {total_in}  |  Total Out: {total_out}", (35, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    cv2.putText(frame, f"Engine FPS: {fps:.1f}", (35, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Automated Alert Trigger on Capacity Breach
    if occupancy > MAX_CAPACITY:
        cv2.rectangle(frame, (w - 320, 20), (w - 20, 75), (0, 0, 220), -1)
        cv2.putText(frame, "ALERT: OVERCAPACITY", (w - 305, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("RetailSense Enterprise Vision Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()