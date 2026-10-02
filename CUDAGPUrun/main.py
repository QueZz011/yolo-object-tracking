import json
import time

import cv2
import torch
from ultralytics import YOLO

if not torch.cuda.is_available():
    print("CUDA bulunamadi.")
    exit()

with open("config.json") as f:
    config = json.load(f)

model = YOLO(config["model"])

cap = cv2.VideoCapture(config["camera_index"], cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
if not cap.isOpened():
    print("Webcam açılamadı.")
    exit()

prev_time = time.time()
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Goruntu alinamadi.")
        break

    results = model.track(frame, persist=True, device=0, conf=config["conf"],
                          tracker=config["tracker"], verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1

        if track_id == -1:
            color = (200, 200, 200)
        else:
            color = ((track_id * 67) % 256, (track_id * 123) % 256, (track_id * 191) % 256)

        label = f"ID {track_id if track_id != -1 else '-'} {name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1 - h - 10, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, top), (x1 + w + 6, top + h + 10), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    now = time.time()
    fps_list.append(1 / (now - prev_time))
    fps_list = fps_list[-30:]
    prev_time = now
    fps = sum(fps_list) / len(fps_list)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"Nesne: {len(results[0].boxes)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLO Object Tracking (GPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
