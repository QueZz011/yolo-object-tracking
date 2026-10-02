import json
import time

import cv2
from ultralytics import YOLO

with open("config.json") as f:
    config = json.load(f)

model = YOLO(config["model"])

cap = cv2.VideoCapture(config["camera_index"], cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
if not cap.isOpened():
    print("Webcam acilamadi.")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı.")
        break

    results = model.track(frame, persist=True, conf=config["conf"],
                          tracker=config["tracker"], verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else "-"

        label = f"ID {track_id} {name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("YOLO Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
