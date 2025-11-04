import cv2
import csv
import math
import threading
from playsound import playsound
from ultralytics import YOLO

FOCAL_LENGTH = 700
THRESHOLD_CM = 20
CSV_FILE = "data/object_widths.csv"

model_coco = YOLO("model/yolov8n.pt")
model_custom = YOLO("model/weights_custom/best.pt")

OBJECT_WIDTHS_CM = {}
try:
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                OBJECT_WIDTHS_CM[row[0]] = float(row[1])
except FileNotFoundError:
    print("Chưa có file object_widths.csv. Hãy chạy collect_data.py trước.")
    exit()

def play_alert():
    threading.Thread(target=lambda: playsound("alert.wav")).start()

def calculate_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width if per_width > 0 else None

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    results_custom = list(model_custom(frame, stream=True, verbose=False))
    results_coco = list(model_coco(frame, stream=True, verbose=False))

    detections_custom = []
    detections_coco = []

    for r in results_custom:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls = int(box.cls[0])
            label = model_custom.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections_custom.append({
                "label": label,
                "conf": conf,
                "bbox": (x1, y1, x2, y2)
            })

    for r in results_coco:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls = int(box.cls[0])
            label = model_coco.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections_coco.append({
                "label": label,
                "conf": conf,
                "bbox": (x1, y1, x2, y2)
            })

    final_boxes = []

    for det_c in detections_custom:
        label = det_c["label"].lower()
        conf = det_c["conf"]
        if label == "calculator" and conf >= 0.7:
            final_boxes.append({
                "label": det_c["label"],
                "conf": conf,
                "bbox": det_c["bbox"],
                "model": "custom"
            })

    for det_coco in detections_coco:
        x1, y1, x2, y2 = det_coco["bbox"]
        skip = False

        for f in final_boxes:
            if f["label"].lower() == "calculator" and iou(det_coco["bbox"], f["bbox"]) > 0.5:
                skip = True
                break
        if skip:
            continue

        for det_c in detections_custom:
            if iou(det_c["bbox"], det_coco["bbox"]) > 0.5:
                if det_c["conf"] >= det_coco["conf"]:
                    final_boxes.append({
                        "label": det_c["label"],
                        "conf": det_c["conf"],
                        "bbox": det_c["bbox"],
                        "model": "custom"
                    })
                else:
                    final_boxes.append({
                        "label": det_coco["label"],
                        "conf": det_coco["conf"],
                        "bbox": det_coco["bbox"],
                        "model": "coco"
                    })
                break
        else:
            final_boxes.append({
                "label": det_coco["label"],
                "conf": det_coco["conf"],
                "bbox": det_coco["bbox"],
                "model": "coco"
            })

    for det in final_boxes:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]
        width_px = x2 - x1

        color = (0, 255, 0)
        if det["model"] == "custom":
            color = (255, 0, 0)
        if label in OBJECT_WIDTHS_CM:
            distance_cm = calculate_distance(OBJECT_WIDTHS_CM[label], FOCAL_LENGTH, width_px)
            if distance_cm and distance_cm < THRESHOLD_CM:
                color = (0, 0, 255)
                cv2.putText(frame, "CANH BAO", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                play_alert()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {distance_cm:.1f} cm ({conf*100:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.putText(frame, f"{label} (no data {conf*100:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

    cv2.imshow("Đo khoảng cách (2 mô hình YOLO)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
