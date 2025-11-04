import cv2
import csv
import os
from ultralytics import YOLO

# CẤU HÌNH CƠ BẢN
CSV_FILE = "data/object_widths.csv"

model_coco = YOLO("model/yolov8n.pt")

model_custom = YOLO("model/weights_custom/best.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera.")
    exit()


# HÀM LƯU CHIỀU RỘNG
def save_width(label, width_cm):
    data = {}
    # Đọc dữ liệu cũ nếu có
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    data[row[0]] = float(row[1])
    if label in data:
        data[label] = (data[label] + width_cm) / 2
    else:
        data[label] = width_cm

    # Ghi lại CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for k, v in data.items():
            writer.writerow([k, v])

    print(f"Đã lưu {label}: {width_cm:.2f} cm (TB hiện tại: {data[label]:.2f} cm)")


# VÒNG LẶP CAMERA
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    results_coco = model_coco(frame, stream=True, verbose=False)
    results_custom = model_custom(frame, stream=True, verbose=False)

    for r in results_coco:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls = int(box.cls[0])
            label = model_coco.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for r in results_custom:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls = int(box.cls[0])
            label = model_custom.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Thu thập dữ liệu (YOLO kép) - Nhấn SPACE để lưu", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC → thoát
        break
    elif key == 32:  # SPACE → nhập thông tin vật thể
        label = input("Nhập tên vật thể: ").strip()
        width_cm = float(input("Nhập chiều rộng thực của vật thể (cm): ").strip())
        save_width(label, width_cm)
cap.release()
cv2.destroyAllWindows()
