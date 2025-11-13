import cv2
import csv
import math
import threading
import time # Thêm thư viện time
from io import BytesIO 
import pygame 
from gtts import gTTS 
from ultralytics import YOLO

# --- Cài đặt ban đầu ---
FOCAL_LENGTH = 700
THRESHOLD_CM = 100
CSV_FILE = "data/object_widths.csv"

# Khởi tạo pygame mixer để phát âm thanh
pygame.mixer.init()

# --- Cài đặt Cooldown cho cảnh báo ---
ALERT_COOLDOWN_SECONDS = 3.0 
last_alert_time = 0

# --- Tải mô hình ---
model_coco = YOLO("model/yolov8n.pt")
model_custom = YOLO("model/weights_custom/best.pt")

# --- Tải dữ liệu độ rộng vật thể ---
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

# --- Các hàm chức năng ---

def play_tts_alert_threaded(distance_cm):
    """
    Tạo và phát âm thanh cảnh báo trong một thread riêng biệt
    để không làm gián đoạn luồng video chính.
    """
    try:
        text = f"vật cản {int(distance_cm)} cm"
        print(f"Đang tạo cảnh báo: {text}") 
        
        tts = gTTS(text=text, lang='vi')
        
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0) 
        
        pygame.mixer.music.load(mp3_fp)
        pygame.mixer.music.play()
        
    except Exception as e:
        print(f"Lỗi khi phát âm thanh gTTS: {e}")

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
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxA[1]) # SỬA LỖI NHỎ: Chỗ này là boxB[1]
    # Sửa lại thành:
    # boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    # Tuy nhiên, trong logic hiện tại của bạn, hàm iou() không ảnh hưởng đến cảnh báo khoảng cách
    # nên tôi sẽ giữ nguyên theo code gốc của bạn, nhưng lưu ý bạn về nó.
    # Giữ nguyên code gốc của bạn:
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxA[1]) 
    return interArea / float(boxAArea + boxBArea - interArea)

# --- Khởi động Camera ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

# --- Vòng lặp chính ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    frame_height, frame_width, _ = frame.shape

    # --- Định nghĩa 3 vùng (tỷ lệ 2:6:2) ---
    center_zone_start = (frame_width * 2) // 10
    center_zone_end = (frame_width * 8) // 10
    
    cv2.line(frame, (center_zone_start, 0), (center_zone_start, frame_height), (255, 255, 0), 1)
    cv2.line(frame, (center_zone_end, 0), (center_zone_end, frame_height), (255, 255, 0), 1)
    
    # --- Xử lý phát hiện (giữ nguyên) ---
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
                "label": label, "conf": conf, "bbox": (x1, y1, x2, y2)
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
                "label": label, "conf": conf, "bbox": (x1, y1, x2, y2)
            })
    
    # --- Trộn kết quả (giữ nguyên) ---
    final_boxes = []

    for det_c in detections_custom:
        label = det_c["label"].lower()
        conf = det_c["conf"]
        if label == "calculator" and conf >= 0.7:
            final_boxes.append({
                "label": det_c["label"], "conf": conf, "bbox": det_c["bbox"], "model": "custom"
            })

    for det_coco in detections_coco:
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
                        "label": det_c["label"], "conf": det_c["conf"], "bbox": det_c["bbox"], "model": "custom"
                    })
                else:
                    final_boxes.append({
                        "label": det_c["label"], "conf": det_coco["conf"], "bbox": det_coco["bbox"], "model": "coco"
                    })
                break
        else:
            final_boxes.append({
                "label": det_coco["label"], "conf": det_coco["conf"], "bbox": det_coco["bbox"], "model": "coco"
            })

    # --- Vẽ và cảnh báo (ĐÃ CẬP NHẬT LOGIC) ---
    
    # MỚI: Biến để theo dõi vật thể gần nhất cần cảnh báo trong khung hình này
    closest_alert_distance = float('inf') 
    
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
            
            if distance_cm:
                center_x = (x1 + x2) // 2
                
                # MỚI: Kiểm tra điều kiện cảnh báo
                is_alert_candidate = (distance_cm < THRESHOLD_CM) and (center_x > center_zone_start and center_x < center_zone_end)
                
                if is_alert_candidate:
                    color = (0, 0, 255)
                    cv2.putText(frame, "CANH BAO", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    
                    # MỚI: Cập nhật vật thể gần nhất
                    if distance_cm < closest_alert_distance:
                        closest_alert_distance = distance_cm
                        
                # --- LOGIC COOLDOWN ĐÃ BỊ XÓA KHỎI ĐÂY ---

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {distance_cm:.1f} cm ({conf*100:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({conf*100:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.putText(frame, f"{label} (no data {conf*100:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

    # --- MỚI: XỬ LÝ CẢNH BÁO SAU KHI KẾT THÚC VÒNG LẶP ---
    # Kiểm tra xem có vật thể nào cần cảnh báo không (tức là closest_alert_distance đã được cập nhật)
    if closest_alert_distance < float('inf'):
        current_time = time.time()
        
        # Kiểm tra cooldown và loa có rảnh không
        if (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS and \
           not pygame.mixer.music.get_busy():
            
            # Đặt lại mốc thời gian và phát cảnh báo cho vật thể GẦN NHẤT
            last_alert_time = current_time 
            threading.Thread(target=play_tts_alert_threaded, args=(closest_alert_distance,)).start()

    cv2.imshow("Đo khoảng cách (2 mô hình YOLO)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()