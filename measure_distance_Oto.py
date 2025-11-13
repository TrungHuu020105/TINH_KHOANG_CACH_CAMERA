#!/usr/bin/env python3
"""
lane_and_object_detection.py

Phát hiện vạch kẻ đường VÀ vật thể (xe) trong video hoặc webcam.
Cảnh báo va chạm dựa trên vị trí làn đường và khoảng cách.

Algorithm:
 - Resize ảnh về kích thước chuẩn (960x540).
 - (Lane) Chuyển ảnh sang grayscale, Gaussian blur, Canny, ROI.
 - (Lane) HoughLinesP để tìm các đoạn thẳng.
 - (Lane) Tách trái/phải, trung bình hóa, nội suy thành 2 đường -> lấy (slope, intercept).
 - (Lane) Vẽ 2 đường này lên ảnh.
 - (YOLO) Chạy YOLOv8 (custom + coco) trên ảnh gốc đã resize.
 - (YOLO) Lọc và gộp kết quả (NMS) -> final_boxes.
 - (Logic) Với mỗi box:
     - Tính khoảng cách dùng FOCAL_LENGTH.
     - Tính tâm đáy (bottom_center_x, bottom_y).
     - Dùng (slope, intercept) của làn để tìm (lane_x_left, lane_x_right) tại y = bottom_y.
     - So sánh bottom_center_x với 2 mốc x của làn -> Phân loại "Left", "Right", "Ahead".
     - Áp dụng ngưỡng (10m trước, 7m hai bên) để cảnh báo.
 - (Display) Vẽ box, khoảng cách, vị trí, và cảnh báo lên ảnh đã có vạch kẻ đường.
"""
import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

# ### SỬA ĐỔI ###: Thêm CompositeAudioClip để trộn âm thanh
# THAY BẰNG DÒNG NÀY:
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips, CompositeAudioClip


import cv2
import numpy as np

# ### START: THÊM TỪ YOLO SCRIPT ###
import csv
import threading
from playsound import playsound
from ultralytics import YOLO
from gtts import gTTS

# Hằng số cho YOLO và đo khoảng cách
FOCAL_LENGTH = 700 
CSV_FILE = "data/object_widths.csv"

# Hằng số cho cảnh báo (đơn vị cm)
AHEAD_THRESHOLD_CM = 1200 
SIDE_THRESHOLD_CM = 500   
AUDIO_AHEAD_FILE = "tts_ahead.mp3"
AUDIO_LEFT_FILE = "tts_left.mp3"
AUDIO_RIGHT_FILE = "tts_right.mp3"
FRAMES_PER_ALERT_COOLDOWN = 60 

### HÀM MỚI NÀY ###
def generate_tts_files():
    """
    Kiểm tra và tạo các file âm thanh TTS nếu chúng chưa tồn tại.
    """
    print("Đang kiểm tra/tạo file âm thanh TTS...")
    files_to_generate = {
        AUDIO_AHEAD_FILE: "Cảnh báo, có xe phía trước",
        AUDIO_LEFT_FILE: "có xe ở làn bên trái",
        AUDIO_RIGHT_FILE: "có xe ở làn bên phải"
    }
    
    try:
        for filename, text in files_to_generate.items():
            if not os.path.exists(filename):
                print(f"Đang tạo {filename}...")
                tts = gTTS(text=text, lang='vi')
                tts.save(filename)
        print("Các file TTS đã sẵn sàng.")
    except Exception as e:
        print(f"Lỗi khi tạo file TTS: {e}")
        print("Vui lòng kiểm tra kết nối internet (cho lần chạy đầu) và quyền ghi file.")

# Tải mô hình
print("Loading YOLO models...")
try:
    model_coco = YOLO("model/yolov8n.pt")
    model_custom = YOLO("model/weights_custom/best.pt")
    print("YOLO models loaded.")
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLO: {e}")
    sys.exit(1)

# Tải kích thước vật thể
OBJECT_WIDTHS_CM = {}
try:
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                OBJECT_WIDTHS_CM[row[0]] = float(row[1])
    print(f"Loaded {len(OBJECT_WIDTHS_CM)} object widths from {CSV_FILE}")
except FileNotFoundError:
    print(f"Không tìm thấy file {CSV_FILE}. Cần chạy collect_data.py hoặc tạo file.")
    
# --- (Các hàm xử lý ảnh và YOLO giữ nguyên) ---
def calculate_distance(known_width, focal_length, per_width):
    if per_width == 0: return None
    return (known_width * focal_length) / per_width

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked = cv2.bitwise_and(img, mask)
    return masked

def display_lines(img: np.ndarray, lines: Optional[np.ndarray]) -> np.ndarray:
    line_image = np.zeros_like(img)
    if lines is None: return line_image
    for line in lines:
        if line is not None and len(line) == 4:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def make_coordinates(img_shape: Tuple[int, int], line_parameters: Tuple[float, float]) -> Optional[List[int]]:
    try:
        slope, intercept = line_parameters
        height = img_shape[0]
        y1 = height
        y2 = int(height * 0.6)
        x1 = int((y1 - intercept) / (slope + 1e-6))
        x2 = int((y2 - intercept) / (slope + 1e-6))
        return [x1, y1, x2, y2]
    except (OverflowError, ValueError):
        return None

def average_slope_intercept(img: np.ndarray, lines: Optional[np.ndarray]) \
        -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[np.ndarray]]:
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                if abs(slope) < 0.3: continue
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
    img_shape = img.shape
    lines_to_draw = []
    left_fit_avg = None
    right_fit_avg = None
    if left_fit:
        left_fit_avg = np.mean(left_fit, axis=0)
        left_line_coords = make_coordinates(img_shape, left_fit_avg)
        if left_line_coords:
            lines_to_draw.append(left_line_coords)
    if right_fit:
        right_fit_avg = np.mean(right_fit, axis=0)
        right_line_coords = make_coordinates(img_shape, right_fit_avg)
        if right_line_coords:
            lines_to_draw.append(right_line_coords)
    drawn_lines_array = np.array(lines_to_draw) if lines_to_draw else None
    return (left_fit_avg, right_fit_avg), drawn_lines_array

def process_frame_lanes(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]]:
    gray = grayscale(frame)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 50, 150)
    height, width = frame.shape[0], frame.shape[1]
    polygons = np.array([[
        (int(0.1 * width), height), (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)), (int(0.9 * width), height),
    ]], dtype=np.int32)
    masked = region_of_interest(edges, polygons)
    lines = cv2.HoughLinesP(masked, rho=1, theta=np.pi / 180, threshold=50, minLineLength=40, maxLineGap=100)
    fits, averaged_lines = average_slope_intercept(frame, lines)
    # Tạm thời bỏ qua vẽ làn ở đây, vì chúng ta sẽ vẽ box lên frame gốc
    # và chỉ dùng 'fits' cho logic
    return frame, fits

def run_yolo_detection(frame: np.ndarray) -> List[dict]:
    results_custom = model_custom(frame, stream=True, verbose=False)
    results_coco = model_coco(frame, stream=True, verbose=False)
    detections_custom = []
    detections_coco = []
    for r in results_custom:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5: continue
            cls = int(box.cls[0])
            label = model_custom.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections_custom.append({"label": label, "conf": conf, "bbox": (x1, y1, x2, y2)})
    for r in results_coco:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5: continue
            cls = int(box.cls[0])
            label = model_coco.names[cls]
            if label not in ['car', 'motorcycle', 'bus', 'truck', 'person']:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections_coco.append({"label": label, "conf": conf, "bbox": (x1, y1, x2, y2)})
    final_boxes = []
    for det_c in detections_custom:
        label = det_c["label"].lower()
        conf = det_c["conf"]
        if label == "calculator" and conf >= 0.7:
            final_boxes.append({**det_c, "model": "custom"})
    for det_coco in detections_coco:
        skip = False
        for f in final_boxes:
            if f["label"].lower() == "calculator" and iou(det_coco["bbox"], f["bbox"]) > 0.5:
                skip = True
                break
        if skip: continue
        found_custom_match = False
        for det_c in detections_custom:
            if iou(det_c["bbox"], det_coco["bbox"]) > 0.5:
                found_custom_match = True
                if det_c["conf"] >= det_coco["conf"]:
                    if det_c not in final_boxes:
                        final_boxes.append({**det_c, "model": "custom"})
                else:
                    final_boxes.append({**det_coco, "model": "coco"})
                break
        if not found_custom_match:
            final_boxes.append({**det_coco, "model": "coco"})
    return final_boxes
# --- (Kết thúc các hàm xử lý ảnh và YOLO) ---


def process_video(input_path: Optional[str], output_path: Optional[str], display: bool) -> None:
    """
    ### SỬA ĐỔI LỚN ###
    Vòng lặp chính tích hợp cả hai pipeline.
    """
    generate_tts_files()
    if input_path is None:
        cap = cv2.VideoCapture(0) # Dùng webcam 0
        print("Using webcam 0")
    else:
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}")
            return
        cap = cv2.VideoCapture(input_path)
        print(f"Processing video: {input_path}")

    if not cap.isOpened():
        print("Không thể mở video/webcam.")
        return

    # Kích thước xử lý chuẩn
    PROC_WIDTH = 960
    PROC_HEIGHT = 540
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = None
    
    # ### MỚI ###: Danh sách ghi lại các cảnh báo
    recorded_alerts = [] # List of (alert_file, timestamp_in_seconds)
    frame_number = 0
    
    # ### MỚI ###: Đường dẫn file video tạm (không tiếng)
    temp_output_path = "temp_video_silent.mp4"

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # ### SỬA ĐỔI ###: Lưu vào file tạm
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (PROC_WIDTH, PROC_HEIGHT))
        print(f"Saving temporary (silent) video to: {temp_output_path}")

    if display:
        cv2.namedWindow('Lane and Object Detection', cv2.WINDOW_NORMAL)

    alert_cooldown_frames = 0

    while cap.isOpened():
        if alert_cooldown_frames > 0:
            alert_cooldown_frames -= 1
            
        ### MỚI ###
        frame_number += 1
        current_timestamp_sec = frame_number / fps

        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))
        except cv2.error as e:
            print(f"Lỗi resize khung hình: {e}")
            continue

        # 2. Xử lý làn đường
        processed_image, lane_fits = process_frame_lanes(frame.copy()) # Dùng frame.copy()
        left_fit, right_fit = lane_fits

        # 3. Xử lý phát hiện vật thể (YOLO)
        final_boxes = run_yolo_detection(frame) # Chạy YOLO trên frame gốc

        # 4. Phân tích, cảnh báo và vẽ
        alert_to_play = None 
        
        for det in final_boxes:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            width_px = x2 - x1
            
            distance_cm = None
            if label in OBJECT_WIDTHS_CM:
                distance_cm = calculate_distance(OBJECT_WIDTHS_CM[label], FOCAL_LENGTH, width_px)

            # --- LOGIC PHÂN LOẠI VỊ TRÍ ---
            bottom_center_x = (x1 + x2) / 2
            bottom_y = y2
            position = "Unknown"
            warning_text = None
            color = (0, 255, 0) # Xanh lá (mặc định)

            if left_fit is not None and right_fit is not None:
                try:
                    slope_l, intercept_l = left_fit
                    slope_r, intercept_r = right_fit
                    
                    lane_x_l = (bottom_y - intercept_l) / (slope_l + 1e-6)
                    lane_x_r = (bottom_y - intercept_r) / (slope_r + 1e-6)

                    if bottom_center_x < lane_x_l:
                        position = "Left"
                        if distance_cm and distance_cm < SIDE_THRESHOLD_CM:
                            warning_text = f"XE TRAI: {distance_cm/100:.1f}m"
                            color = (0, 165, 255) # Cam
                            if alert_to_play is None:
                                alert_to_play = AUDIO_LEFT_FILE
                    
                    elif bottom_center_x > lane_x_r:
                        position = "Right"
                        if distance_cm and distance_cm < SIDE_THRESHOLD_CM:
                            warning_text = f"XE PHAI: {distance_cm/100:.1f}m"
                            color = (0, 165, 255) # Cam
                            if alert_to_play is None:
                                alert_to_play = AUDIO_RIGHT_FILE
                    
                    else:
                        position = "Ahead"
                        if distance_cm and distance_cm < AHEAD_THRESHOLD_CM:
                            warning_text = f"XE TRUOC: {distance_cm/100:.1f}m"
                            color = (0, 0, 255) # Đỏ
                            alert_to_play = AUDIO_AHEAD_FILE # Ưu tiên cao nhất
                
                except Exception as e:
                    print(f"Lỗi tính toán vị trí: {e}")
            
            # --- Vẽ lên ảnh (processed_image là ảnh gốc đã resize) ---
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
            
            dist_text = f"{distance_cm/100:.1f}m" if distance_cm else "N/A"
            info_text = f"{label} | {dist_text} | {position}"
            
            cv2.putText(processed_image, info_text,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if warning_text:
                cv2.putText(processed_image, warning_text,
                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        ### HẾT VÒNG LẶP FOR ###

        # --- LOGIC PHÁT ÂM THANH VÀ CẢNH BÁO CHUNG (SAU VÒNG LẶP) ---
        if alert_to_play and alert_cooldown_frames == 0:
            
            # ### SỬA ĐỔI ###: Ghi lại cảnh báo để lưu file
            if output_path:
                recorded_alerts.append((alert_to_play, current_timestamp_sec))
            
            # Phát âm thanh nếu đang display
            if display:
                threading.Thread(target=lambda: playsound(alert_to_play), daemon=True).start()
            
            # Đặt lại cooldown
            alert_cooldown_frames = FRAMES_PER_ALERT_COOLDOWN

        if alert_to_play:
            cv2.putText(processed_image, "!!! CANH BAO !!!", (PROC_WIDTH // 2 - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 5. Hiển thị và lưu video (frame đã xử lý)
        if out:
            out.write(processed_image)
        if display:
            cv2.imshow('Lane and Object Detection', processed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Đổi về 1ms để xử lý nhanh
                break

    # --- KẾT THÚC VÒNG LẶP WHILE ---
    cap.release()
    if out:
        out.release()
    if display:
        cv2.destroyAllWindows()
    print("Xử lý khung hình video đã xong.")

    # 
    # ### MỚI: BƯỚC 2 - GỘP ÂM THANH VÀO VIDEO ###
    #
    if output_path:
        print(f"Đang bắt đầu gộp âm thanh vào file output: {output_path}...")
        if not os.path.exists(temp_output_path):
            print("Lỗi: Không tìm thấy file video tạm. Bỏ qua gộp âm thanh.")
            return

        try:
            # 1. Tải video câm đã tạo
            video_clip = VideoFileClip(temp_output_path)
            
            all_audio_clips = []
            
            # 2. Lấy âm thanh gốc từ video input (nếu có)
            if input_path:
                try:
                    original_audio = VideoFileClip(input_path).audio
                    if original_audio:
                        all_audio_clips.append(original_audio)
                        print("Đã thêm âm thanh gốc từ video input.")
                except Exception as e:
                    print(f"Không thể tải âm thanh gốc (video có thể không có tiếng): {e}")

            # 3. Thêm các clip âm thanh cảnh báo
            print(f"Đang thêm {len(recorded_alerts)} cảnh báo âm thanh...")
            for alert_file, timestamp in recorded_alerts:
                try:
                    alert_clip = AudioFileClip(alert_file)
                    # Đặt thời gian bắt đầu cho clip cảnh báo
                    alert_clip = alert_clip.set_start(timestamp)
                    all_audio_clips.append(alert_clip)
                except Exception as e:
                    print(f"Lỗi khi tải file {alert_file}: {e}")

            if not all_audio_clips:
                print("Không có âm thanh nào để gộp. Lưu video câm.")
                # Đổi tên file tạm thành file output
                os.rename(temp_output_path, output_path)
            else:
                # 4. Trộn tất cả âm thanh lại
                # CompositeAudioClip sẽ overlay các âm thanh lên nhau
                final_audio = CompositeAudioClip(all_audio_clips)
                
                # 5. Gán âm thanh vào video
                final_clip = video_clip.set_audio(final_audio)
                # Đảm bảo thời lượng audio khớp với video
                final_clip.duration = video_clip.duration
                
                # 6. Ghi file cuối cùng
                print("Đang ghi file video cuối cùng (có âm thanh)...")
                final_clip.write_videofile(
                    output_path, 
                    codec='libx264',        # codec video phổ biến
                    audio_codec='aac',      # codec âm thanh phổ biến
                    temp_audiofile='temp_audio.m4a', 
                    remove_temp=True
                )
                print(f"Đã lưu file thành công: {output_path}")

        except Exception as e:
            print(f"!!! Lỗi nghiêm trọng khi gộp âm thanh: {e}")
            print(f"Video không có tiếng đã được lưu tại: {temp_output_path}")
        
        finally:
            # 7. Dọn dẹp file video tạm
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
                print(f"Đã xóa file tạm: {temp_output_path}")

    else:
        print("Không có output_path, bỏ qua bước gộp âm thanh.")


def parse_args():
    parser = argparse.ArgumentParser(description='Lane and Object detection for video or webcam')
    parser.add_argument('--input', '-i', type=str, default=None, help='Input video path. If omitted, webcam 0 is used.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Optional output video path (mp4).')
    parser.add_argument('--display', '-d', action='store_true', help='Show video window during processing')
    return parser.parse_args()


def main():
    # args = parse_args()
    # process_video(args.input, args.output, args.display)
    
    # ### SỬA ĐỔI ###: Hardcode đường dẫn output để test
    process_video(
         input_path="data/demo_video.mp4", 
         #output_path = None,
         output_path="output_demo.mp4", # Đặt tên file output
         display=True
     )

if __name__ == '__main__':
    main()