from ultralytics import YOLO
import os
import shutil

# CẤU HÌNH
DATA_YAML = "data.yaml"
PRETRAINED_MODEL = "model/yolov8n.pt" 
EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 16
OUTPUT_DIR = "model/weights_custom"     

# KHỞI TẠO & HUẤN LUYỆN
print("Khởi tạo mô hình từ pretrained:", PRETRAINED_MODEL)
model = YOLO(PRETRAINED_MODEL)

print("Bắt đầu huấn luyện với dữ liệu:", DATA_YAML)
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    name="custom_training",
    project="runs/train",
    lr0=0.001,
    patience=15,
    augment=True,
    verbose=True,
)

# SAO CHÉP FILE SAU TRAIN
trained_path = "runs/train/custom_training/weights/best.pt"
output_path = os.path.join(OUTPUT_DIR, "best.pt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(trained_path):
    shutil.copy(trained_path, output_path)
    print(f"\nMô hình đã được lưu tại: {output_path}")
else:
    print("\nKhông tìm thấy file best.pt, vui lòng kiểm tra lại thư mục runs/train/custom_training/weights/")

# DỌN DẸP FILE TẠM
#shutil.rmtree("runs/train/custom_training", ignore_errors=True)
#print("Đã dọn thư mục tạm.")
