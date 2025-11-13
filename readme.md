# TINH_KHOANG_CACH — Mô tả toàn bộ thư mục

File này mô tả cấu trúc, các file quan trọng, và cách chạy các script trong thư mục gốc của dự án.

**Mục tiêu**: Hỗ trợ thu thập dữ liệu, huấn luyện model nhận dạng (ví dụ YOLO), đo khoảng cách dựa trên kích thước đối tượng, và một script phụ cho phát hiện vạch kẻ đường.

**Danh sách file & mô tả (gốc thư mục)**
- `adjust.ipynb`: Notebook Jupyter để tinh chỉnh và demo xử lý/hiển thị ảnh.
- `caotoc.py`: Script xử lý (tên file có thể là viết tắt chức năng cụ thể). Mở file để biết logic chi tiết.
- `collect_data.py`: Script hỗ trợ thu thập/copy dữ liệu ảnh, dùng khi chuẩn bị dataset.
- `data.yaml`: File cấu hình dataset (đường dẫn train/val, classes) — dùng khi huấn luyện với frameworks như YOLO.
- `lane_detection.py`: Script phát hiện vạch kẻ đường (OpenCV). Chạy trên webcam hoặc video.
- `measure_distance.py`, `measure_distance_IVcam.py`, `measure_distance_Oto.py`, `measure_distance_DT.py`: Các script đo khoảng cách/thuật toán khác nhau (tên gợi ý phương pháp hoặc loại camera). Mở từng file để xem input/output.
- `train.py`: Script huấn luyện model (xem chi tiết bên trong để biết framework/tham số).
- `readme.md`: (tệp này) mô tả tổng quan dự án.
- `requirements.txt`: Danh sách thư viện Python cần cài để chạy các script cơ bản (ví dụ `opencv-python`, `numpy`).
- `model/`: Thư mục chứa weights và checkpoint (ví dụ `yolov8n.pt`, `weights_custom/best.pt`).
- `data/`: Thư mục dataset, chứa `object_widths.csv`, `train/images`, `train/labels` (các file `.txt` nhãn theo định dạng YOLO hoặc định dạng tương tự).
- `runs/`: Kết quả huấn luyện (logs, checkpoints) — thường do framework training tạo ra.
- `venv310/`: Virtual environment có sẵn trong repo (Windows). Dùng nếu muốn giữ môi trường giống tác giả.
- `output_demo.mp4`, `SoDo1.png`, `tts_*.mp3`: File demo/hỗ trợ khác.

**Cấu trúc `data/` chi tiết**
- `data/object_widths.csv`: Bảng kích thước thực tế các đối tượng dùng để tính khoảng cách (pixel → khoảng cách thực tế).
- `data/train/images/`: Ảnh huấn luyện.
- `data/train/labels/`: File nhãn tương ứng (ví dụ YOLO `.txt`). Các file trong `labels/` như `calculator (1).txt`, `lipstick (1).txt`, ... là nhãn tương ứng từng ảnh.

**Cách cài & chạy nhanh (Windows PowerShell)**
1. Kích hoạt virtualenv (nếu dùng `venv310`):

```powershell
.\venv310\Scripts\Activate.ps1
```

2. Cài dependencies:

```powershell
pip install -r requirements.txt
```

3. Chạy phát hiện vạch (webcam):

```powershell
python lane_detection.py --display
```

4. Chạy trên file video và lưu kết quả:

```powershell
python lane_detection.py --input path\to\video.mp4 --output out.mp4 --display
```

5. Huấn luyện (ví dụ dùng `train.py` — lệnh thực tế tùy file):

```powershell
python train.py --config data.yaml --epochs 50
```

**Lưu ý quan trọng**
- Mỗi script có thể yêu cầu input/argument khác nhau — mở file `.py` để đọc docstring hoặc phần `argparse` để biết chi tiết.
- Nếu gặp lỗi import `cv2` hoặc thiếu gói, cài `opencv-python` từ `requirements.txt`.
- Thông tin dataset và label format: kiểm tra `data/train/labels` để biết format (như YOLO: `class x_center y_center width height` normalize).

**Gợi ý tinh chỉnh**
- Với `lane_detection.py`, tinh chỉnh ngưỡng Canny, ROI và tham số HoughLinesP để phù hợp với video thực tế.
- Với đo khoảng cách, đảm bảo `object_widths.csv` chứa chiều rộng thật của đối tượng tương ứng class để tính đúng khoảng cách.

**Người làm dự án**
- Lê Trung Hữu: 23666491
- Môn học Trí Tuệ Nhân tạo tại đại học Công Nghiệp thành phố Hồ Chính Minh

--------------------------------------------------------------------- CẢM ƠN MỌI NGƯỜI ĐÃ XEM!! ------------------------------------------------------------------------------
