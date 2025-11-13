#!/usr/bin/env python3
"""
lane_detection.py

Phát hiện vạch kẻ đường trong video hoặc webcam.

Usage:
    python lane_detection.py --input path/to/video.mp4 --output out.mp4 --display
    python lane_detection.py --display  # dùng webcam

Algorithm (simple pipeline):
 - chuyển ảnh sang grayscale
 - Gaussian blur
 - Canny edge detection
 - mask vùng quan tâm (ROI)
 - HoughLinesP để tìm các đoạn thẳng
 - tách thành left/right theo độ dốc, trung bình hóa và nội suy thành 2 đường dài
 - vẽ lên khung hình

Thiết kế đơn giản, dễ tuỳ chỉnh cho các dự án thực tế.
"""
import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np


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
    if lines is None:
        return line_image
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image


def make_coordinates(img_shape: Tuple[int, int], line_parameters: Tuple[float, float]) -> List[int]:
    slope, intercept = line_parameters
    height = img_shape[0]
    # vẽ từ y = height (đáy ảnh) tới y = int(height * 0.6)
    y1 = height
    y2 = int(height * 0.6)
    # x = (y - b) / m
    if slope == 0:
        slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]


def average_slope_intercept(img: np.ndarray, lines: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if lines is None:
        return None
    left_fit = []
    right_fit = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            # loại bỏ đoạn gần ngang
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    img_shape = img.shape
    lines_to_draw = []
    if left_fit:
        left_fit_avg = np.mean(left_fit, axis=0)
        lines_to_draw.append(make_coordinates(img_shape, left_fit_avg))
    if right_fit:
        right_fit_avg = np.mean(right_fit, axis=0)
        lines_to_draw.append(make_coordinates(img_shape, right_fit_avg))
    if not lines_to_draw:
        return None
    return np.array(lines_to_draw)


def process_frame(frame: np.ndarray) -> np.ndarray:
    # Các tham số có thể điều chỉnh theo video
    gray = grayscale(frame)
    blur = gaussian_blur(gray, 5)
    edges = canny(blur, 50, 150)

    height, width = frame.shape[0], frame.shape[1]
    # Define a triangular region of interest roughly covering the road ahead
    polygons = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height),
    ]], dtype=np.int32)

    masked = region_of_interest(edges, polygons)

    # Hough transform parameters - tuỳ chỉnh nếu cần
    lines = cv2.HoughLinesP(masked, rho=1, theta=np.pi / 180, threshold=50, minLineLength=40, maxLineGap=100)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combo


def process_video(input_path: Optional[str], output_path: Optional[str], display: bool) -> None:
    if input_path is None:
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}")
            return
        cap = cv2.VideoCapture(input_path)

    fourcc = None
    out = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    if output_path:
        # Xác định codec dựa trên phần mở rộng
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if display:
        cv2.namedWindow('lane detection', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        if out:
            out.write(processed)
        if display:
            cv2.imshow('lane detection', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    if display:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='Lane detection for video or webcam')
    parser.add_argument('--input', '-i', type=str, default=None, help='Input video path. If omitted, webcam is used.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Optional output video path (mp4).')
    parser.add_argument('--display', '-d', action='store_true', help='Show video window during processing')
    return parser.parse_args()


def main():
    args = parse_args()
    process_video(args.input, args.output, args.display)


if __name__ == '__main__':
    input_path = "data/demo_video.mp4"  
    #output_path = "output.mp4"  
    display = True          

    process_video(input_path, None, display)

