# raft_process.py
import os
import sys
import cv2
import numpy as np
import torch
import imageio
import math
import time
from argparse import Namespace
from collections import OrderedDict
from picamera2 import Picamera2

# Add RAFT module paths (assumes RAFT folder is in the same directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "RAFT", "core"))
sys.path.append(os.path.join(current_dir, "RAFT"))

from raft import RAFT
from utils.utils import InputPadder

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

def load_raft():
    args = Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args).eval().to(device)
    state_dict = torch.load("raft-things.pth", map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Pretrained weights loaded successfully!")
    return model

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    os.makedirs(output_folder, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:04d}.png"), frame)
        frame_count += 1
    cap.release()
    return frame_count

def frame_difference(frame1, frame2, roi):
    x, y, w, h = roi
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1[y:y+h, x:x+w], gray2[y:y+h, x:x+w])
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    frame_diff = np.zeros_like(gray1)
    frame_diff[y:y+h, x:x+w] = thresh
    return cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)

def compute_optical_flow(model, frame1, frame2, roi):
    x, y, w, h = roi
    frame1_roi = frame1[y:y+h, x:x+w]
    frame2_roi = frame2[y:y+h, x:x+w]
    frame1_roi = torch.from_numpy(frame1_roi).permute(2, 0, 1).unsqueeze(0).float().to(device)
    frame2_roi = torch.from_numpy(frame2_roi).permute(2, 0, 1).unsqueeze(0).float().to(device)
    padder = InputPadder(frame1_roi.shape)
    frame1_roi, frame2_roi = padder.pad(frame1_roi, frame2_roi)
    with torch.no_grad():
        flow = model(frame1_roi, frame2_roi)[-1].cpu().numpy()[0].transpose(1, 2, 0)
    flow = cv2.resize(flow, (w, h))
    full_flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
    full_flow[y:y+h, x:x+w, :] = flow
    return full_flow

def visualize_flow(flow, roi):
    x, y, w, h = roi
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[y:y+h, x:x+w, 0] = ang[y:y+h, x:x+w] * 180 / np.pi / 2
    hsv[y:y+h, x:x+w, 1] = 255
    hsv[y:y+h, x:x+w, 2] = cv2.normalize(mag[y:y+h, x:x+w], None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def calculate_drop_angle(white_pixels, reference_point):
    # 此函数不再使用，因为已取消 reference_point
    if not white_pixels:
        return None
    x_coords, y_coords = zip(*white_pixels)
    centroid_x = int(np.mean(x_coords))
    centroid_y = int(np.mean(y_coords))
    dx = centroid_x - reference_point[0]
    dy = centroid_y - reference_point[1]
    angle = math.degrees(math.atan2(dy, dx))
    if dx == 0:
        angle = 0
    return (centroid_x, centroid_y, angle)

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Unable to open video file. Please check the path or file format.")
        return {"error": "Unable to open video file"}
    else:
        print("✅ Video opened successfully.")
    cap.release()

    frame_folder = "frames"
    if os.path.exists(frame_folder):
        for f in os.listdir(frame_folder):
            os.remove(os.path.join(frame_folder, f))
    else:
        os.makedirs(frame_folder, exist_ok=True)
      
    frame_count = extract_frames(video_path, frame_folder)
    roi = (150, 250, 320, 160)

    output_frames = []
    drop_count = 0
    last_drop_angle = None
    cooldown_frames = 0
    WHITE_AREA_THRESHOLD = 6500

    model = load_raft()
    
    # 用于保存上一次检测到的 centroid
    prev_centroid = None

    for i in range(frame_count - 1):
        frame1 = cv2.imread(os.path.join(frame_folder, f"frame_{i:04d}.png"))
        frame2 = cv2.imread(os.path.join(frame_folder, f"frame_{i+1:04d}.png"))
        if frame1 is None or frame2 is None:
            print(f"❌ Failed to read frame_{i:04d}.png or frame_{i+1:04d}.png")
            continue

        frame_diff = frame_difference(frame1, frame2, roi)

        roi_mask = frame_diff[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        white_pixels = list(zip(np.where(roi_mask == 255)[1] + roi[0],
                                 np.where(roi_mask == 255)[0] + roi[1]))
        white_area = len(white_pixels)
        drop_angle = None
        centroid_x, centroid_y = None, None

        if white_area > WHITE_AREA_THRESHOLD and cooldown_frames == 0:
            # 根据 white_pixels 计算当前 centroid
            if white_pixels:
                x_coords, y_coords = zip(*white_pixels)
                current_centroid = (int(np.mean(x_coords)), int(np.mean(y_coords)))
            else:
                current_centroid = None

            if current_centroid is not None:
                if prev_centroid is None:
                    # 第一次检测：存储 centroid，并设置冷却 2 帧
                    prev_centroid = current_centroid
                    cooldown_frames = 1
                else:
                    # 计算上一次 centroid 与当前 centroid 之间的角度
                    dx = current_centroid[0] - prev_centroid[0]
                    dy = current_centroid[1] - prev_centroid[1]
                    drop_angle = math.degrees(math.atan2(dy, dx))
                    old_centroid = prev_centroid  # 保存上一次的 centroid 用于绘制
                    prev_centroid = current_centroid  # 更新为当前点
                    drop_count += 1
                    cooldown_frames = 15
                    last_drop_angle = drop_angle
                    print(f"✅ Successful drop #{drop_count} at frame {i+1}, angle: {drop_angle:.2f}°")
                centroid_x, centroid_y = current_centroid

        if cooldown_frames > 0:
            cooldown_frames -= 1

        cv2.rectangle(frame_diff, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
        # 绘制检测到的点、连线及角度标注
        if drop_angle is not None:
            # 绘制上一次检测到的点（黄色）
            cv2.circle(frame_diff, old_centroid, 5, (0, 255, 255), -1)
            # 绘制当前检测到的点（红色）
            cv2.circle(frame_diff, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            # 绘制两点之间的连线（蓝色）
            cv2.line(frame_diff, old_centroid, (centroid_x, centroid_y), (255, 0, 0), 2)
            # 在当前点旁标注角度数值
            cv2.putText(frame_diff, f"Angle: {drop_angle:.2f}", (centroid_x+10, centroid_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        drop_text = f"Success Drops: {drop_count}"
        if last_drop_angle is not None:
            drop_text += f" | Angle: {last_drop_angle:.2f}"
        cv2.putText(frame_diff, drop_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack((frame1, frame_diff))
        output_frames.append(combined)

    imageio.mimsave(output_path, output_frames, fps=30, codec="libx264")
    print(f"✅ Processing complete. Video saved to {output_path}")
    print(f"🔢 Total successful drops: {drop_count}")

    return {"output_video": output_path, "drop_count": drop_count, "angle": last_drop_angle}


def record_video(duration=10, output_file="captured.mp4"):
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(video_config)
    picam2.start()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))
    start_time = time.time()
    print("Recording video for", duration, "seconds...")
    while time.time() - start_time < duration:
        frame = picam2.capture_array()
        out.write(frame)
    picam2.stop()
    out.release()
    picam2.close()   # <-- 释放摄像头资源
    print("Video saved to:", output_file)
    return output_file

# 在 raft_process.py 中新增如下函数
def record_video_until(stop_event, output_file="captured.mp4", resolution=(640,480), fps=20):
    from picamera2 import Picamera2
    import cv2
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": resolution})
    picam2.configure(video_config)
    picam2.start()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, resolution)
    print("Recording video until stop event is set...")
    while not stop_event.is_set():
        frame = picam2.capture_array()
        if frame is not None:
            out.write(frame)
    picam2.stop()
    out.release()
    picam2.close()   # 释放摄像头资源
    print("Video recording stopped, saved to:", output_file)
    return output_file
