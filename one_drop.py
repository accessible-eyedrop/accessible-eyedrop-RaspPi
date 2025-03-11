import time
from picamera2 import Picamera2
import numpy as np
import cv2
import RPi.GPIO as GPIO
import stepper_motor  # Import stepper motor control module

# ----------------------------
# 摄像头配置及参数设置
# ----------------------------
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(video_config)

custom_center = (230, 350)

# ----------------------------
# 全局变量：drop计数器、连续帧计数、冷却期，以及实时角度
# ----------------------------
drop_count = 0
consecutive_frames = 0
drop_counted = False
cooldown_until = 0
global_angle = 0.0
prev_drop_count = None
last_angle_upload_time = 0

# ----------------------------
# 新增：按钮和步进电机控制参数
# ----------------------------
BUTTON_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

active = False         # True when button is pressed and cycle is active
motor_reversed = False # Indicates if reverse movement has been executed
forward_steps = 0      # Total forward steps accumulated
extra_forward_steps = 0  # Steps taken after the drop is detected
MOTOR_STEP_SIZE = 10   # Steps to move forward per iteration
DROP_THRESHOLD = 1     # Drop threshold
EXTRA_FORWARD_STEPS = 100  # Extra steps to move forward after drop is detected

# ----------------------------
# 辅助函数（保持不变）
# ----------------------------
def has_sharp_bend(segment, threshold_angle=30):
    if len(segment) < 3:
        return False
    vector1 = segment[1] - segment[0]
    vector2 = segment[-1] - segment[1]
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * (180 / np.pi)
    return angle < threshold_angle

def smooth_contour(contour, window_size=5):
    smoothed = []
    for i in range(len(contour)):
        segment = contour[max(0, i - window_size//2) : min(len(contour), i + window_size//2 + 1)]
        avg_point = np.mean(segment, axis=0).astype(int)
        smoothed.append(avg_point)
    return np.array(smoothed)

def is_almost_straight(segment, relative_threshold=0.1, absolute_threshold=2.0):
    p0 = segment[0]
    p1 = segment[-1]
    chord_length = np.linalg.norm(p1 - p0)
    if chord_length < 1e-6:
        return True
    deviations = []
    for p in segment:
        deviation = np.abs((p1[0]-p0[0])*(p0[1]-p[1]) - (p1[1]-p0[1])*(p0[0]-p[0])) / chord_length
        deviations.append(deviation)
    max_dev = max(deviations)
    return (max_dev < chord_length * relative_threshold) and (max_dev < absolute_threshold)

def fit_circle(points):
    A = np.hstack((points, np.ones((points.shape[0], 1))))
    B = -(points[:, 0]**2 + points[:, 1]**2)
    sol, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    D, E, F = sol
    center = (-D/2, -E/2)
    radius = np.sqrt((D/2)**2 + (E/2)**2 - F)
    return center, radius

# ----------------------------
# 实时图像处理函数
# ----------------------------
def process_frame(frame):
    global global_angle
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = gray_frame.shape
    roi_width, roi_height = 300, 160
    roi_x = width // 2 - roi_width // 2
    roi_y = height // 2 - roi_height // 2 + 100
    roi = gray_frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    blurred_roi = cv2.GaussianBlur(roi, (3, 3), 1)
    edges = cv2.Canny(blurred_roi, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_segments = []
    best_segment = None
    smallest_radius = float('inf')
    segment_length = 50
    segment_step = 5

    for contour in contours:
        for i in range(0, len(contour) - segment_length + 1, segment_step):
            segment_points = contour[i:i+segment_length].reshape(-1, 2)
            if len(segment_points) < 3 or has_sharp_bend(segment_points):
                continue
            smoothed_segment = smooth_contour(segment_points, window_size=5)
            if is_almost_straight(smoothed_segment, relative_threshold=0.1, absolute_threshold=2.0):
                continue
            center_fit, radius_fit = fit_circle(smoothed_segment)
            if radius_fit < smallest_radius:
                smallest_radius = radius_fit
                best_segment = smoothed_segment
            valid_segments.append(smoothed_segment)

    detected = best_segment is not None
    output_frame = frame.copy()
    augmented_frame = frame.copy()
    colored_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    augmented_frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = colored_edges
    cv2.rectangle(augmented_frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 255), 2)

    if best_segment is not None:
        center_fit, radius_fit = fit_circle(best_segment)
        circle_center = (int(center_fit[0]), int(center_fit[1]))
        full_circle_center = (roi_x + circle_center[0], roi_y + circle_center[1])
        cv2.circle(augmented_frame, full_circle_center, int(radius_fit), (255, 0, 0), 2)
        best_segment_shifted = best_segment + np.array([roi_x, roi_y])
        cv2.polylines(augmented_frame, [best_segment_shifted.astype(np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.circle(augmented_frame, custom_center, 5, (0, 255, 0), -1)
        cv2.line(augmented_frame, custom_center, full_circle_center, (255, 255, 0), 2)
        cv2.line(augmented_frame, custom_center, (custom_center[0], height), (255, 255, 255), 2)
        vector1 = np.array(full_circle_center) - np.array(custom_center)
        vector2 = np.array((custom_center[0], height)) - np.array(custom_center)
        angle_rad = np.arccos(np.dot(vector1, vector2) / ((np.linalg.norm(vector1) * np.linalg.norm(vector2)) + 1e-6))
        angle_deg = np.degrees(angle_rad)
        global_angle = angle_deg
        angle_text = f'Angle: {angle_deg:.2f} degrees'
        cv2.putText(augmented_frame, angle_text, (custom_center[0] + 20, custom_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output_frame, augmented_frame, detected

# ----------------------------
# 主循环
# ----------------------------
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        current_time = time.time()
        
        # 按钮检测
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            if not active:
                active = True
                motor_reversed = False
                forward_steps = 0
                extra_forward_steps = 0
                drop_count = 0  # Reset drop count on new cycle
                print("Button pressed – starting motor and video detection.")

        if active:
            original_frame, augmented_frame, detected = process_frame(frame)
        else:
            original_frame = frame.copy()
            augmented_frame = frame.copy()
            detected = False

        # Drop计数逻辑（仅在active状态下执行）
        if active:
            if not detected:
                if cooldown_until == 0:
                    cooldown_until = current_time + 3  # 开启3秒冷却期
                consecutive_frames = 0
                drop_counted = False
            else:
                if current_time < cooldown_until:
                    pass
                else:
                    consecutive_frames += 1
                    if consecutive_frames >= 5 and not drop_counted:
                        drop_count += 1
                        drop_counted = True
                    if current_time >= cooldown_until:
                        cooldown_until = 0

            # ----------------------------
            # 步进电机控制逻辑：
            # - 在 drop_count 未达到阈值前，持续前进
            # - 一旦达到 drop_threshold，继续前进额外的EXTRA_FORWARD_STEPS
            # - 当额外前进步数达到后，反向移动与前进总步数相同的步数
            # ----------------------------
            if drop_count < DROP_THRESHOLD:
                # 正常前进
                stepper_motor.move_forward(MOTOR_STEP_SIZE)
                forward_steps += MOTOR_STEP_SIZE
                print(f"Moving forward. Total forward steps: {forward_steps}")
            else:
                # 当drop达到阈值后，继续前进额外步数
                if extra_forward_steps < EXTRA_FORWARD_STEPS:
                    stepper_motor.move_forward(MOTOR_STEP_SIZE)
                    forward_steps += MOTOR_STEP_SIZE
                    extra_forward_steps += MOTOR_STEP_SIZE
                    print(f"Extra forward movement. Extra steps: {extra_forward_steps} | Total: {forward_steps}")
                else:
                    if not motor_reversed:
                        print(f"Extra steps reached. Reversing {forward_steps} steps.")
                        stepper_motor.move_backward(forward_steps)
                        motor_reversed = True
                        active = False
                        # 重置状态以便下一次触发
                        forward_steps = 0
                        extra_forward_steps = 0
                        drop_count = 0

        display_text = f"Drops: {drop_count}"
        cv2.putText(augmented_frame, display_text, (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Original Camera View", original_frame)
        cv2.imshow("Augmented Preview", augmented_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()

