import cv2
import numpy as np
import time
import threading
from picamera2 import Picamera2
import stepper_motor  # 同目录下的步进电机模块

# 全局变量，用于存储液滴检测结果
drop_detected = False
drop_lock = threading.Lock()

def image_processing_thread():
    global drop_detected
    # 初始化摄像头
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(video_config)
    picam2.start()

    # 获取第一帧确定尺寸，并定义ROI参数（根据测试成功的参数）
    frame = picam2.capture_array()
    if frame is None:
        print("无法捕获图像。")
        return
    h, w = frame.shape[:2]
    x_offset = w // 5
    y_offset = h * 4 // 5
    roi_w = w // 2
    roi_h = h // 6

    # 黑色区域面积阈值（可根据实际情况调整）
    AREA_THRESHOLD = 600

    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue

        # 提取整个ROI区域并转换为灰度图
        roi = frame[y_offset:y_offset + roi_h, x_offset:x_offset + roi_w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 反阈值处理：利用Otsu自动阈值，并适当降低阈值（例如减20）以捕获较暗区域
        ret, _ = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adjusted_threshold = max(ret - 20, 0)
        _, roi_mask = cv2.threshold(gray_roi, adjusted_threshold, 255, cv2.THRESH_BINARY_INV)

        # 形态学操作：先开运算去除噪声，再闭运算填补空洞
        kernel = np.ones((3, 3), np.uint8)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 查找ROI中的轮廓，并判断是否存在面积超过阈值的区域
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        for cnt in contours:
            if cv2.contourArea(cnt) > AREA_THRESHOLD:
                detected = True
                break

        # 更新全局检测结果（加锁保证线程安全）
        with drop_lock:
            drop_detected = detected

        # 可视化处理后的结果（整个ROI区域）
        processed_view = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Processed view", processed_view)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出图像处理线程
            break

    picam2.stop()
    cv2.destroyAllWindows()

# 启动图像处理线程（设置为守护线程，主线程退出时自动结束）
img_thread = threading.Thread(target=image_processing_thread, daemon=True)
img_thread.start()

# 主线程负责步进电机控制
cumulative_forward_steps = 0
FORWARD_STEP_SIZE = 150  # 每次未检测到液滴时前进的步数

try:
    while True:
        # 获取最新检测结果
        with drop_lock:
            detected = drop_detected

        if detected:
            # 检测到液滴：先额外前进100步，再将所有累计步数回退
            print("Liquid drop detected! Moving forward 30 steps then moving backward.")
            stepper_motor.move_forward(30)
            cumulative_forward_steps += 30
            stepper_motor.move_backward(cumulative_forward_steps)
            print(f"Moved backward {cumulative_forward_steps} steps.")
            cumulative_forward_steps = 0
        else:
            # 未检测到液滴，继续前进
            stepper_motor.move_forward(FORWARD_STEP_SIZE)
            cumulative_forward_steps += FORWARD_STEP_SIZE
            print(f"Moving forward. Cumulative steps: {cumulative_forward_steps}")

        # 根据需要调整延时，避免主线程过于频繁地调用电机控制函数
        time.sleep(0.01)

except KeyboardInterrupt:
    print("程序中断。")

stepper_motor.cleanup()
