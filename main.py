import cv2
import numpy as np
import time
import threading
from queue import Queue
import RPi.GPIO as GPIO

# Import stepper motor control, RAFT video processing, and buzzer sound functions
from stepper_motor import move_forward, move_backward, cleanup as motor_cleanup
from raft_process import process_video  # Using process_video from RAFT processing module
from buzzer import play_melody

#############################################
# Global variables and shared data
#############################################
frame_queue = Queue(maxsize=10)  # Queue to store camera frames
camera_stop_event = threading.Event()  # To stop the entire camera capture system

# Shared drop detection result between threads
shared_drop = {"detected": False}

# Recording stop event will be created per recording session (used only by the video record thread)
record_stop_event = None

#############################################
# Camera Capture Thread (Single global camera instance)
#############################################
def camera_capture_thread():
    from picamera2 import Picamera2
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(video_config)
    picam2.start()
    try:
        while not camera_stop_event.is_set():
            frame = picam2.capture_array()
            if frame is not None:
                if frame_queue.full():
                    frame_queue.get()  # Discard the oldest frame
                frame_queue.put(frame)
            time.sleep(0.01)
    except Exception as e:
        print("Camera capture error:", e)
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

#############################################
# Image Processing Thread: Display original and processed ROI,
# and update drop detection result
#############################################
def image_processing_thread():
    # ROI parameters (fixed resolution: 640x480)
    x_offset = 640 // 5
    y_offset = (480 * 4) // 5
    roi_w = 640 // 2
    roi_h = 480 // 6
    AREA_THRESHOLD = 600

    while not camera_stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Show original frame
            cv2.imshow("Original view", frame)

            # Extract ROI area
            roi = frame[y_offset:y_offset+roi_h, x_offset:x_offset+roi_w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, _ = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            adjusted_threshold = max(ret - 20, 0)
            _, roi_mask = cv2.threshold(gray_roi, adjusted_threshold, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            # Detect drop (if any contour area exceeds threshold)
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected = any(cv2.contourArea(cnt) > AREA_THRESHOLD for cnt in contours)
            shared_drop["detected"] = detected

            processed_view = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Processed view", processed_view)
            if cv2.waitKey(1) & 0xFF == 27:
                camera_stop_event.set()
        else:
            time.sleep(0.005)

#############################################
# Video Recording Thread: Write frames from the shared queue to a video file
#############################################
def video_record_thread(output_file="captured.mp4", resolution=(640,480), fps=20, rec_stop_event=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, resolution)
    while not rec_stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            out.write(frame)
        else:
            time.sleep(0.005)
    out.release()
    print("Video recording stopped, saved to:", output_file)

#############################################
# Main process: Button, stepper motor control, and video processing
#############################################
# Configure external button (GPIO 4, with internal pull-up)
BUTTON_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Stepper motor steps parameters
FORWARD_STEP_SIZE = 100  # Steps to move forward when no drop detected
EXTRA_FORWARD = 0       # Extra forward steps when drop detected

def main():
    global record_stop_event
    print("Waiting for button press...")

    # Start camera capture and image processing threads
    cam_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    cam_thread.start()
    proc_thread = threading.Thread(target=image_processing_thread, daemon=True)
    proc_thread.start()

    try:
        while True:
            # Wait for button press (active low)
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("Button pressed! Starting motor control and video recording.")
                # Create a stop event for this recording session
                record_stop_event = threading.Event()
                rec_thread = threading.Thread(target=video_record_thread, kwargs={
                    "output_file": "captured.mp4",
                    "resolution": (640,480),
                    "fps": 20,
                    "rec_stop_event": record_stop_event
                })
                rec_thread.start()

                cumulative_steps = 0
                # Motor control loop
                while True:
                    detected = shared_drop.get("detected", False)
                    if detected:
                        print("Drop detected! Moving extra forward and then backward.")
                        move_forward(EXTRA_FORWARD)
                        cumulative_steps += EXTRA_FORWARD
                        move_backward(cumulative_steps)
                        print(f"Moved backward {cumulative_steps} steps.")
                        cumulative_steps = 0
                        break  # End current motion process
                    else:
                        move_forward(FORWARD_STEP_SIZE)
                        cumulative_steps += FORWARD_STEP_SIZE
                        print(f"Continuing forward, cumulative steps: {cumulative_steps}")
                    time.sleep(0.01)

                # Motion process finished, stop video recording
                record_stop_event.set()
                rec_thread.join()
                print("Video recording ended.")

                # Process the recorded video using RAFT functions
                processed_data = process_video("captured.mp4", "outputcv.mp4")
                print("Video processing result:", processed_data)

                # Sound feedback (e.g., if drop_count > 0 then success, else fail)
                if processed_data.get("drop_count", 0) > 0:
                    play_melody("success")
                else:
                    play_melody("fail")

                # Wait until button is released to avoid repeated triggers
                while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    time.sleep(0.1)
                print("Waiting for next button press...")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        camera_stop_event.set()
        motor_cleanup()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
