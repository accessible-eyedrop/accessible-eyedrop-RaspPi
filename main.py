# main.py
import time
import threading
import RPi.GPIO as GPIO
from stepper_motor import move_forward, move_backward, stop_motor, cleanup as motor_cleanup
from raft_process import record_video, process_video
# from firebase_utils import update_firestore

# Configure button on GPIO pin 4 (using pull-up)
BUTTON_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# For a motor with 2038 steps per revolution, one revolution moves 2mm.
STEPS_PER_REV = 2038//2

def main():
    print("Waiting for button press...")
    try:
        while True:
            # Button pressed (LOW because of internal pull-up)
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("Button pressed!")
                # Start video recording concurrently
                record_thread = threading.Thread(target=record_video, args=(10, "captured.mp4"))
                record_thread.start()
                
                # Move motor 2mm forward (1 full revolution)
                print("Moving motor forward 2mm...")
                move_forward(STEPS_PER_REV)
                stop_motor()  # ensure coils are off
                
                # Wait 3 seconds
                time.sleep(3)
                
                # Move motor 2mm backward
                print("Moving motor backward 2mm...")
                move_backward(STEPS_PER_REV)
                stop_motor()
                
                # Wait for recording to finish
                record_thread.join()
                
                # Process the recorded video using RAFT
                process_video("captured.mp4", "outputcv.mp4")
                
                # Dummy processed data for Firebase update (replace with actual results if available)
                processed_data = {"angle": 0, "drop_count": 1}
#                 update_firestore(processed_data["angle"], processed_data["drop_count"], processed_data["drop_count"] > 0)
                
                # Debounce: wait until the button is released
                while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    time.sleep(0.1)
                print("Waiting for next button press...")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        motor_cleanup()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
