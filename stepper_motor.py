import RPi.GPIO as GPIO
import time

# Define GPIO pins for ULN2003 inputs
IN1 = 17
IN2 = 18
IN3 = 27
IN4 = 22

# Set up GPIO mode and configure pins as outputs
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# Full-step sequence (4-step) for 28BYJ-48
full_step_sequence = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

# New default delay: 1.5x faster than 0.002 sec (~0.00133 sec)
DEFAULT_DELAY = 0.002

def step_motor(steps, delay=DEFAULT_DELAY):
    seq_len = len(full_step_sequence)
    for i in range(steps):
        current_step = full_step_sequence[i % seq_len]
        GPIO.output(IN1, current_step[0])
        GPIO.output(IN2, current_step[1])
        GPIO.output(IN3, current_step[2])
        GPIO.output(IN4, current_step[3])
        time.sleep(delay)
    # De-energize coils to reduce heating after movement
    stop_motor()

def move_forward(steps, delay=DEFAULT_DELAY):
    step_motor(steps, delay)

def move_backward(steps, delay=DEFAULT_DELAY):
    seq_len = len(full_step_sequence)
    for i in range(steps):
        current_step = full_step_sequence[(-i) % seq_len]
        GPIO.output(IN1, current_step[0])
        GPIO.output(IN2, current_step[1])
        GPIO.output(IN3, current_step[2])
        GPIO.output(IN4, current_step[3])
        time.sleep(delay)
    stop_motor()

def stop_motor():
    # De-energize all coils
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 0)

def cleanup():
    GPIO.cleanup()

# Connection Instructions:
# - Connect ULN2003 board’s IN1, IN2, IN3, and IN4 to Raspberry Pi GPIO 17, 18, 27, and 22 respectively.
# - Connect VCC to 5V and GND to Raspberry Pi ground.
#
# To move 2 mm forward (one revolution for your motor):
#     move_forward(2038)
# Then call stop_motor() to turn off coils.
