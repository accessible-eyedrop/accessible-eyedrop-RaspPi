# Final\_demo

This repository demonstrates a complete project that integrates:

1. **Firebase Data Upload** – Upload processed data to Firebase using the Firebase Admin SDK.
2. **RAFT-based Optical Flow Video Processing** – Process recorded video using the RAFT model for optical flow analysis.
3. **Stepper Motor Control** – Drive a 28BYJ-48 stepper motor with a ULN2003 driver.
4. **Main Application** – Wait for a button press to simultaneously record video and drive the stepper motor, then process the video and upload results to Firebase.

---

## Repository Structure

```
Final_demo/
├── firebase_utils.py         # Firebase initialization and update functions
├── raft_process.py           # RAFT model loading, video processing, and video recording functions using Picamera2
├── stepper_motor.py          # Stepper motor control functions (move forward, move backward, stop, and cleanup)
├── main.py                   # Main entry point: waits for a button press, drives motor, records video, processes video, uploads data
├── RAFT/                     # The RAFT repository folder (clone from https://github.com/princeton-vl/RAFT.git)
├── raft-things.pth           # Pretrained RAFT weights (download from https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth)
└── serviceAccountKey.json    # Firebase service account key file
```

---

## Dependencies

### System Dependencies (Raspberry Pi 4B)

- **Operating System:** Raspberry Pi OS Bullseye (or later)
- **Camera:** Raspberry Pi Camera v2 connected to the CSI port
  - **Enable:** Run `sudo raspi-config` → Interfacing Options → Camera.
- **Button:** Connected to a GPIO pin (see Hardware Connections below)
- **Stepper Motor & Driver:** 28BYJ-48 stepper motor with ULN2003 driver board
- **FFmpeg:** Required for video encoding

Install system packages via apt:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv ffmpeg python3-rpi.gpio
```

### Python Dependencies

It is **highly recommended** to use a virtual environment.

1. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install required Python packages:**

   ```bash
   pip install torch torchvision numpy imageio matplotlib tqdm tensorboardX picamera2
   pip install "imageio[ffmpeg]"
   ```

**Note:**

- Download the pretrained RAFT weights from [raft-things.pth](https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth) and place it in the repository root.
- Place your Firebase service account key file as `serviceAccountKey.json` in the repository root.

---

## Hardware Connections

### Raspberry Pi Camera v2

- Connect the Camera v2 to the CSI port on the Raspberry Pi 4B.
- Enable the camera via `sudo raspi-config` under Interfacing Options.

### Button

- Connect one terminal of the button to **GPIO pin 4**.
- Connect the other terminal to ground.
- The code uses an internal pull-up resistor on GPIO 4.

### Stepper Motor (28BYJ-48) and ULN2003 Driver

- **Connections:**
  - ULN2003 IN1 → GPIO 17
  - ULN2003 IN2 → GPIO 18
  - ULN2003 IN3 → GPIO 27
  - ULN2003 IN4 → GPIO 22
- Connect the ULN2003 board’s VCC to the 5V supply and GND to the Raspberry Pi ground.
- **Note:** For this project, the motor’s specification is 2038 steps per revolution, and one full revolution moves the mechanism 2 mm. Thus, moving 2 mm forward requires issuing 2038 steps.

---

## How to Run the Code

### 1. Clone the RAFT Repository

From the project root, run:

```bash
git clone https://github.com/princeton-vl/RAFT.git
```

Ensure the `RAFT` folder is in the same directory as `main.py`.

### 2. Place Required Files

- Place `raft-things.pth` (downloaded from the RAFT releases) in the repository root.
- Place your Firebase service account key as `serviceAccountKey.json` in the repository root.

### 3. Activate Your Virtual Environment

```bash
source venv/bin/activate
```

### 4. Run the Main Program

```bash
python3 main.py
```

---

## What the Program Does

1. **Waiting for a Button Press:**\
   The program continuously waits for the button (on GPIO 4) to be pressed.

2. **On Button Press:**

   - A separate thread starts recording video for 5 seconds using Picamera2. This records the process, including the motor’s movement.
   - Simultaneously, the stepper motor is driven forward for one full revolution (2038 steps, moving 2 mm forward), and then the coils are de-energized.
   - The system waits for 3 seconds.
   - Then, the motor is driven backward for 2038 steps (2 mm backward) and stopped.

3. **Video Processing and Data Upload:**

   - After recording, the video is processed using the RAFT optical flow algorithm (in `raft_process.py`), and an annotated output video is saved.
   - Processed data (e.g., drop count and angle) is uploaded to Firebase via the functions in `firebase_utils.py`.

---

## Troubleshooting

- **Camera Errors:**\
  If you encounter camera initialization errors (e.g., "Camera **init** sequence did not complete"), ensure no other application is using the camera and that you properly close the camera (the code calls `picam2.close()` after recording).

- **Stepper Motor Issues:**\
  If the motor does not stop, ensure that `stop_motor()` is called after each movement and verify your wiring and GPIO assignments.

- **Firebase Issues:**\
  Ensure that your `serviceAccountKey.json` is correct and that your Firebase project is properly configured.

- **General:**\
  Double-check that all files are in the correct directories and that all dependencies are installed.

---


