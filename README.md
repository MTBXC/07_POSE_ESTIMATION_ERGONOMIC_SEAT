# 07_POSE_ESTIMATION_ERGONOMIC_SEAT# Ergonomic Seat Pose Estimation (YOLOv8-Pose / NVIDIA Jetson)

This project implements a real-time human pose estimation system optimized for NVIDIA Jetson devices, utilizing the YOLOv8n-pose model to monitor user posture and detect prolonged inactivity while seated.

## Project Architecture

The system provides two distinct inference backends for model execution:

1.  **ONNX Runtime + TensorRT (Optimized):** The primary production backend, utilizing **FP16** precision for maximum performance on Jetson hardware (benchmarked at **~46 FPS**). This is the default execution method.
2.  **PyTorch + Native CUDA (Baseline):** The original implementation used for performance benchmarking and comparison against the optimized solution.

## Business Logic Modules

The application is built around two core analytical functions, encapsulated in the `PostureAnalyzer` module:

### 1. Posture Monitoring (Slouching Detection)

* **Goal:** Calculate the average angle of the head-neck vector (ear-to-shoulder line) relative to the vertical axis.
* **Trigger:** Detects and warns the user when the calculated angle deviates from the vertical by more than a defined threshold (e.g., 25 degrees).

### 2. Inactivity Detection (Movement Reminder)

* **Goal:** Track micro-movements of the user's reference keypoint (e.g., ear position) to encourage mobility during long periods of sitting.
* **Trigger:** Alerts the user if the reference keypoint remains within a small pixel radius (e.g., 10px) for longer than a specified time (e.g., 5 seconds).

## Running the Application

### Prerequisites

* NVIDIA Jetson device (with JetPack and CUDA libraries).
* CSI Camera configured.
* Python 3.x and required dependencies (numpy, opencv-python, onnxruntime-gpu).

### Execution

To run the **optimized ONNX/TensorRT** version:

```bash
python3 pose_estimation_onnx.py