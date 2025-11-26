import cv2
import time
import numpy as np
from ultralytics import YOLO
from PostureAnalyzer import PostureAnalyzer

# Keypoint connections (COCO standard)
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), 
    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]
KEYPOINT_COLOR = (0, 255, 255)
SKELETON_COLOR = (255, 0, 0)

def draw_keypoints_and_skeleton(img, keypoints, conf_threshold=0.5):
    """Draws keypoints and skeleton lines on the frame."""
    kpts_xy = keypoints[:, :2].astype(int)
    kpts_conf = keypoints[:, 2]

    # Draw skeleton lines
    for i, j in SKELETON_EDGES:
        if kpts_conf[i] > conf_threshold and kpts_conf[j] > conf_threshold:
            pt1 = tuple(kpts_xy[i])
            pt2 = tuple(kpts_xy[j])
            cv2.line(img, pt1, pt2, SKELETON_COLOR, 2)

    # Draw keypoints
    for k in range(kpts_xy.shape[0]):
        if kpts_conf[k] > conf_threshold:
            center = tuple(kpts_xy[k])
            cv2.circle(img, center, 4, KEYPOINT_COLOR, -1)
            
    return img


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=2,
):
    """Generates the GStreamer string for the CSI camera on Jetson."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! appsink"
    )

def run_pose_estimation():
    print("Loading YOLOv8-Pose model to CUDA...")
    # Load YOLOv8 Nano Pose model and move it to GPU (cuda:0)
    model = YOLO('yolov8n-pose.pt').to('cuda:0') 

    pipeline = gstreamer_pipeline(display_width=1280, display_height=720, framerate=30, flip_method=2)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Cannot open GStreamer camera.")
        return

    print("Camera ready. Starting frame processing (PyTorch CUDA)...")
    
    posture_analyzer = PostureAnalyzer(threshold_deg=25, inactivity_time_sec=5, position_threshold_px=100)
    inference_fps = 0.0 

    # FPS calculation variables
    fps_history = []
    
    # Wait for camera stabilization before measuring
    time.sleep(1.0) 
    
    print("Starting FPS logging...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot receive frame.")
            break

        inf_start_time = time.time()
        
        # 1. Inference
        results = model(frame, verbose=False, conf=0.60, save_conf=True) 
        
        inf_end_time = time.time()
        inference_time = inf_end_time - inf_start_time
        
        if inference_time > 0:
            inference_fps = 1.0 / inference_time
            # Log FPS only for meaningful values
            if inference_fps < 1000:
                fps_history.append(inference_fps)

        annotated_frame = frame.copy() 
        
        # 2. Detection Filtering (Find best/highest confidence person)
        best_keypoints = None
        best_bbox = None
        max_confidence = 0.0
        
        if results and results[0].boxes:
            for r in results:
                for i in range(len(r.boxes)):
                    box = r.boxes[i]
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Check for person (cls=0) and max confidence
                    if cls == 0 and conf > max_confidence:
                        max_confidence = conf
                        if r.keypoints.data.shape[0] > i:
                            best_keypoints = r.keypoints.data[i].cpu().numpy()
                            best_bbox = box.xyxy[0].cpu().numpy().astype(int)
                            
        # 3. Analysis and Drawing
        if best_keypoints is not None:
            
            # Drawing BBox and Skeleton
            x1, y1, x2, y2 = best_bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            annotated_frame = draw_keypoints_and_skeleton(annotated_frame, best_keypoints, conf_threshold=0.5) 
            
            # Posture Analysis
            is_slouching, angle, is_inactivity_warning = posture_analyzer.check_slouching(best_keypoints)
            
            # Text Setup
            info_text = f"Avg. Posture Angle: {angle:.1f} degrees" 
            color = (0, 255, 0)
            warning_text = ""
            
            if is_slouching:
                warning_text = f"BAD POSTURE! (Deviation > {posture_analyzer.threshold_deg} degrees)" 
                color = (0, 165, 255) # Orange
            
            if is_inactivity_warning:
                warning_text = "Please move - back likes to be in the move! :)" 
                color = (0, 0, 255) # Red
            
            # Display Text
            cv2.putText(annotated_frame, info_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if warning_text:
                cv2.putText(annotated_frame, warning_text, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
            
            # Display Inference FPS
            fps_text = f"Inference FPS (CUDA): {inference_fps:.2f}"
            cv2.putText(annotated_frame, fps_text, (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated_frame, f"No person detected (Conf < 0.60)", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Jetson Pose Estimation", annotated_frame) 

        # Handle 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- FPS SUMMARY ---
    cap.release()
    cv2.destroyAllWindows()
    
    if fps_history:
        avg_fps = np.mean(fps_history)
        min_fps = np.min(fps_history)
        max_fps = np.max(fps_history)
        
        print("\n--- CUDA PERFORMANCE SUMMARY (BASELINE) ---")
        print(f"Number of samples: {len(fps_history)}")
        print(f"AVERAGE INFERENCE FPS (CUDA): {avg_fps:.2f}")
        print(f"Minimum FPS: {min_fps:.2f}")
        print(f"Maximum FPS: {max_fps:.2f}")
        print("---------------------------------------------")
    else:
        print("\nNo FPS samples recorded.")

if __name__ == '__main__':
    run_pose_estimation()