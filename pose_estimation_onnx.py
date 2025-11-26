import cv2
import time
import numpy as np
import onnxruntime as ort
from PostureAnalyzer import PostureAnalyzer

# --- ONNX Parameters ---
MODEL_PATH = 'yolov8n-pose.onnx' 
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.5
# --- ONNX Parameters ---

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

def preprocess(frame):
    """Prepares the frame for ONNX inference (resize, normalize, CHW)."""
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1) # HWC to CHW
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0) # Add Batch dimension (1, 3, 640, 640)


def postprocess(output, original_shape):
    """Processes raw ONNX output to return the best detection (keypoints and bbox)."""
    
    # Step 1: Transpose and filter by confidence
    output = output[0].T # Transpose to [8400, N_columns]
    
    # Find max confidence score (assuming single person, simple post-processing)
    max_conf_idx = np.argmax(output[:, 4], axis=0)
    max_conf = output[max_conf_idx, 4]
    
    if max_conf < CONF_THRESHOLD:
        return None, None
        
    best_detection = output[max_conf_idx]
    
    # Bounding box: [x, y, w, h] - normalized to 640
    bbox_normalized = best_detection[:4]
    
    # Keypoints (start from index 5)
    kpts_raw = best_detection[5:]
    
    if kpts_raw.size == 51:
        # Ideal size: 17 keypoints * 3 values (x, y, conf)
        kpts_normalized = kpts_raw.reshape(17, 3)
    else:
        # Handle the size 50 error encountered previously: add padding
        kpts_temp = np.zeros(51)
        kpts_temp[:kpts_raw.size] = kpts_raw
        kpts_normalized = kpts_temp.reshape(17, 3)

    # Step 3: Rescale to original frame size
    
    # Rescale Bounding Box
    ratio_x = original_shape[1] / INPUT_WIDTH
    ratio_y = original_shape[0] / INPUT_HEIGHT
    
    # Denormalize bbox (x, y, w, h)
    x, y, w, h = bbox_normalized
    x_center = x * ratio_x
    y_center = y * ratio_y
    width = w * ratio_x
    height = h * ratio_y
    
    # Convert to x1, y1, x2, y2 format
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    best_bbox = np.array([x1, y1, x2, y2]).astype(int)
    
    # Rescale Keypoints
    best_keypoints = np.zeros((17, 3))
    for i in range(17):
        # Scale X and Y (conf remains unchanged)
        best_keypoints[i, 0] = kpts_normalized[i, 0] * ratio_x
        best_keypoints[i, 1] = kpts_normalized[i, 1] * ratio_y
        best_keypoints[i, 2] = kpts_normalized[i, 2]
    
    return best_keypoints, best_bbox


def run_pose_estimation_onnx():
    print("Loading ONNX Runtime...")
    
    # --- TENSORRT OPTIMIZATION CONFIGURATION ---
    
    # TensorRT Provider Options (FP16 enabled for max performance)
    trt_options = {
        'trt_fp16_enable': True, 
        'trt_max_workspace_size': 1073741824, # 1 GB
        'trt_engine_cache_enable': True 
    }

    try:
        # Attempt to use TENSORRT Execution Provider first
        session = ort.InferenceSession(MODEL_PATH, 
                                         providers=[('TensorrtExecutionProvider', trt_options), 
                                                    'CUDAExecutionProvider'])
        print(f"ONNX Model: {MODEL_PATH} loaded successfully using TENSORRT.")
    except Exception as e:
        print(f"Error loading TENSORRT ({e}). Falling back to CUDA Execution Provider.")
        # Fallback to CUDA if TensorRT fails
        session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])
        print(f"ONNX Model: {MODEL_PATH} loaded successfully using CUDA (fallback).")


    input_name = session.get_inputs()[0].name
    
    pipeline = gstreamer_pipeline(display_width=1280, display_height=720, framerate=30, flip_method=2)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Cannot open GStreamer camera.")
        return

    print("Camera ready. Starting frame processing (ONNX)...")
    
    posture_analyzer = PostureAnalyzer(threshold_deg=25, inactivity_time_sec=5, position_threshold_px=10)
    inference_fps = 0.0 

    fps_history = []
    time.sleep(1.0) 
    print("Starting FPS logging...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot receive frame.")
            break

        original_shape = frame.shape
        inf_start_time = time.time()
        
        # Inference
        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        
        inf_end_time = time.time()
        inference_time = inf_end_time - inf_start_time
        
        if inference_time > 0:
            inference_fps = 1.0 / inference_time
            if inference_fps < 1000:
                fps_history.append(inference_fps)

        annotated_frame = frame.copy() 
        
        # Postprocessing: Get best detection
        best_keypoints, best_bbox = postprocess(outputs[0], original_shape)
                            
        # --- Analysis and Drawing ---

        if best_keypoints is not None:
            
            x1, y1, x2, y2 = best_bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            annotated_frame = draw_keypoints_and_skeleton(annotated_frame, best_keypoints, conf_threshold=CONF_THRESHOLD) 
            
            is_slouching, angle, is_inactivity_warning = posture_analyzer.check_slouching(best_keypoints)
            
            info_text = f"Avg. Posture Angle: {angle:.1f}°" 
            color = (0, 255, 0)
            warning_text = ""
            
            if is_slouching:
                warning_text = f"BAD POSTURE! (Deviation > {posture_analyzer.threshold_deg}°)" 
                color = (0, 165, 255)
            
            if is_inactivity_warning:
                warning_text = "Please move - back likes to be in the move! :)" 
                color = (0, 0, 255)
            
            cv2.putText(annotated_frame, info_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if warning_text:
                cv2.putText(annotated_frame, warning_text, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
            
            fps_text = f"Inference FPS (ONNX): {inference_fps:.2f}"
            cv2.putText(annotated_frame, fps_text, (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated_frame, f"No person detected (Conf < {CONF_THRESHOLD:.2f})", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Jetson Pose Estimation (ONNX)", annotated_frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- FPS SUMMARY ---
    cap.release()
    cv2.destroyAllWindows()
    
    if fps_history:
        avg_fps = np.mean(fps_history)
        min_fps = np.min(fps_history)
        max_fps = np.max(fps_history)
        
        print("\n--- ONNX PERFORMANCE SUMMARY (OPTIMIZED) ---")
        print(f"Number of samples: {len(fps_history)}")
        print(f"AVERAGE INFERENCE FPS (ONNX): {avg_fps:.2f}")
        print(f"Minimum FPS: {min_fps:.2f}")
        print(f"Maximum FPS: {max_fps:.2f}")
        print("---------------------------------------------")
    else:
        print("\nNo FPS samples recorded.")

if __name__ == '__main__':
    run_pose_estimation_onnx()