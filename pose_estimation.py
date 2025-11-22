import cv2
from ultralytics import YOLO
import time
from PostureAnalyzer import PostureAnalyzer

# --- Configuration Camera (CSI) ---
def gstreamer_pipeline(capture_width=1920, capture_height=1080, display_width=1280, display_height=720, framerate=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method=2 ! " 
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink"
    )

# --- Main Function ---
def run_pose_estimation():
    print("Loading YOLOv8-Pose...")
    # Smallest model 'n' (nano)
    model = YOLO('yolov8n-pose.pt').to('cuda:0') 

    # Initialization of camera
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    print("Kamera gotowa. Rozpoczęcie przetwarzania klatek...")
    
    # Variables for calculate FPS
    start_time = time.time()
    frame_count = 0
    interference_fps = 0.0

    posture_analyzer = PostureAnalyzer(threshold_deg=25)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error frame load")
            break

        # Start calcualte inference FPS
        inf_start_time = time.time()

        # Start interference
        results = model(frame, verbose=False, conf=0.5)

        #Stop Time calculate inteference FPS

        inf_end_time = time.time()
        inference_time = inf_end_time - inf_start_time

        # Prevent dividing by zero

        if inference_time > 0:
            inference_fps = 1.0 / inference_time
        
        # Plot key points and bounding box
        annotated_frame = results[0].plot()

        results = model(frame, verbose=False, conf=0.5) 
        annotated_frame = results[0].plot()

        # --- Analiza Postawy z Użyciem Klasy ---
        if results and results[0].keypoints.shape[0] > 0:
            keypoints = results[0].keypoints.data.cpu().numpy()[0] 
            
            is_slouching, angle = posture_analyzer.check_slouching(keypoints)
            
            # --- Sygnalizacja ---
            # Zmieniamy tekst na angielski
            info_text = f"Ear-Shoulder Angle: {angle:.1f}°" 
            color = (0, 255, 0) # Green (BGR)
            warning_text = ""
            
            if is_slouching:
                warning_text = "BAD POSTURE" # Zmieniamy tekst ostrzeżenia
                color = (0, 165, 255) # Orange (BGR)
                
            cv2.putText(annotated_frame, info_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if warning_text:
                cv2.putText(annotated_frame, warning_text, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            print(f"FPS: {fps:.2f}")
            start_time = end_time
            frame_count = 0

        # Displaying inference FPS
        
        fps_text = f"Inference FPS: {inference_fps:.2f}"
        cv2.putText(annotated_frame, fps_text, (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Wyświetlenie wyniku
        cv2.imshow("Jetson Pose Estimation", annotated_frame)

        # Exit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pose_estimation()
