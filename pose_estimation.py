# --- Plik: pose_estimation.py ---

import cv2
import time
import numpy as np
from ultralytics import YOLO
from PostureAnalyzer import PostureAnalyzer

# Definicja połączeń dla szkieletu (standard COCO)
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), 
    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]
KEYPOINT_COLOR = (0, 255, 255) # Żółty
SKELETON_COLOR = (255, 0, 0) # Niebieski

def draw_keypoints_and_skeleton(img, keypoints, conf_threshold=0.5):
    """Ręczne rysowanie keypoints i szkieletu na ramce."""
    kpts_xy = keypoints[:, :2].astype(int)
    kpts_conf = keypoints[:, 2]

    # Rysowanie połączeń (szkieletu)
    for i, j in SKELETON_EDGES:
        # Sprawdzamy pewność dla obu końców linii
        if kpts_conf[i] > conf_threshold and kpts_conf[j] > conf_threshold:
            pt1 = tuple(kpts_xy[i])
            pt2 = tuple(kpts_xy[j])
            cv2.line(img, pt1, pt2, SKELETON_COLOR, 2)

    # Rysowanie punktów kluczowych
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
    """Generuje ciąg GStreamer dla kamery CSI na Jetsonie"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! appsink"
    )

def run_pose_estimation():
    print("Loading YOLOv8-Pose...")
    model = YOLO('yolov8n-pose.pt').to('cuda:0') 

    pipeline = gstreamer_pipeline(display_width=1280, display_height=720, framerate=30, flip_method=2)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery GStreamer.")
        return

    print("Kamera gotowa. Rozpoczęcie przetwarzania klatek...")
    
    # NOWY PRÓG KĄTA: 140 stopni; STARY PRÓG: 25 stopni
    posture_analyzer = PostureAnalyzer(threshold_deg=25, inactivity_time_sec=5, position_threshold_px=100)
    inference_fps = 0.0 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie można odebrać klatki.")
            break

        inf_start_time = time.time()
        
        # NOWY PRÓG PEWNOŚCI: 0.60
        results = model(frame, verbose=False, conf=0.55, save_conf=True) 
        
        inf_end_time = time.time()
        inference_time = inf_end_time - inf_start_time
        if inference_time > 0:
            inference_fps = 1.0 / inference_time

        annotated_frame = frame.copy() 
        
        # --- Krok 2: Filtracja Detekcji i Wyszukanie Najlepszego Człowieka ---
        
        best_keypoints = None
        best_bbox = None
        max_confidence = 0.0
        
        if results and results[0].boxes:
            for r in results:
                for i in range(len(r.boxes)):
                    box = r.boxes[i]
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls == 0 and conf > max_confidence:
                        max_confidence = conf
                        
                        if r.keypoints.data.shape[0] > i:
                            # Pobieramy punkty, ale też dodajemy kolumnę konfidencji z BB do dalszej pracy
                            kpts = r.keypoints.data[i].cpu().numpy()
                            # W keypoints [x, y, conf], musimy użyć conf z keypoints.
                            best_keypoints = kpts
                            best_bbox = box.xyxy[0].cpu().numpy().astype(int)
                            
        # --- Krok 3: Analiza i Rysowanie TYLKO NAJLEPSZEJ DETEKCJI ---

        if best_keypoints is not None:
            
            # --- Rysowanie Bounding Box i Keypoints ---
            x1, y1, x2, y2 = best_bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            annotated_frame = draw_keypoints_and_skeleton(annotated_frame, best_keypoints, conf_threshold=0.5) 
            
            # --- Analiza Postawy ---
            is_slouching, angle, is_inactivity_warning = posture_analyzer.check_slouching(best_keypoints)
            
            # --- Ustawienia Tekstu ---
            info_text = f"Avg. Posture Angle: {angle:.1f} degrees" 
            color = (0, 255, 0) # Green (BGR)
            warning_text = ""
            
            # --- Fragment z pose_estimation.py (~linia 158) ---
            # 1. Ostrzeżenie o złej postawie
            if is_slouching:
                # Zmieniamy tekst ostrzeżenia:
                warning_text = f"BAD POSTURE! (Deviation > {posture_analyzer.threshold_deg} degrees)" 
                color = (0, 165, 255) # Orange (BGR)
            
            # 2. Ostrzeżenie o bezruchu
            if is_inactivity_warning:
                warning_text = "Please move - back likes to be in the move! :)" 
                color = (0, 0, 255) # Red (BGR)
            
            # --- Wyświetlanie Tekstu ---
            
            cv2.putText(annotated_frame, info_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if warning_text:
                cv2.putText(annotated_frame, warning_text, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
            
            fps_text = f"Inference FPS: {inference_fps:.2f} (CUDA)"
            cv2.putText(annotated_frame, fps_text, (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated_frame, f"No person detected (Conf < {posture_analyzer.threshold_deg})", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Jetson Pose Estimation", annotated_frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_pose_estimation()