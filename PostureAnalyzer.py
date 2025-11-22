import numpy as np
import math

class PostureAnalyzer:
    """
    Klasa do analizowania kluczowych punktów zwróconych przez model Pose Estimation 
    i oceniania, czy postawa jest zgarbiona (Slouching).
    """
    
    # Indeksy kluczowych punktów w modelu YOLOv8-Pose
    R_EAR_IDX = 3
    L_EAR_IDX = 4
    R_SHOULDER_IDX = 5
    L_SHOULDER_IDX = 6
    
    def __init__(self, threshold_deg=25):
        """
        Inicjalizacja analizatora z domyślnym progiem dla złej postawy.
        :param threshold_deg: Kąt (w stopniach) powyżej którego postawa jest uznawana za złą.
        """
        self.threshold_deg = threshold_deg

    def calculate_tilt_angle(self, p_ear, p_shoulder):
        """
        Oblicza kąt nachylenia linii Ucho (p_ear) - Ramię (p_shoulder) względem osi pionowej.
        Używa cosinusa kąta między wektorem Ucho->Ramię a wektorem pionowym (0, -1).
        Wektor Ucho->Ramię: v_arm = p_shoulder - p_ear
        Wektor Pionowy: v_vertical = (0, 1) lub (0, -1) - dowolny pionowy.
        Zwraca kąt (w stopniach) odchylenia od pionu.
        """
        
        p_ear = np.array(p_ear)      # Ucho
        p_shoulder = np.array(p_shoulder)  # Ramię
        
        # 1. Wektor od Ucha do Ramienia (kierunek, w którym jest ramię od głowy)
        v_head_arm = p_shoulder - p_ear
        
        # 2. Wektor Pionowy (w kierunku w dół)
        # Musimy użyć pionowego wektora, żeby porównać z nim nachylenie szyi/głowy
        # Zwykle to po prostu (0, 1), zakładając, że Y rośnie w dół w OpenCV
        v_vertical = np.array([0, 1]) 
        
        # 3. Iloczyn skalarny i normy
        dot_product = np.dot(v_head_arm, v_vertical)
        magnitude_v_head_arm = np.linalg.norm(v_head_arm)
        magnitude_v_vertical = np.linalg.norm(v_vertical)

        # Obliczenie cosinusa kąta
        if magnitude_v_head_arm == 0:
            return 0.0
            
        cosine_angle = dot_product / (magnitude_v_head_arm * magnitude_v_vertical)
        
        # 4. Kąt w radianach i stopniach
        # Kąt jest mierzony od wektora pionowego. Kąt bliski 0 oznacza, że Ucho jest nad Ramieniem (dobra postawa).
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Idealna postawa (prosto w dół) to kąt bliski 0°
        # Zgarbiona postawa to kąt bliski 90° lub więcej.
        
        # Aby postawa była prosta, wektor Ucho->Ramię (v_head_arm) powinien być skierowany pionowo (kąt 0°)
        # Gdy się garbisz, Ucho przesuwa się do przodu, a kąt względem pionu rośnie.
        # W widoku z profilu, gdy jest prosto, wektor v_head_arm jest w dół, a kąt jest bliski 0 stopni.
        # Jeśli Ucho się wysuwa (garbienie), kąt rośnie, np. do 25 stopni lub więcej.

        # Wymuszamy, aby mierzyć odchylenie od osi pionowej w dół, co jest prawidłowe dla postawy.
        return angle_deg

    def check_slouching(self, keypoints):
        """
        Ocenia, czy postawa jest zgarbiona, mierząc nachylenie linii Ucho-Ramię.

        :param keypoints: Macierz kluczowych punktów dla jednej osoby [33, 3 (x, y, conf)].
        :return: Tuple (is_slouching, angle), gdzie is_slouching to bool.
        """
        
        current_angle = 0.0
        detected = False
        
        # --- 1. Próba użycia Prawy Profil ---
        if keypoints[self.R_EAR_IDX, 2] > 0.5 and keypoints[self.R_SHOULDER_IDX, 2] > 0.5:
            p_ear = keypoints[self.R_EAR_IDX, :2]
            p_shoulder = keypoints[self.R_SHOULDER_IDX, :2]
            current_angle = self.calculate_tilt_angle(p_ear, p_shoulder)
            detected = True
        
        # --- 2. Próba użycia Lewy Profil ---
        elif keypoints[self.L_EAR_IDX, 2] > 0.5 and keypoints[self.L_SHOULDER_IDX, 2] > 0.5:
            p_ear = keypoints[self.L_EAR_IDX, :2]
            p_shoulder = keypoints[self.L_SHOULDER_IDX, :2]
            current_angle = self.calculate_tilt_angle(p_ear, p_shoulder)
            detected = True

        # Ocena postawy
        if detected:
            is_slouching = current_angle > self.threshold_deg
            return is_slouching, current_angle
        
        return False, 0.0 # Jeśli punkty nie zostały wykryte lub mają niską pewność