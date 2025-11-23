# --- Plik: PostureAnalyzer.py (FINALNY KOD) ---
import numpy as np
import time

class PostureAnalyzer:
    """
    Analizator postawy, który oblicza uśredniony kąt pochylenia (0 st. na 12stej, mierzy przeciwnie do ruchu wskazówek)
    i monitoruje bezruch.
    """
    
    def __init__(self, threshold_deg=20, inactivity_time_sec=5, position_threshold_px=10):
        # Próg odchylenia (np. 20 stopni od 0/360)
        self.threshold_deg = threshold_deg 
        self.inactivity_time_sec = inactivity_time_sec
        self.position_threshold_px = position_threshold_px
        
        # Zmienne do śledzenia bezruchu
        self.last_position = None
        self.last_move_time = time.time()
        self.is_inactivity_warning = False

    def calculate_angle_360(self, p1, p2):
        """
        Oblicza kąt wektora P1->P2 (ramię do ucha) względem pionu (0 stopni na 12stej).
        Mierzy przeciwnie do ruchu wskazówek zegara.
        UWZGLĘDNIA FLIP KAMERY O 180 STOPNI (flip_method=2).
        """
        v = p2 - p1 # Wektor od ramienia (P1) do ucha (P2)
        
        # KORYGUJEMY OBRÓT 180 STOPNI:
        # Odwracamy znak obu składowych wektora V
        vx_corrected = -v[0]
        vy_corrected = -v[1]
        
        # atan2(x, y) - to daje nam 0 na 12:00 i mierzy zgodnie z ruchem wskazówek
        # atan2(y, x) - to mierzy przeciwnie
        
        # Używamy kombinacji, która daje 0 na 12:00 i mierzy przeciwnie do ruchu wskazówek
        angle_rad = np.arctan2(vx_corrected, vy_corrected) 
        angle_deg = np.degrees(angle_rad)
        
        # Normalizacja do zakresu 0-360
        if angle_deg < 0:
            angle_deg += 360
            
        return angle_deg

    def check_slouching(self, keypoints):
        """
        Analizuje postawę i bezruch.
        Zwraca: (is_slouching, angle_avg, is_inactivity_warning)
        """
        # --- 1. Obliczenie Kątów ---
        angles = []
        
        # Indeksy keypoints: 3: ear(L), 4: ear(R), 5: shoulder(L), 6: shoulder(R)
        
        # Lewa strona
        ear_L = keypoints[3]
        shoulder_L = keypoints[5]
        if ear_L[2] > 0.1 and shoulder_L[2] > 0.1: 
             angle_L = self.calculate_angle_360(shoulder_L[:2], ear_L[:2])
             angles.append(angle_L)

        # Prawa strona
        ear_R = keypoints[4]
        shoulder_R = keypoints[6]
        if ear_R[2] > 0.1 and shoulder_R[2] > 0.1: 
             angle_R = self.calculate_angle_360(shoulder_R[:2], ear_R[:2])
             angles.append(angle_R)

        # Uśrednianie kątów
        if angles:
            angle_avg = np.mean(angles)
        else:
            angle_avg = 0.0
            
        # Weryfikacja garbienia
        # Obliczamy, jak daleko kąt jest od pionu (0/360)
        distance_from_vertical = min(angle_avg, 360 - angle_avg)
        
        # Zła postawa, gdy odchylenie od pionu jest większe niż próg (np. 20 stopni)
        is_slouching = distance_from_vertical > self.threshold_deg 
        
        # --- 2. LOGIKA BEZRUCHU ---
        
        current_position_full = ear_L if ear_L[2] > ear_R[2] else ear_R
        
        if current_position_full[2] < 0.1: 
             current_position_coords = np.array([0, 0])
        else:
             current_position_coords = current_position_full[:2] 
        
        if self.last_position is None:
            self.last_position = current_position_coords
            self.last_move_time = time.time()
            self.is_inactivity_warning = False
        else:
            distance = np.linalg.norm(current_position_coords - self.last_position)
            
            if distance > self.position_threshold_px:
                self.last_position = current_position_coords
                self.last_move_time = time.time()
                self.is_inactivity_warning = False
            else:
                inactivity_duration = time.time() - self.last_move_time
                if inactivity_duration > self.inactivity_time_sec:
                    self.is_inactivity_warning = True
                
        return is_slouching, angle_avg, self.is_inactivity_warning