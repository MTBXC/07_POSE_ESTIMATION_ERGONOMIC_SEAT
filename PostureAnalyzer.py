import numpy as np
import time

class PostureAnalyzer:
    """
    Posture analysis class that calculates the averaged head angle (0 deg at 12 o'clock, counter-clockwise)
    and monitors prolonged inactivity.
    """
    
    def __init__(self, threshold_deg=20, inactivity_time_sec=5, position_threshold_px=10):
        # Deviation threshold (e.g., 20 degrees from vertical 0/360)
        self.threshold_deg = threshold_deg 
        self.inactivity_time_sec = inactivity_time_sec
        self.position_threshold_px = position_threshold_px
        
        # Variables for tracking inactivity
        self.last_position = None
        self.last_move_time = time.time()
        self.is_inactivity_warning = False

    def calculate_angle_360(self, p1, p2):
        """
        Calculates the angle of the vector P1->P2 (shoulder to ear) relative to the vertical (0 deg at 12 o'clock).
        Measures counter-clockwise. Corrects for 180-degree camera flip (flip_method=2).
        """
        v = p2 - p1 # Vector from shoulder (P1) to ear (P2)
        
        # Correct for 180-degree flip by inverting V vector components
        vx_corrected = -v[0]
        vy_corrected = -v[1]
        
        # atan2(x, y) gives 0 at 12:00, measures counter-clockwise
        angle_rad = np.arctan2(vx_corrected, vy_corrected) 
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to 0-360 range
        if angle_deg < 0:
            angle_deg += 360
            
        return angle_deg

    def check_slouching(self, keypoints):
        """
        Analyzes posture and inactivity.
        Returns: (is_slouching, angle_avg, is_inactivity_warning)
        """
        # --- 1. Angle Calculation ---
        angles = []
        
        # Keypoint indices: 3: ear(L), 4: ear(R), 5: shoulder(L), 6: shoulder(R)
        
        # Left side
        ear_L = keypoints[3]
        shoulder_L = keypoints[5]
        if ear_L[2] > 0.1 and shoulder_L[2] > 0.1: 
            angle_L = self.calculate_angle_360(shoulder_L[:2], ear_L[:2])
            angles.append(angle_L)

        # Right side
        ear_R = keypoints[4]
        shoulder_R = keypoints[6]
        if ear_R[2] > 0.1 and shoulder_R[2] > 0.1: 
            angle_R = self.calculate_angle_360(shoulder_R[:2], ear_R[:2])
            angles.append(angle_R)

        # Averaging angles
        if angles:
            angle_avg = np.mean(angles)
        else:
            angle_avg = 0.0
            
        # Slouch verification
        # Calculate distance from the vertical axis (0/360)
        distance_from_vertical = min(angle_avg, 360 - angle_avg)
        
        # Bad posture if deviation from vertical exceeds the threshold
        is_slouching = distance_from_vertical > self.threshold_deg 
        
        # --- 2. INACTIVITY LOGIC ---
        
        # Use the ear with higher confidence as the reference point
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