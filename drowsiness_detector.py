"""
Enhanced Drowsiness Detector with Additional Safety Features
Real-time detection via Webcam (Python + OpenCV + MediaPipe)

Features:
- Drowsiness detection (Eye Aspect Ratio)
- Yawn detection (Mouth Aspect Ratio)
- Head movement/shake detection
- SOS alert system with simulated emergency features
- All in a single file with no external dependencies

Usage:
1. Install required packages: opencv-python, mediapipe, numpy
2. Run this script
3. Press 's' to trigger manual SOS alert
4. Press 'q' to quit
"""

import math
import time
import threading
import random
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import winsound   # Windows built-in beep

# ---------- CONFIG ----------
EAR_THRESHOLD = 0.23       # Higher = less sensitive, Lower = more sensitive
MAR_THRESHOLD = 0.75       # Mouth Aspect Ratio threshold for yawn detection
CONSEC_FRAMES = 18         # ~0.6s at 30 FPS for eye closure
CONSEC_YAWN_FRAMES = 15    # Frames for yawn detection
HEAD_MOVEMENT_THRESHOLD = 15  # Head movement sensitivity
SMOOTHING_WINDOW = 10      # Moving average for stability
BEEP_FREQ = 1400           # Hz
BEEP_DUR_MS = 600          # milliseconds
BEEP_COOLDOWN = 2.0        # seconds between beeps
CAM_INDEX = 0              # default webcam

# Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [61, 84, 17, 314, 405, 320, 307, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# For head pose estimation
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_CONTOURS

# ---------- Emergency Simulation (Fake Data) ----------
FAKE_GPS = "12.9716° N, 77.5946° E"  # Bangalore coordinates
FAKE_CONTACTS = ["Emergency Contact 1: +91 98765 43210", 
                 "Emergency Contact 2: +91 91234 56789",
                 "Nearby Hospital: City General Hospital (1.2 km away)"]

# ---------- Global Variables ----------
_last_beep_time = 0.0
_head_position_history = deque(maxlen=10)
_sos_active = False
_sos_start_time = 0

# ---------- Utility Functions ----------
def _euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def compute_ear(eye_pts):
    """Compute Eye Aspect Ratio (EAR)"""
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = 2.0 * _euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def compute_mar(mouth_pts):
    """Compute Mouth Aspect Ratio (MAR) for yawn detection"""
    # Vertical distances
    vert1 = _euclidean(mouth_pts[13], mouth_pts[14])  # Inner lips vertical
    vert2 = _euclidean(mouth_pts[15], mouth_pts[16])  # Inner lips vertical
    
    # Horizontal distance
    horiz = _euclidean(mouth_pts[0], mouth_pts[10])   # Mouth width
    
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)

def estimate_head_pose(face_landmarks, frame_shape):
    """Estimate head position and movement"""
    h, w = frame_shape[:2]
    nose_tip = face_landmarks.landmark[1]  # Nose tip landmark
    
    # Convert to pixel coordinates
    x = int(nose_tip.x * w)
    y = int(nose_tip.y * h)
    
    return (x, y)

def detect_head_movement(head_positions, threshold=HEAD_MOVEMENT_THRESHOLD):
    """Detect significant head movement"""
    if len(head_positions) < 2:
        return False
    
    movements = []
    for i in range(1, len(head_positions)):
        dx = abs(head_positions[i][0] - head_positions[i-1][0])
        dy = abs(head_positions[i][1] - head_positions[i-1][1])
        movements.append(math.sqrt(dx*dx + dy*dy))
    
    avg_movement = sum(movements) / len(movements) if movements else 0
    return avg_movement > threshold

# ---------- Alert Functions ----------
def beep_non_blocking():
    global _last_beep_time
    now = time.time()
    if now - _last_beep_time < BEEP_COOLDOWN:
        return
    _last_beep_time = now

    def _run():
        winsound.Beep(BEEP_FREQ, BEEP_DUR_MS)

    threading.Thread(target=_run, daemon=True).start()

def trigger_sos_alert():
    """Trigger SOS emergency alert with simulated features"""
    global _sos_active, _sos_start_time
    _sos_active = True
    _sos_start_time = time.time()
    
    # Simulate emergency actions in a thread
    def _run_sos():
        # Beep pattern for SOS (···---···)
        for _ in range(3):  # Three short beeps
            winsound.Beep(1500, 200)
            time.sleep(0.2)
        time.sleep(0.3)
        for _ in range(3):  # Three long beeps
            winsound.Beep(1500, 500)
            time.sleep(0.2)
        time.sleep(0.3)
        for _ in range(3):  # Three short beeps
            winsound.Beep(1500, 200)
            time.sleep(0.2)
            
        # Print simulated emergency info (in a real app, this would send actual alerts)
        print("\n=== EMERGENCY ALERT TRIGGERED ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Location: {FAKE_GPS}")
        print("Emergency Contacts Notified:")
        for contact in FAKE_CONTACTS:
            print(f"  - {contact}")
        print("===============================\n")
    
    threading.Thread(target=_run_sos, daemon=True).start()

# ---------- Main loop ----------
def main():
    global _sos_active, _sos_start_time
    
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        # Try different camera indices if default doesn't work
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Found webcam at index {i}")
                break
        else:
            raise SystemExit("Could not open any webcam. Please check your camera connection.")

    # Set camera resolution (adjust based on your webcam capabilities)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Allow camera to warm up
    print("Initializing camera...")
    time.sleep(2)

    drowsy_counter = 0
    yawn_counter = 0
    ear_history = deque(maxlen=SMOOTHING_WINDOW)
    mar_history = deque(maxlen=SMOOTHING_WINDOW)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        print("Starting detection. Press 's' for SOS, 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            h, w = frame.shape[:2]
            alert_text = ""
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                lm = face_landmarks.landmark

                # Extract eye landmarks
                left_eye_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
                right_eye_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE]
                
                # Extract mouth landmarks
                mouth_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH]

                # Calculate EAR and MAR
                left_ear = compute_ear(left_eye_pts)
                right_ear = compute_ear(right_eye_pts)
                ear = (left_ear + right_ear) / 2.0
                ear_history.append(ear)
                smooth_ear = sum(ear_history) / len(ear_history) if ear_history else 0
                
                mar = compute_mar(mouth_pts)
                mar_history.append(mar)
                smooth_mar = sum(mar_history) / len(mar_history) if mar_history else 0

                # Estimate head pose and detect movement
                head_pos = estimate_head_pose(face_landmarks, frame.shape)
                _head_position_history.append(head_pos)
                head_movement = detect_head_movement(_head_position_history)

                # Draw landmarks
                for pt in left_eye_pts + right_eye_pts:
                    cv2.circle(frame, pt, 1, (0, 255, 0), -1)
                for pt in mouth_pts:
                    cv2.circle(frame, pt, 1, (255, 0, 0), -1)
                
                # Draw head position
                cv2.circle(frame, head_pos, 5, (0, 0, 255), -1)

                # Drowsiness logic
                if smooth_ear < EAR_THRESHOLD:
                    drowsy_counter += 1
                else:
                    drowsy_counter = max(0, drowsy_counter - 1)  # Gradual recovery

                # Yawn detection logic
                if smooth_mar > MAR_THRESHOLD:
                    yawn_counter += 1
                else:
                    yawn_counter = max(0, yawn_counter - 1)

                # Alert conditions
                if drowsy_counter >= CONSEC_FRAMES:
                    alert_text = "DROWSINESS ALERT!"
                    beep_non_blocking()
                
                if yawn_counter >= CONSEC_YAWN_FRAMES:
                    alert_text = "YAWN DETECTED! Take a break."
                    beep_non_blocking()
                    yawn_counter = 0  # Reset after detection
                
                if head_movement:
                    alert_text = "HEAD MOVEMENT DETECTED!"
                    # For sustained head shaking, trigger SOS
                    if len([p for p in list(_head_position_history)[-5:] if p]) >= 3:
                        trigger_sos_alert()

                # Display metrics
                cv2.putText(frame, f"EAR: {smooth_ear:.3f}", (40, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {smooth_mar:.3f}", (40, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Eye Frames: {drowsy_counter}", (40, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Yawn Frames: {yawn_counter}", (40, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                drowsy_counter = 0
                yawn_counter = 0
                cv2.putText(frame, "No face detected", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Display alerts
            if alert_text:
                cv2.putText(frame, alert_text, (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Display SOS status
            if _sos_active:
                # Flash SOS warning
                if (int(time.time() * 3) % 2) == 0:  # Blinking effect
                    cv2.putText(frame, "SOS EMERGENCY ALERT!", (w//2 - 200, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                
                # Show emergency info
                cv2.putText(frame, f"GPS: {FAKE_GPS}", (w - 500, h - 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                for i, contact in enumerate(FAKE_CONTACTS):
                    cv2.putText(frame, contact, (w - 500, h - 90 + i*30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Auto-cancel SOS after 10 seconds
                if time.time() - _sos_start_time > 10:
                    _sos_active = False

            # UI Elements
            cv2.rectangle(frame, (20, 20), (500, 250), (50, 50, 50), 2)
            cv2.putText(frame, "Press 's' for SOS, 'q' to quit", (40, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow('Enhanced Drowsiness Detector', frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                trigger_sos_alert()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()