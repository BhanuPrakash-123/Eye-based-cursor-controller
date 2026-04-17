import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import os
import urllib.request
import math

# 1. Download the Modern Task Model 
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading modern Face Landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# 2. Initialize the Modern Vision API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# 3. Setup Camera and OS Control
cam = cv2.VideoCapture(0)  
screen_w, screen_h = pyautogui.size()
cv2.namedWindow("Eye Controlled Mouse", cv2.WINDOW_NORMAL)

# Stats and state variables
gesture_stats = {
    "Cursor":      {"attempts": 0, "success": 0, "response": 0},
    "Left_Click":  {"attempts": 0, "success": 0, "response": 0},
    "Right_Click": {"attempts": 0, "success": 0, "response": 0},
    "Scroll_Up":   {"attempts": 0, "success": 0, "response": 0},
    "Scroll_Down": {"attempts": 0, "success": 0, "response": 0}
}
last_left_state  = False
last_right_state = False
last_click_time  = time.time()
last_scroll_time = time.time()
last_stat_print  = time.time()

# EAR Settings and Constants
BLINK_EAR_THRESHOLD = 0.20   # EAR drops below this when blinking
SCROLL_THRESHOLD = 0.03      
DEBOUNCE_TIME    = 0.5       
MOUTH_OPEN_THRESHOLD = 0.08  

# Standard MediaPipe 6-point EAR indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2, lm, w, h):
    """Calculates the 2D Euclidean distance between two landmarks."""
    x1, y1 = lm[p1].x * w, lm[p1].y * h
    x2, y2 = lm[p2].x * w, lm[p2].y * h
    return math.hypot(x1 - x2, y1 - y2)

def calculate_ear(eye_indices, lm, w, h):
    """Calculates the Eye Aspect Ratio (EAR) using 6 landmarks."""
    # Vertical distances
    v1 = euclidean_distance(eye_indices[1], eye_indices[5], lm, w, h)
    v2 = euclidean_distance(eye_indices[2], eye_indices[4], lm, w, h)
    # Horizontal distance
    h1 = euclidean_distance(eye_indices[0], eye_indices[3], lm, w, h)
    
    # EAR Formula
    ear = (v1 + v2) / (2.0 * h1)
    return ear

def update_stats(gesture, success, start_ts):
    gesture_stats[gesture]["attempts"] += 1
    if success:
        gesture_stats[gesture]["success"] += 1
        gesture_stats[gesture]["response"] += (time.time() - start_ts) * 1000

def print_stats():
    print("\nGesture Accuracy Report:")
    for key, s in gesture_stats.items():
        at, su = s["attempts"], s["success"]
        if at > 0:
            acc = su / at * 100
            avg = (s["response"] / su) if su > 0 else 0
            print(f"  {key:12s}  Acc: {acc:5.1f}% ({su}/{at})   Avg resp: {avg:.2f} ms")
        else:
            print(f"  {key:12s}  No attempts yet")
    print("-" * 50)

print("Eye Controlled Mouse is active")
print("Open your mouth wide to exit the program")
print("-" * 50)

start_time = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Modern API processing
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(time.time() * 1000)
    
    # Process the frame
    result = landmarker.detect_for_video(mp_image, timestamp)
    h, w, _ = frame.shape

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # Failsafe: Check if mouth is open
        mouth_open = False
        try:
            if (lm[14].y - lm[13].y) > MOUTH_OPEN_THRESHOLD:
                mouth_open = True
                print("Mouth open detected - exiting program")
                break
        except:
            pass 

        # Draw dots on all EAR points for visual feedback
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

        # Calculate Eye Aspect Ratio dynamically
        left_ear = calculate_ear(LEFT_EYE_INDICES, lm, w, h)
        right_ear = calculate_ear(RIGHT_EYE_INDICES, lm, w, h)

        # Cursor Logic
        t0 = time.time()
        gaze = lm[477]
        screen_x, screen_y = gaze.x * screen_w, gaze.y * screen_h
        try:
            pyautogui.moveTo(screen_x, screen_y)
            update_stats("Cursor", True, t0)
        except pyautogui.FailSafeException:
            pass
        
        cv2.circle(frame, (int(gaze.x*w), int(gaze.y*h)), 4, (0,255,0), -1)

        # Left Click Logic (using EAR)
        left_closed = left_ear < BLINK_EAR_THRESHOLD
        if left_closed and not last_left_state and (time.time()-last_click_time) > DEBOUNCE_TIME:
            t0 = time.time()
            pyautogui.click()
            update_stats("Left_Click", True, t0)
            last_click_time = time.time()
        last_left_state = left_closed

        # Right Click Logic (using EAR)
        right_closed = right_ear < BLINK_EAR_THRESHOLD
        if right_closed and not last_right_state and (time.time()-last_click_time) > DEBOUNCE_TIME:
            t0 = time.time()
            pyautogui.rightClick()
            update_stats("Right_Click", True, t0)
            last_click_time = time.time()
        last_right_state = right_closed

        # Scroll Logic
        if (time.time() - last_scroll_time) > DEBOUNCE_TIME:
            tilt = lm[374].y - lm[145].y
            if abs(tilt) > SCROLL_THRESHOLD:
                t0 = time.time()
                if tilt < 0:
                    pyautogui.scroll(300)
                    update_stats("Scroll_Up", True, t0)
                else:
                    pyautogui.scroll(-300)
                    update_stats("Scroll_Down", True, t0)
                last_scroll_time = time.time()

    cv2.putText(frame, f"EAR Threshold: < {BLINK_EAR_THRESHOLD}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, "Open mouth to exit", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("Eye Controlled Mouse", frame)

    if (time.time() - last_stat_print) > 15:
        print_stats()
        last_stat_print = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break

    if time.time() - start_time > 300:  
        print("Session time limit reached - exiting")
        break

print("\nFinal Accuracy Report:")
print_stats()
cam.release()
cv2.destroyAllWindows()
