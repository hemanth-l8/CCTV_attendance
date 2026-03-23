"""
attendance_emulator.py - Face Recognition Attendance using Emulator Screen Capture.
"""

import cv2
import numpy as np
import face_recognition
import pyautogui
import datetime
import os
import time
import sys

# Import shared helpers from utils.py
from utils import (
    load_embeddings,
    init_attendance_csv,
    write_attendance,
    find_best_match,
    THRESHOLD,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
# Region defined by user: (x, y, width, height)
CAPTURE_REGION = (8, 430, 504, 282)

PROCESS_SCALE   = 0.5    # Increased to 0.5 for better detection in emulator
WINDOW_TITLE    = "Emulator Attendance System | Q to quit"

# Bounding-box & label colours (BGR)
COLOR_KNOWN     = (0, 220, 100)   # Green for recognised
COLOR_UNKNOWN   = (0, 0, 255)     # Red for unknown
font            = cv2.FONT_HERSHEY_SIMPLEX

# Folder to store photos of unregistered people
UNKNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "unknown_faces")
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)


def capture_emulator_frame():
    """Captures emulator region and converts to BGR."""
    try:
        screenshot = pyautogui.screenshot(region=CAPTURE_REGION)
        frame = np.array(screenshot)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        print(f"[ERROR] Screen capture failed: {e}")
        return None


def draw_face_box(frame, top, right, bottom, left, label: str, color: tuple) -> None:
    """Draws bounding box and name label on the frame."""
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, label, (left + 6, bottom - 8), font, 0.55, (255, 255, 255), 1)


def process_frame(frame_bgr, database):
    """Detects and recognizes faces."""
    results = []
    if frame_bgr is None or frame_bgr.size == 0:
        return results
    
    # Normalise for dlib
    frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)

    # Step 1: Resize
    small_frame = cv2.resize(frame_bgr, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Step 2: Detect
    face_locations = face_recognition.face_locations(rgb_small)
    if not face_locations:
        return results

    print(f"[DEBUG] Detected {len(face_locations)} face(s)")

    # Step 3: Encode and Match
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
    scale = int(1 / PROCESS_SCALE)
    
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale

        name, score = find_best_match(encoding, database)
        results.append({
            "name": name,
            "score": score,
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left
        })

    return results


def run_attendance(database):
    """Main attendance loop."""
    print(f"[INFO] Starting Emulator Attendance capture at {CAPTURE_REGION}")
    print("[INFO] Press 'Q' to quit.\n")

    init_attendance_csv()
    marked_today = set()
    
    last_unknown_save = 0
    unknown_save_cooldown = 10 
    
    frame_count = 0
    last_detections = [] # To prevent flickering

    while True:
        frame = capture_emulator_frame()
        if frame is None:
            time.sleep(1)
            continue

        frame_count += 1
        
        # Process every 2nd frame but keep drawing the old detections on skipped frames
        if frame_count % 2 == 0:
            last_detections = process_frame(frame, database)

        # Draw detections
        for det in last_detections:
            name = det["name"]
            score = det["score"]
            top, right, bottom, left = det["top"], det["right"], det["bottom"], det["left"]

            if name != "Unknown":
                if name not in marked_today:
                    now = datetime.datetime.now()
                    write_attendance(name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
                    marked_today.add(name)
                label = f"{name} ({score:.2f})"
                color = COLOR_KNOWN
            else:
                label = "NOT REGISTERED"
                color = COLOR_UNKNOWN

                now_ts = time.time()
                if now_ts - last_unknown_save > unknown_save_cooldown:
                    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    unknown_path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{ts_str}.jpg")
                    cv2.imwrite(unknown_path, frame)
                    print(f"[ALERT] Unknown person detected! Saved to {unknown_path}")
                    last_unknown_save = now_ts

            draw_face_box(frame, top, right, bottom, left, label, color)

        # HUD
        cv2.putText(frame, f"Marked: {len(marked_today)} | Q = Quit", (10, 20), font, 0.5, (255, 255, 0), 1)
        cv2.imshow(WINDOW_TITLE, frame)

        # Smaller sleep for smoother UI
        time.sleep(0.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    database = load_embeddings()
    if not database:
        print("[ERROR] Database empty. Use register.py.")
        sys.exit(1)
    run_attendance(database)


if __name__ == "__main__":
    main()
