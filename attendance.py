"""
attendance.py - Real-time Face Recognition Attendance Marking.

Usage:
    python attendance.py

Steps:
    1. Load face embeddings from database/embeddings.pkl.
    2. Open the webcam and process frames in real-time.
    3. Detect and encode each face in frame.
    4. Compare against stored embeddings.
    5. Mark attendance in attendance.csv (once per session per person).
    6. Draw bounding boxes and names on screen.
    7. Press Q to quit.
"""

import cv2
import face_recognition
import datetime
import sys
import os
import time

# Import shared helpers
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
WEBCAM_INDEX    = 0      # 0 = default webcam
PROCESS_SCALE   = 0.25   # Resize factor for face detection (1/4 size = faster)
WINDOW_TITLE    = "Attendance System  |  Press Q to quit"

# Bounding-box & label colours (BGR)
COLOR_KNOWN     = (0, 220, 100)   # Green  for recognised faces
COLOR_UNKNOWN   = (0, 0, 255)     # Bright Red for unknown faces/intruders
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Folder to store photos of unregistered people
UNKNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "unknown_faces")
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)


def draw_face_box(frame, top, right, bottom, left, label: str, color: tuple) -> None:
    """
    Draw a rectangle and a filled label banner around a detected face.

    Args:
        frame          : Full-resolution BGR frame to draw on.
        top/right/bottom/left : Face location coordinates (full-res scale).
        label          : Text to display (e.g., "Alice (0.91)").
        color          : BGR tuple for the rectangle and banner.
    """
    # ── Bounding rectangle ──
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # ── Filled banner below the box ──
    cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)

    # ── Name text on the banner ──
    cv2.putText(
        frame,
        label,
        (left + 6, bottom - 8),
        FONT,
        0.55,
        (255, 255, 255),   # White text
        1,
    )


def process_frame(frame_bgr, database: dict):
    """
    Detect all faces in one frame and return their locations + identities.

    Performance notes:
        - Frame is resized to PROCESS_SCALE for fast detection.
        - Coordinates are scaled back to the original resolution for drawing.

    Args:
        frame_bgr : Full-resolution BGR frame.
        database  : { "name": [embedding, ...], ... }

    Returns:
        list of dicts:
            [
                {
                    "name"   : str,
                    "score"  : float,
                    "top"    : int,
                    "right"  : int,
                    "bottom" : int,
                    "left"   : int,
                },
                ...
            ]
    """
    results = []

    # ── Step 1: Validate & normalise frame ──
    import numpy as np
    if frame_bgr is None or frame_bgr.size == 0:
        return results
    if frame_bgr.dtype != np.uint8:
        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
    frame_bgr = np.ascontiguousarray(frame_bgr)

    # ── Step 2: Resize frame for faster processing ──
    small_frame = cv2.resize(
        frame_bgr, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE
    )

    # ── Step 3: Convert BGR → RGB (face_recognition requires RGB) ──
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # ── Step 4: Detect face locations in the small frame ──
    face_locations = face_recognition.face_locations(rgb_small)

    if not face_locations:
        return results   # No faces found — return empty list

    # ── Step 4: Generate 128-D encodings for all detected faces ──
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    # ── Step 5: Match each face against the database ──
    scale = int(1 / PROCESS_SCALE)   # e.g., 0.25 → scale back by 4×

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        # Scale coordinates back to original frame size
        top    *= scale
        right  *= scale
        bottom *= scale
        left   *= scale

        # Find best match
        name, score = find_best_match(encoding, database)

        results.append(
            {
                "name"  : name,
                "score" : score,
                "top"   : top,
                "right" : right,
                "bottom": bottom,
                "left"  : left,
            }
        )

    return results


def run_attendance(database: dict) -> None:
    """
    Main attendance loop.

    Opens the webcam, processes frames in real-time, marks attendance,
    and shows annotated video until the user presses Q.

    Args:
        database : Loaded face embeddings dict.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index / drivers.")
        sys.exit(1)

    print("[INFO] Webcam opened. Attendance is running...")
    print("[INFO] Press Q to quit.\n")

    # ── Track who has already been marked this session ──
    marked_today: set = set()
    
    # ── Cooldown for saving 'Unknown' photos (save 1 photo every 10 seconds per session) ──
    last_unknown_save = 0
    unknown_save_cooldown = 10 

    # ── Initialise CSV (create with header if not present) ──
    init_attendance_csv()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        # ── Process one frame per loop ──
        detections = process_frame(frame, database)

        # ── Handle detections ──
        for det in detections:
            name   = det["name"]
            score  = det["score"]
            top    = det["top"]
            right  = det["right"]
            bottom = det["bottom"]
            left   = det["left"]

            if name != "Unknown":
                # ── Mark attendance ONCE per session per person ──
                if name not in marked_today:
                    now      = datetime.datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    write_attendance(name, date_str, time_str)
                    marked_today.add(name)

                label = f"{name} ({score:.2f})"
                color = COLOR_KNOWN
            else:
                label = "NOT REGISTERED"
                color = COLOR_UNKNOWN
                
                # ── Save a photo of the unknown person every X seconds ──
                current_time = time.time()
                if current_time - last_unknown_save > unknown_save_cooldown:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    unknown_path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{timestamp}.jpg")
                    cv2.imwrite(unknown_path, frame)
                    print(f"[ALERT] Unknown person detected! Photo saved to: {unknown_path}")
                    last_unknown_save = current_time

            # ── Draw annotation on the frame ──
            draw_face_box(frame, top, right, bottom, left, label, color)

        # ── Show HUD overlay ──
        cv2.putText(
            frame,
            f"Recognised: {len(marked_today)}  |  Q = quit",
            (10, 30),
            FONT,
            0.65,
            (255, 255, 0),
            2,
        )

        cv2.imshow(WINDOW_TITLE, frame)

        # ── Quit on Q or ESC ──
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            print("[INFO] Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Session ended. Total people marked: {len(marked_today)}")
    if marked_today:
        print(f"       Names: {', '.join(sorted(marked_today))}")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   FACE RECOGNITION ATTENDANCE — ATTENDANCE MODULE")
    print("=" * 55)

    # Step 1: Load stored face embeddings
    database = load_embeddings()

    if not database:
        print("[ERROR] No face embeddings found in database!")
        print("[TIP]   Run register.py first to register at least one person.")
        sys.exit(1)

    print(f"[INFO] Database loaded. Registered people: {list(database.keys())}")
    print(f"[INFO] Similarity threshold: {THRESHOLD}\n")

    # Step 2: Start real-time attendance
    run_attendance(database)


if __name__ == "__main__":
    main()
