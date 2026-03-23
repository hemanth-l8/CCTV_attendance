"""
register.py - Register a new face into the attendance system database.

Usage:
    python register.py

How to capture:
    1. The webcam preview opens automatically.
    2. Go to the terminal, type  C  and press Enter.
    3. A 3-second countdown runs, then the photo is taken.
    4. A green "CAPTURED!" confirmation flashes on screen.
    5. last_capture.jpg is saved — open it to verify.
"""

import cv2
import face_recognition
import sys
import os
import threading
import time
import numpy as np

from utils import load_embeddings, save_embeddings

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
WEBCAM_INDEX = 0
WINDOW_TITLE = "Register Face  —  Watch terminal for instructions"
SAVE_DIR     = os.path.dirname(os.path.abspath(__file__))

# Thread-shared state
_trigger_capture = False
_quit_flag       = False


# ──────────────────────────────────────────────
# Terminal listener (background thread)
# ──────────────────────────────────────────────
def terminal_listener():
    """Read commands from terminal in a background thread."""
    global _trigger_capture, _quit_flag
    print("\n[TERMINAL] Type a command here and press Enter:")
    print("           C  →  Start 3-sec countdown then capture")
    print("           Q  →  Quit\n")
    while not _quit_flag:
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd == "c":
            _trigger_capture = True
        elif cmd == "q":
            _quit_flag = True
            break


# ──────────────────────────────────────────────
# Webcam capture
# ──────────────────────────────────────────────
def read_clean_frame(cap) -> np.ndarray:
    """
    Read one frame from the camera and guarantee it is a clean
    uint8, 3-channel, C-contiguous BGR numpy array.
    Returns None on failure.
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    # Force correct dtype and memory layout
    frame = np.ascontiguousarray(frame, dtype=np.uint8)

    # Handle unexpected channel count
    if frame.ndim == 2:                      # grayscale → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:               # BGRA → BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


def do_countdown(cap, seconds: int = 3) -> np.ndarray:
    """
    Show a countdown overlay on the live feed for `seconds` seconds.
    Captures and returns a FRESH, clean frame AFTER the countdown ends.
    The overlay is drawn on a copy so the returned frame is always clean.
    """
    deadline = time.time() + seconds

    while time.time() < deadline:
        frame = read_clean_frame(cap)
        if frame is None:
            continue

        remaining = int(deadline - time.time()) + 1
        display   = frame.copy()               # draw ONLY on the copy
        h, w = display.shape[:2]
        cv2.putText(display, str(remaining),
                    (w // 2 - 40, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
        cv2.putText(display, "GET READY!",
                    (w // 2 - 110, h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow(WINDOW_TITLE, display)
        cv2.waitKey(100)

    # ── Capture a brand-new, clean frame AFTER countdown finishes ──
    captured = read_clean_frame(cap)
    return captured


def show_captured_flash(cap, captured_frame: np.ndarray):
    """Show a green CAPTURED! banner for 2 seconds."""
    h, w = captured_frame.shape[:2]
    flash    = captured_frame.copy()
    overlay  = flash.copy()
    cv2.rectangle(overlay, (0, h // 2 - 70), (w, h // 2 + 70), (0, 180, 0), -1)
    cv2.addWeighted(overlay, 0.55, flash, 0.45, 0, flash)
    cv2.putText(flash, "CAPTURED!",
                (w // 2 - 160, h // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (255, 255, 255), 6)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        cv2.imshow(WINDOW_TITLE, flash)
        cv2.waitKey(30)


def capture_frame_from_webcam():
    """
    Open the webcam. Background thread watches terminal for C / Q.
    Returns (clean_bgr_frame, success_bool).
    """
    global _trigger_capture, _quit_flag

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return None, False

    # Start terminal listener
    t = threading.Thread(target=terminal_listener, daemon=True)
    t.start()

    print("[INFO] Webcam is live. Go to the TERMINAL, type  C  and press Enter.")

    captured_frame = None

    while not _quit_flag:
        frame = read_clean_frame(cap)
        if frame is None:
            continue

        # Live preview with instruction overlay
        display = frame.copy()
        cv2.putText(display, "Type  C + Enter  in TERMINAL to capture",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display, "Type  Q + Enter  to quit",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.imshow(WINDOW_TITLE, display)
        cv2.waitKey(30)

        if _trigger_capture:
            _trigger_capture = False
            print("[INFO] Starting 3-second countdown...")
            captured_frame = do_countdown(cap)
            if captured_frame is not None:
                show_captured_flash(cap, captured_frame)
                # Save preview image (reloaded clean by imwrite/imread round-trip)
                preview_path = os.path.join(SAVE_DIR, "last_capture.jpg")
                cv2.imwrite(preview_path, captured_frame)
                print(f"[INFO] Photo saved → {preview_path}")
                print("[INFO] Open last_capture.jpg to verify your photo.")
            else:
                print("[ERROR] Failed to grab frame after countdown.")
            _quit_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        return captured_frame, True
    return None, False


# ──────────────────────────────────────────────
# Face encoding
# ──────────────────────────────────────────────
def extract_face_encoding(frame_bgr: np.ndarray) -> list:
    """
    Take a clean BGR uint8 frame, convert to RGB, detect faces,
    and return a list of 128-D face encodings.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        print("[ERROR] Captured frame is empty.")
        return []

    # Reload from disk to guarantee a perfectly clean uint8 RGB-ready image
    # (avoids any residual memory-layout issues from the webcam pipeline)
    preview_path = os.path.join(SAVE_DIR, "last_capture.jpg")
    if os.path.exists(preview_path):
        frame_bgr = cv2.imread(preview_path)   # imread always returns clean uint8 BGR
        print("[INFO] Using saved image for face detection.")

    # BGR → RGB  (face_recognition requires RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)

    print(f"[DEBUG] Frame shape={frame_rgb.shape}  dtype={frame_rgb.dtype}")

    face_locations = face_recognition.face_locations(frame_rgb)

    if not face_locations:
        print("[WARNING] No face detected in the captured image.")
        print("[TIP]     Make sure your face is clearly visible and well-lit.")
        return []

    encodings = face_recognition.face_encodings(frame_rgb, face_locations)
    print(f"[INFO] {len(encodings)} face(s) encoded successfully.")
    return encodings


# ──────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────
def register_face(name: str, encoding) -> None:
    database = load_embeddings()

    if name in database:
        database[name].append(encoding)
        print(f"[INFO] Added new embedding for '{name}' "
              f"(total: {len(database[name])})")
    else:
        database[name] = [encoding]
        print(f"[INFO] Registered new person: '{name}'")

    save_embeddings(database)


def get_person_name() -> str:
    while True:
        name = input("\n[INPUT] Enter the person's name to register: ").strip()
        if name:
            return name
        print("[WARNING] Name cannot be empty. Try again.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   FACE RECOGNITION ATTENDANCE — REGISTER MODULE")
    print("=" * 55)

    name = get_person_name()
    print(f"[INFO] Registering: '{name}'")

    frame, success = capture_frame_from_webcam()

    if not success or frame is None:
        print("[ERROR] No frame captured. Exiting.")
        sys.exit(1)

    encodings = extract_face_encoding(frame)

    if not encodings:
        print("[ERROR] Registration failed — no face found.")
        sys.exit(1)

    register_face(name, encodings[0])

    print(f"\n[SUCCESS] '{name}' registered successfully!")
    print("          Run  python attendance.py  to mark attendance.\n")


if __name__ == "__main__":
    main()
