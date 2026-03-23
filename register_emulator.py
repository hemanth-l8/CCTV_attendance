"""
register_emulator.py - Register a new face using the Emulator Screen Capture.

How to use:
    1. Make sure the emulator (EzyKam+) is visible on your screen.
    2. Run this script: python register_emulator.py
    3. Type the person's name in the terminal.
    4. Type 'C' in the terminal to start the 3-second countdown.
    5. The system will grab the face from the emulator and save it.
"""

import cv2
import numpy as np
import face_recognition
import pyautogui
import os
import sys
import threading
import time

from utils import load_embeddings, save_embeddings

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
# Using your verified coordinates
CAPTURE_REGION = (8, 430, 504, 282)
WINDOW_TITLE   = "Register via Emulator — Watch terminal"
SAVE_DIR       = os.path.dirname(os.path.abspath(__file__))

_trigger_capture = False
_quit_flag       = False


def terminal_listener():
    """Reads commands from terminal."""
    global _trigger_capture, _quit_flag
    print("\n[TERMINAL] Instructions:")
    print("           Type  C  and Enter  →  Capture after 3s countdown")
    print("           Type  Q  and Enter  →  Quit\n")
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


def capture_emulator_frame():
    """Grabs frame from screen region."""
    try:
        screenshot = pyautogui.screenshot(region=CAPTURE_REGION)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ERROR] Capture failed: {e}")
        return None


def do_countdown_and_capture():
    """Shows countdown on screen and returns the final clean frame."""
    deadline = time.time() + 3.0
    while time.time() < deadline:
        frame = capture_emulator_frame()
        if frame is None: continue
        
        remaining = int(deadline - time.time()) + 1
        display = frame.copy()
        h, w = display.shape[:2]
        cv2.putText(display, str(remaining), (w // 2 - 40, h // 2 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)
        cv2.imshow(WINDOW_TITLE, display)
        cv2.waitKey(100)
    
    # Grab final clean frame
    return capture_emulator_frame()


def extract_and_register(frame_bgr, name):
    """Processes frame and saves to database."""
    if frame_bgr is None:
        print("[ERROR] No frame grabbed.")
        return False

    # Save temp file for verification
    preview_path = os.path.join(SAVE_DIR, "last_register_cap.jpg")
    cv2.imwrite(preview_path, frame_bgr)
    
    # Normalise for face_recognition
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    
    print(f"[DEBUG] Checking for faces in {preview_path}...")
    face_locations = face_recognition.face_locations(rgb)
    
    if not face_locations:
        print("[WARNING] No face found in the emulator feed!")
        print("[TIP] Make sure the person is looking towards the CCTV camera.")
        return False
        
    encodings = face_recognition.face_encodings(rgb, face_locations)
    encoding = encodings[0]
    
    # Save to database
    db = load_embeddings()
    if name in db:
        db[name].append(encoding)
    else:
        db[name] = [encoding]
    
    save_embeddings(db)
    print(f"[SUCCESS] Registered '{name}' from emulator stream!")
    return True


def main():
    global _trigger_capture, _quit_flag
    
    print("=" * 55)
    print("   FACE REGISTRATION — EMULATOR CAPTURE MODULE")
    print("=" * 55)
    
    name = input("\n[INPUT] Enter person's name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        return

    # Start terminal listener
    t = threading.Thread(target=terminal_listener, daemon=True)
    t.start()
    
    print(f"[INFO] Feed area: {CAPTURE_REGION}")

    while not _quit_flag:
        frame = capture_emulator_frame()
        if frame is None: continue
        
        # Show preview
        display = frame.copy()
        cv2.putText(display, f"Registering: {name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Type C in terminal to start", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WINDOW_TITLE, display)
        cv2.waitKey(30)
        
        if _trigger_capture:
            _trigger_capture = False
            print("[INFO] Countdown started...")
            captured_frame = do_countdown_and_capture()
            if extract_and_register(captured_frame, name):
                # Flash captured result
                h, w = captured_frame.shape[:2]
                cv2.putText(captured_frame, "CAPTURED!", (w // 2 - 140, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.imshow(WINDOW_TITLE, captured_frame)
                cv2.waitKey(2000)
            _quit_flag = True
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
