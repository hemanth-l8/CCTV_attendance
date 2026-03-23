"""
utils.py - Helper functions for the Face Recognition Attendance System
"""

import numpy as np
import os
import pickle

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "database", "embeddings.pkl")
ATTENDANCE_CSV  = os.path.join(os.path.dirname(__file__), "attendance.csv")
THRESHOLD       = 0.6   # Cosine-similarity threshold (0 = different, 1 = identical)


# ──────────────────────────────────────────────
# Cosine Similarity
# ──────────────────────────────────────────────
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Returns a value in [0, 1]:
        - 1.0  → identical vectors (same person)
        - 0.0  → completely different vectors (different people)

    Args:
        vec_a: First face embedding (128-D numpy array).
        vec_b: Second face embedding (128-D numpy array).

    Returns:
        float: Cosine similarity score.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a      = np.linalg.norm(vec_a)
    norm_b      = np.linalg.norm(vec_b)

    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ──────────────────────────────────────────────
# Database helpers
# ──────────────────────────────────────────────
def load_embeddings() -> dict:
    """
    Load the face embeddings database from disk.

    Returns:
        dict: { "name": [embedding1, embedding2, ...], ... }
              Returns an empty dict if the file does not exist.
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        return {}

    with open(EMBEDDINGS_PATH, "rb") as f:
        database = pickle.load(f)

    print(f"[INFO] Loaded embeddings for: {list(database.keys())}")
    return database


def save_embeddings(database: dict) -> None:
    """
    Save the face embeddings database to disk.

    Args:
        database (dict): { "name": [embedding1, ...], ... }
    """
    # Make sure the database folder exists
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(database, f)

    print(f"[INFO] Embeddings saved to: {EMBEDDINGS_PATH}")


# ──────────────────────────────────────────────
# Attendance CSV helper
# ──────────────────────────────────────────────
def init_attendance_csv() -> None:
    """
    Create attendance.csv with a header row if it does not already exist.
    """
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w") as f:
            f.write("Name,Date,Time\n")
        print(f"[INFO] Created attendance file: {ATTENDANCE_CSV}")


def write_attendance(name: str, date_str: str, time_str: str) -> None:
    """
    Append one attendance record to attendance.csv.

    Args:
        name     : Person's name.
        date_str : Date string (YYYY-MM-DD).
        time_str : Time string (HH:MM:SS).
    """
    with open(ATTENDANCE_CSV, "a") as f:
        f.write(f"{name},{date_str},{time_str}\n")

    print(f"[ATTENDANCE] Marked: {name}  |  {date_str}  |  {time_str}")


# ──────────────────────────────────────────────
# Face matching
# ──────────────────────────────────────────────
def find_best_match(face_encoding: np.ndarray, database: dict) -> tuple:
    """
    Compare a face encoding against all stored embeddings and return the
    best matching name together with its similarity score.

    Strategy:
        1. Use face_recognition.compare_faces() as a quick pre-filter.
        2. Among passing candidates, pick the one with highest cosine similarity.

    Args:
        face_encoding : 128-D numpy array for the detected face.
        database      : { "name": [embedding1, ...], ... }

    Returns:
        (name, score):
            - name  = matched person's name, or "Unknown" if no match.
            - score = best cosine similarity found (float).
    """
    import face_recognition  # imported here to keep utils importable without the lib

    best_name  = "Unknown"
    best_score = 0.0

    for name, embeddings in database.items():
        for stored_embedding in embeddings:
            # ── Step 1: Quick boolean check via face_recognition ──
            match_result = face_recognition.compare_faces(
                [stored_embedding], face_encoding, tolerance=THRESHOLD
            )

            if match_result[0]:
                # ── Step 2: Compute cosine similarity for ranking ──
                score = cosine_similarity(face_encoding, stored_embedding)
                if score > best_score:
                    best_score = score
                    best_name  = name

    return best_name, best_score
