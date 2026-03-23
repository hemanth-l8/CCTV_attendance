# 🎭 CPU-based Face Recognition Attendance System

A lightweight, beginner-friendly attendance system using **OpenCV** and **face_recognition** — no GPU required.

---

## 📁 Project Structure

```
face_attendance/
│
├── register.py          # Register a new person's face
├── attendance.py        # Real-time attendance marking
├── utils.py             # Shared helpers (cosine similarity, DB I/O, CSV)
├── requirements.txt     # Python dependencies
│
├── database/
│   └── embeddings.pkl   # Auto-created: stores face embeddings
│
└── attendance.csv       # Auto-created: attendance log
```

---

## ⚙️ Installation

### 1. Install Dependencies

> **Note:** `dlib` (needed by `face_recognition`) requires **CMake** and a **C++ compiler**.
> On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install cmake dlib face_recognition opencv-python numpy
```

---

## 🚀 Usage

### Step 1 — Register a Person

```bash
python register.py
```

- Enter the person's name when prompted.
- A webcam preview will open.
- **Press SPACE** to capture the face.
- Press **Q** or **ESC** to cancel.

Run this once per person (or multiple times to add more embeddings for the same person).

---

### Step 2 — Run Attendance

```bash
python attendance.py
```

- The webcam opens with a live annotated feed.
- Recognised faces get a **green** bounding box with confidence score.
- Unknown faces get a **red-blue** bounding box labelled "Unknown".
- Attendance is logged to `attendance.csv` (once per session per person).
- Press **Q** or **ESC** to quit.

---

## 📊 Attendance CSV Format

```
Name,Date,Time
Alice,2026-02-22,09:15:32
Bob,2026-02-22,09:16:01
```

---

## 🔧 Configuration (in `utils.py`)

| Constant          | Default | Description                               |
|-------------------|---------|-------------------------------------------|
| `THRESHOLD`       | `0.6`   | Cosine similarity threshold (higher = stricter) |

(in `attendance.py`)

| Constant          | Default | Description                               |
|-------------------|---------|-------------------------------------------|
| `PROCESS_SCALE`   | `0.25`  | Frame resize factor for faster detection  |
| `WEBCAM_INDEX`    | `0`     | Camera index (change if using external cam) |

---

## 🧠 How It Works

```
Webcam Frame
     │
     ▼
Resize to 25% ──► BGR → RGB
     │
     ▼
face_recognition.face_locations()   ← detect faces
     │
     ▼
face_recognition.face_encodings()   ← 128-D embeddings
     │
     ▼
find_best_match()
     ├── face_recognition.compare_faces()  ← quick boolean filter
     └── cosine_similarity()               ← rank candidates
     │
     ▼
Mark in attendance.csv (once/session)
     │
     ▼
Draw bounding box + label on frame
```

---

## 💡 Tips

- Register **2–3 different captures** of each person (different angles, lighting) for better accuracy.
- Ensure **good lighting** when both registering and running attendance.
- Keep `THRESHOLD = 0.6` for balanced accuracy vs. false positives.
