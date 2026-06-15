<div align="center">

# 🏋️ Curl Counter

**Real-time bicep-curl rep counter — live human-pose estimation fused with a custom YOLO dumbbell detector.**

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Landmarker-FF6F00)](https://ai.google.dev/edge/mediapipe)
[![Ultralytics](https://img.shields.io/badge/YOLO26--L-Ultralytics-7C3AED)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

**Curl Counter** turns your webcam into an AI gym buddy. It tracks your skeleton in real time, measures the angle at your elbow, and automatically counts bicep-curl repetitions — while a custom-trained **YOLO26-L** model simultaneously detects the dumbbell in frame.

<div align="center">

<!-- 📸 Drop a screen recording at assets/demo.gif and it will show up here. -->
<img src="assets/demo.gif" alt="Curl Counter demo" width="720">

<sub><i>Replace <code>assets/demo.gif</code> with your own recording.</i></sub>

</div>

---

## ✨ Features

- 🤸 **Live pose tracking** — MediaPipe **Pose Landmarker (Heavy)** running in `LIVE_STREAM` mode for smooth, low-latency skeleton overlays.
- 📐 **On-frame angle readout** — the elbow angle is computed every frame and drawn right next to the joint being measured.
- 🔢 **Automatic rep counting** — a simple, robust state machine counts a rep only on a full extend → curl cycle (no double counting).
- 🏋️ **Custom dumbbell detection** — a YOLO26-L model (classes: `dumbbell`, `other`) draws a bounding box and a *"Dumbell Detected"* label above the weight.
- 🖥️ **Clean HUD** — rep count, current stage (`up`/`down`), live angle, and detection status, all rendered with OpenCV.
- 💻 **CPU-friendly** — ships with CPU-only PyTorch wheels; no GPU required.

---

## 🧠 How It Works

```
┌──────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
│  Webcam      │ ──▶ │  YOLO26-L (dumbbell)   │ ──▶ │  Draw box + label    │
│  frame (BGR) │     └────────────────────────┘     └──────────────────────┘
│              │     ┌────────────────────────┐     ┌──────────────────────┐
│              │ ──▶ │  MediaPipe PoseLandmark │ ──▶ │  Elbow angle + reps  │
└──────────────┘     └────────────────────────┘     └──────────────────────┘
                                                              │
                                                              ▼
                                                     cv2.imshow("Curl Counter")
```

1. **Pose estimation** — each frame is sent to MediaPipe's Pose Landmarker. The result arrives asynchronously via a callback and is cached for the main loop.
2. **Elbow angle** — using the right-arm landmarks **shoulder (12) → elbow (14) → wrist (16)**, two vectors are built *from the elbow* and the angle between them is computed via the dot product (`arccos`, converted to degrees).
3. **Rep state machine**
   - Angle **> 160°** → arm extended → stage = `down` (re-arms the counter).
   - Angle **< 30°** while previously `down` → arm fully curled → stage = `up` → **+1 rep**.
4. **Dumbbell detection** — the same frame is passed to the YOLO model; every detected box is outlined and labelled. If nothing is found, *"No Dumbell Detected"* is shown in the corner.

---

## 🎛️ The On-Screen HUD

| Element | Location | Meaning |
| --- | --- | --- |
| 🟧 **CURLS** banner | Top-left | Total reps counted + current `stage` (`up` / `down`) |
| 🟡 **`NN deg`** | Next to right elbow | Live elbow angle in degrees |
| 🟩 **Skeleton** | Over your body | MediaPipe pose landmarks & connections |
| 🟥 **Box + "Dumbell Detected"** | Above the dumbbell | YOLO detection (drawn per detected object) |
| 🟥 **"No Dumbell Detected"** | Bottom-left | Shown when no dumbbell is in frame |

Press **`q`** at any time to quit.

---

## 🗂️ Project Structure

```
curl_counter/
├── main.py                 # The whole application (pose + YOLO + HUD loop)
├── models/
│   ├── best_dumb_bell.pt          # Custom YOLO26-L dumbbell detector  (tracked)
│   └── pose_landmarker_heavy.task # MediaPipe pose model (download — not tracked)
├── assets/                 # Screenshots / demo gif for the README
├── requirements.txt        # Pinned dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧰 Tech Stack

| Component | Library | Version |
| --- | --- | --- |
| Pose estimation | [MediaPipe](https://ai.google.dev/edge/mediapipe) Pose Landmarker (Heavy) | `0.10.33` |
| Object detection | [Ultralytics](https://docs.ultralytics.com/) YOLO26-L | `8.4.67` |
| Deep-learning backend | [PyTorch](https://pytorch.org/) (CPU) | `2.12.0+cpu` |
| Computer vision / UI | [OpenCV](https://opencv.org/) (contrib) | `4.13.0.92` |
| Numerics | [NumPy](https://numpy.org/) | `2.4.4` |
| Language | Python | `3.10 – 3.12` |

---

## 📋 Prerequisites

- **Python 3.10 – 3.12** (developed on 3.12).
- A **webcam**.
- A desktop environment with a display (the app opens an OpenCV GUI window — it won't work over a plain headless SSH session).
- ~1.5 GB of disk for the dependencies and models.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/curl_counter.git
cd curl_counter
```

### 2. Create & activate a virtual environment

> Using a dedicated virtual environment is **strongly recommended** so these
> packages don't collide with your system Python.

**Linux / macOS**
```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**
```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

You'll know it's active when your prompt is prefixed with `(venv)`.

### 3. Install dependencies

> ⚠️ **Order matters.** Ultralytics normally pulls in `opencv-python`, which
> *clashes* with the `opencv-contrib-python` that MediaPipe needs (both provide
> the `cv2` module). We avoid the clash by installing ultralytics with
> `--no-deps`. `requirements.txt` already includes ultralytics' other runtime
> dependencies (and CPU-only PyTorch).

```bash
# 1) Everything except the ultralytics package itself
#    (this also installs CPU-only torch from the PyTorch index)
pip install -r requirements.txt

# 2) Ultralytics, WITHOUT its dependencies (so opencv-python is never pulled in)
pip install --no-deps ultralytics==8.4.67
```

<details>
<summary>🔍 Why CPU-only PyTorch?</summary>

This project runs comfortably on a CPU, so `requirements.txt` pins the
`+cpu` PyTorch wheels via `--extra-index-url https://download.pytorch.org/whl/cpu`.
This avoids downloading multi-gigabyte CUDA packages you don't need. If you
*do* have an NVIDIA GPU and want CUDA acceleration, install the matching
CUDA build of `torch`/`torchvision` from [pytorch.org](https://pytorch.org/get-started/locally/) instead.
</details>

### 4. Download the MediaPipe pose model

The pose model isn't bundled in the repo. Download it into `models/`:

```bash
mkdir -p models
wget -O models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
# (or: curl -L -o models/pose_landmarker_heavy.task <same URL>)
```

### 5. The dumbbell model

The custom **YOLO26-L** detector lives at **`models/best_dumb_bell.pt`** and is
already included in this repo, so there's nothing to download.

<details>
<summary>📦 Note on the model file size (Git LFS)</summary>

`best_dumb_bell.pt` is ~50 MB. GitHub accepts files up to 100 MB but shows a
warning above 50 MB. For a cleaner history, consider tracking it with
[Git LFS](https://git-lfs.com/):

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```
</details>

---

## ▶️ Running

```bash
python main.py
```

Stand back so your upper body is visible, grab a dumbbell, and start curling.
The counter increments at the top of each rep. Press **`q`** to quit.

---

## 🔧 Configuration & Tuning

All tunables live near the top of / inside the main loop in **`main.py`**:

| What | Where | Default | Notes |
| --- | --- | --- | --- |
| **Camera index** | `cv2.VideoCapture(0)` | `0` | Try `1`, `2`… for external webcams. |
| **"Down" threshold** | `if required_angle > 160:` | `160°` | Angle above which the arm counts as extended. |
| **"Up" threshold** | `if required_angle < 30 ...` | `30°` | A full curl is often only ~40–50°; raise to `~40` if reps don't register. |
| **Tracked arm** | landmark indices `12, 14, 16` | right arm | Use `11, 13, 15` for the left arm. |
| **Detection confidence** | `YOLO_model(frame, ...)` | `0.25` | Add `conf=0.5` to reduce false positives. |

> 💡 **Tip — the `other` class:** the model has two classes (`dumbbell`, `other`).
> The overlay currently labels *every* detection as a dumbbell. To only react to
> actual dumbbells, filter on the class id inside the detection loop:
> ```python
> for box in yolo_results.boxes:
>     if int(box.cls[0]) != 0:   # 0 == "dumbbell"
>         continue
>     ...
> ```

---

## 🧯 Troubleshooting

<details>
<summary><b>"Failed to open camera"</b></summary>

Another app may be using the webcam, or the index is wrong. Close other camera
apps and try `cv2.VideoCapture(1)`. On Linux, check `ls /dev/video*`.
</details>

<details>
<summary><b>The OpenCV window doesn't appear / crashes on a server</b></summary>

`cv2.imshow` needs a graphical display. Run on a local desktop session, not a
headless/SSH-only machine.
</details>

<details>
<summary><b><code>ImportError</code> around <code>cv2</code> after installing</b></summary>

You likely have **both** `opencv-python` and `opencv-contrib-python` installed,
which conflict. Fix it with:
```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.13.0.92
```
and remember to install ultralytics with `--no-deps`.
</details>

<details>
<summary><b>Model file not found</b></summary>

Make sure `models/pose_landmarker_heavy.task` (step 4) and
`models/best_dumb_bell.pt` both exist. Paths are resolved relative to `main.py`,
so you can run the app from any directory.
</details>

---

## 🗺️ Roadmap

- [ ] Count both arms simultaneously
- [ ] Per-set / per-session rep history and stats
- [ ] Rep tempo & form feedback (e.g. partial-rep detection)
- [ ] Only count reps while a dumbbell is actually detected
- [ ] On-screen FPS and a record-to-video option

---

## 🤝 Contributing

This is a personal learning project, but suggestions and PRs are welcome!
Feel free to open an issue to discuss ideas or report bugs.

---

## 📜 License

Released under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://ai.google.dev/edge/mediapipe) — Pose Landmarker.
- [Ultralytics](https://docs.ultralytics.com/) — YOLO framework.
- [OpenCV](https://opencv.org/) — real-time computer vision & rendering.

<div align="center">
<sub>Built with 💪 as a self-project.</sub>
</div>
