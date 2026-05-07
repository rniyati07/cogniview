# COGNIVIEW — Development Blueprint
### Real-Time AI-Powered Surveillance System · Execution Guide v1.0
> **Status:** Prototype · **Stack:** Python 3.9+ · OpenCV 4.8+ · YOLOv8 · NumPy · PyYAML  
> **Target Environment:** Antigravity IDE / Any Python-capable IDE  
> **Audience:** Student Developer — build this top-to-bottom, phase by phase.

---

## TABLE OF CONTENTS

1. [System Breakdown — High-Level Modules](#1-system-breakdown)
2. [Development Workflow — Phased Execution Plan](#2-development-workflow)
3. [File & Code Structure Mapping](#3-file--code-structure-mapping)
4. [Data Strategy Integration](#4-data-strategy-integration)
5. [Core Logic Flow — End-to-End Pipeline](#5-core-logic-flow)
6. [Key Algorithms — Simplified](#6-key-algorithms)
7. [Configuration Design](#7-configuration-design)
8. [Testing Strategy](#8-testing-strategy)
9. [Deliverables Mapping](#9-deliverables-mapping)

---

## 1. SYSTEM BREAKDOWN

> Each module is a single-responsibility unit. Build, test, and validate each independently before wiring them together.

---

### MODULE 1 — Video Input

| Field | Detail |
|---|---|
| **Purpose** | Captures raw video frames from either a live webcam or a pre-recorded file and streams them to the pipeline. |
| **Inputs** | Webcam index (integer, e.g. `0`) OR video file path (`.mp4`, `.avi`) |
| **Outputs** | A continuous stream of BGR-format NumPy frames (H × W × 3 arrays) |
| **Key Functions** | `open_source(source)` · `read_frame()` · `release()` · `get_fps()` |

**What it does in plain terms:**  
This is the entry point of the whole system. Every other module waits for frames from here. It wraps OpenCV's `VideoCapture` and adds basic error handling — if the source fails to open, the system should exit cleanly rather than crash silently.

---

### MODULE 2 — Preprocessing

| Field | Detail |
|---|---|
| **Purpose** | Normalises raw frames so the detection engine always receives consistent, well-formed input. |
| **Inputs** | Raw BGR frame from the Video Input module |
| **Outputs** | Resized, RGB-converted frame ready for YOLO inference |
| **Key Functions** | `resize_frame(frame, width, height)` · `bgr_to_rgb(frame)` · `equalise_histogram(frame)` |

**What it does in plain terms:**  
YOLOv8 expects RGB input at a fixed resolution. OpenCV produces BGR. This module bridges that gap. It also optionally applies histogram equalisation — a single OpenCV call that improves detection in low-light frames by stretching contrast.

---

### MODULE 3 — Detection Engine (YOLOv8)

| Field | Detail |
|---|---|
| **Purpose** | Runs per-frame inference using the pretrained YOLOv8 model and returns person-class detections only. |
| **Inputs** | Preprocessed RGB frame |
| **Outputs** | List of detections, each as `(x1, y1, x2, y2, confidence)` — bounding box + score |
| **Key Functions** | `load_model(model_path)` · `run_inference(frame)` · `filter_persons(results)` |

**What it does in plain terms:**  
Load the model once at startup — never inside the frame loop. Call `model(frame)` on each frame. The result contains detections for all 80 COCO classes; immediately filter to `class_id == 0` (person). Return only the box coordinates and confidence scores for those person detections.

---

### MODULE 4 — Intrusion Logic (ROI Engine)

| Field | Detail |
|---|---|
| **Purpose** | Determines whether any detected person's bounding box overlaps with a defined Restricted Zone (ROI). |
| **Inputs** | List of person bounding boxes · List of ROI rectangles from config |
| **Outputs** | Boolean flag per ROI: `True` if intrusion candidate detected, `False` otherwise |
| **Key Functions** | `load_rois(config)` · `check_intersection(box, roi)` · `get_intruding_boxes(detections, rois)` |

**What it does in plain terms:**  
This is a pure geometry check — no ML involved. For each detected person box and each ROI rectangle, run the AABB (Axis-Aligned Bounding Box) intersection formula. If any corner or region of the person box falls inside the ROI, flag it as a candidate intrusion. A "candidate" is not yet an alert — it still needs to pass the filtering layer.

---

### MODULE 5 — Filtering Layer

| Field | Detail |
|---|---|
| **Purpose** | Eliminates false alarms by requiring intrusions to be both high-confidence and temporally sustained before an alert fires. |
| **Inputs** | Intrusion candidate flags (per ROI) · Confidence score of triggering detection · Frame timestamp |
| **Outputs** | `alert_confirmed: bool` — only `True` when all filters pass |
| **Key Functions** | `update_frame_counter(roi_id, intruding)` · `check_confidence(score, threshold)` · `check_cooldown(roi_id, current_time)` · `reset_counter(roi_id)` |

**What it does in plain terms:**  
Three filters in sequence:
1. **Confidence filter** — drop any detection scoring below 0.5 (configurable).
2. **Temporal filter** — only raise an alert after a person has been inside the ROI for N consecutive frames (default N=3). Maintain an `intrusion_counter` per ROI that increments on positive frames and resets on negative ones.
3. **Cooldown filter** — once an alert fires, record the timestamp and suppress further alerts for that ROI for a configurable cooldown period (default: 10 seconds). This prevents alert storms from a person standing still inside the zone.

---

### MODULE 6 — Alert System

| Field | Detail |
|---|---|
| **Purpose** | Persists confirmed intrusion events by saving annotated images and writing structured log entries. |
| **Inputs** | Confirmed alert flag · Annotated frame · ROI name · Confidence score · Frame number |
| **Outputs** | PNG image file in `/alerts/` · Appended row in `event_log.csv` · Console print |
| **Key Functions** | `save_alert_image(frame, roi_name, timestamp)` · `write_log_entry(log_path, record)` · `print_console_alert(roi_name, confidence)` |

**What it does in plain terms:**  
When a confirmed alert arrives, do three things simultaneously: write the annotated frame to disk as `alert_YYYYMMDD_HHMMSS_ROIID.png`, append a CSV row with `[timestamp, frame_num, roi_name, confidence]`, and print a console message. The image filename must embed the timestamp so that no two alerts ever overwrite each other.

---

### MODULE 7 — Visualization Layer

| Field | Detail |
|---|---|
| **Purpose** | Renders a live annotated video window that gives the operator real-time feedback on detections and intrusion status. |
| **Inputs** | Original BGR frame · Detection list · ROI list · Intrusion status per ROI · FPS value |
| **Outputs** | Annotated frame displayed in an OpenCV `imshow` window |
| **Key Functions** | `draw_detections(frame, detections, intruding_ids)` · `draw_rois(frame, rois, status)` · `draw_overlay_text(frame, fps, status)` · `show_frame(frame)` |

**What it does in plain terms:**  
Draw on a copy of the frame — never modify the original. Use green boxes for persons detected outside any ROI. Switch to red for persons triggering an ROI. Draw the ROI boundary rectangle always visible. Overlay FPS counter and a status string (`"CLEAR"` or `"⚠ INTRUSION DETECTED"`) in the top-left corner. Call `cv2.imshow()` then `cv2.waitKey(1)` — if the user presses `q`, the loop exits.

---

## 2. DEVELOPMENT WORKFLOW

> Build in strict phase order. Each phase has a clear acceptance test before you move forward.

---

### PHASE 1 — Core Detection

**Goal:** Get YOLOv8 detecting persons in a video feed and displaying the result in a live window.

| Task | Description |
|---|---|
| T1.1 | Set up Python virtual environment and install dependencies (`ultralytics`, `opencv-python`, `numpy`, `pyyaml`) |
| T1.2 | Implement `VideoInput` class in `video_input.py` wrapping `cv2.VideoCapture` |
| T1.3 | Implement `preprocess_frame()` in `preprocessing.py` — resize to 640×480, convert BGR→RGB |
| T1.4 | Implement `DetectionEngine` in `detection.py` — load `yolov8n.pt`, run inference, filter class 0 |
| T1.5 | Implement `draw_detections()` in `visualization.py` — green bounding boxes + confidence labels |
| T1.6 | Wire together in `main.py` — capture → preprocess → detect → visualise loop |
| T1.7 | Test with webcam: confirm person detected, bounding box shown, loop runs at ≥ 10 FPS |

**Expected Output:** A live window showing your webcam feed with green bounding boxes drawn around detected persons. FPS counter visible.

**Dependencies:** None (this is the foundation).

---

### PHASE 2 — Intrusion Logic

**Goal:** Define an ROI zone on the frame and detect when a person enters it.

| Task | Description |
|---|---|
| T2.1 | Create `config.yaml` with a hardcoded test ROI (e.g. centre of frame at 640×480) |
| T2.2 | Implement `load_rois()` in `intrusion.py` — parse ROI list from config |
| T2.3 | Implement `check_intersection()` in `intrusion.py` — AABB overlap formula |
| T2.4 | Implement `get_intruding_boxes()` — returns list of detections that intersect any ROI |
| T2.5 | Update `visualization.py` — draw ROI boundary rectangle; switch intruding boxes to red |
| T2.6 | Add status text: `"CLEAR"` vs `"⚠ INTRUSION"` overlay on frame |
| T2.7 | Test: walk into the defined zone, confirm box turns red and status changes |

**Expected Output:** Live window with a visible ROI rectangle. Boxes are green outside the zone, red inside. Status text updates in real time.

**Dependencies:** Phase 1 complete.

---

### PHASE 3 — Filtering System

**Goal:** Eliminate false positives by adding confidence + temporal + cooldown filters.

| Task | Description |
|---|---|
| T3.1 | Implement `check_confidence(score, threshold)` in `filtering.py` |
| T3.2 | Implement `TemporalFilter` class — per-ROI counter, increment/reset logic |
| T3.3 | Implement `CooldownFilter` class — timestamp-based suppression per ROI |
| T3.4 | Wire filters into the main loop: candidate intrusion → confidence check → frame counter → cooldown → confirmed alert |
| T3.5 | Read all filter thresholds from `config.yaml` (confidence, N frames, cooldown seconds) |
| T3.6 | Test: have a person walk partially into the zone — alert should NOT fire until 3 consecutive frames |
| T3.7 | Test: trigger an alert, then remain in zone — only one alert should fire during cooldown window |

**Expected Output:** The system now only fires alerts for sustained, high-confidence detections. Brief walk-throughs under N frames do NOT trigger alerts.

**Dependencies:** Phase 2 complete.

---

### PHASE 4 — Alerts & Logging

**Goal:** Persist confirmed intrusion events as image files and structured log entries.

| Task | Description |
|---|---|
| T4.1 | Create `/alerts/` directory in project root (add `.gitkeep`, not real images, to git) |
| T4.2 | Implement `save_alert_image()` in `alert.py` — `cv2.imwrite` with UTC-timestamped filename |
| T4.3 | Implement `write_log_entry()` — open `event_log.csv` in append mode, write structured row |
| T4.4 | Log fields: `timestamp (ISO8601), frame_number, roi_name, confidence, image_filename` |
| T4.5 | Implement `print_console_alert()` — formatted print with colour if terminal supports it |
| T4.6 | Wire `AlertSystem` into main loop: on confirmed alert, call all three output functions |
| T4.7 | Test: trigger an alert, verify PNG appears in `/alerts/`, verify CSV row appended correctly |

**Expected Output:** Every confirmed intrusion produces: one annotated PNG in `/alerts/` + one row in `event_log.csv` + a console print. Files accumulate correctly across multiple test runs.

**Dependencies:** Phase 3 complete.

---

### PHASE 5 — Testing & Optimization

**Goal:** Run full system against multiple test scenarios, measure performance, tune parameters.

| Task | Description |
|---|---|
| T5.1 | Run system on VIRAT dataset clip — observe detection recall and false positive count |
| T5.2 | Run system on UCF Crime clip — test edge cases (fast movement, crowded frames) |
| T5.3 | Run system on custom low-light webcam recording — enable histogram equalisation, compare results |
| T5.4 | Measure FPS with `cv2.getTickCount()` — document on target hardware |
| T5.5 | Tune `confidence_threshold` and `frame_threshold` in config — re-run tests to compare FP rates |
| T5.6 | Check for memory leaks or performance degradation during a 10-minute continuous run |
| T5.7 | Document all findings: detection recall estimate, FP count, latency, known limitations |

**Expected Output:** A tuned system that runs stably for 10+ minutes, with documented performance numbers and a written evaluation summary.

**Dependencies:** Phases 1–4 complete.

---

## 3. FILE & CODE STRUCTURE MAPPING

```
cogniview/
│
├── main.py                  ← Orchestration: wires all modules, runs the main loop
├── video_input.py           ← Module 1: VideoCapture wrapper
├── preprocessing.py         ← Module 2: resize, BGR→RGB, histogram equalisation
├── detection.py             ← Module 3: YOLOv8 inference, class 0 filter
├── intrusion.py             ← Module 4: ROI loading, AABB intersection logic
├── filtering.py             ← Module 5: confidence, temporal, cooldown filters
├── alert.py                 ← Module 6: PNG save, CSV log, console print
├── visualization.py         ← Module 7: draw boxes, ROI, overlays, imshow
│
├── config.yaml              ← All configurable parameters (see Section 7)
├── requirements.txt         ← Pinned dependency versions
├── README.md                ← Setup instructions, usage, known limitations
│
├── alerts/                  ← Auto-created at runtime; stores alert PNG files
│   └── .gitkeep
│
├── logs/
│   └── event_log.csv        ← Timestamped intrusion event records
│
└── data/
    ├── test_videos/         ← VIRAT clips, UCF clips, custom recordings
    │   ├── virat_sample.mp4
    │   ├── ucf_sample.mp4
    │   └── custom_lowlight.mp4
    └── models/
        └── yolov8n.pt       ← Downloaded on first run by Ultralytics; can be pre-placed here
```

---

### File Responsibilities

**`main.py`**  
The only file you run. Reads `config.yaml` at startup. Instantiates all modules once (model loading happens here — never inside the loop). Runs the main frame loop: read → preprocess → detect → intrusion check → filter → alert → visualise. Handles the `q` key to exit and ensures `release()` is called on exit.

**`video_input.py`**  
A thin wrapper around `cv2.VideoCapture`. Accepts either an integer (webcam) or a string path (file). Raises a clear error if the source fails to open. Exposes `read_frame()` returning `(success: bool, frame: ndarray)`. Single responsibility: only frame acquisition.

**`preprocessing.py`**  
Stateless functions only — no class needed. `resize_frame()`, `bgr_to_rgb()`, `equalise_histogram()`. These are applied before detection and visualization separately (viz needs BGR; detection needs RGB).

**`detection.py`**  
A `DetectionEngine` class. `__init__` loads the model — called once. `detect(frame)` runs inference and returns a list of `(x1, y1, x2, y2, confidence)` tuples for class 0 only. No drawing, no filtering — just raw inference and class filtering.

**`intrusion.py`**  
A `ROIEngine` class. `load_rois(config)` parses the ROI list from config into a list of dicts (`{name, x1, y1, x2, y2}`). `check_intersection(box, roi)` is a pure function implementing the AABB formula. `get_intrusions(detections, rois)` returns a dict mapping each ROI name to a list of boxes intersecting it.

**`filtering.py`**  
A `FilterManager` class holding per-ROI state. `process(roi_name, intruding, confidence, current_time)` runs all three filters in sequence and returns `True` only when all three pass. Internal state: `frame_counters: dict`, `last_alert_times: dict`.

**`alert.py`**  
An `AlertSystem` class. `trigger(frame, roi_name, confidence, frame_num)` orchestrates all three alert outputs. `save_alert_image()` uses `cv2.imwrite`. `write_log_entry()` opens the CSV in append mode. `print_console_alert()` prints a formatted line. The `alerts/` and `logs/` directories are created at `__init__` if they don't exist.

**`visualization.py`**  
Stateless drawing functions operating on a copy of the frame. `draw_bboxes(frame, detections, intruding_ids)` draws green or red boxes. `draw_rois(frame, rois)` draws the ROI rectangle(s). `draw_hud(frame, fps, status)` overlays text. `show(frame)` calls `cv2.imshow()` and returns whether `q` was pressed.

**`config.yaml`**  
The single source of truth for all tunable parameters. No hardcoded thresholds anywhere in the code — every threshold reads from this file at startup. See Section 7 for full content.

---

## 4. DATA STRATEGY INTEGRATION

> No training, no annotation. All data is used purely to test and tune the running system.

---

### 4.1 Pretrained Model — Download & Placement

```bash
# Ultralytics downloads the model automatically on first run.
# To pre-place it manually:
mkdir -p data/models
# Download from: https://github.com/ultralytics/assets/releases
# Place yolov8n.pt in data/models/
```

In `config.yaml`, set:
```yaml
model:
  path: "data/models/yolov8n.pt"   # or just "yolov8n.pt" for auto-download
  variant: "yolov8n"
```

---

### 4.2 Data Folder Organisation

```
data/
├── models/
│   └── yolov8n.pt
└── test_videos/
    ├── scenario_01_normal_light/
    │   ├── single_person_entry.mp4       # Baseline: one person enters ROI
    │   └── multi_person.mp4              # Multiple people, mixed ROI/non-ROI
    ├── scenario_02_edge_cases/
    │   ├── fast_movement.mp4             # Person running through ROI
    │   ├── partial_occlusion.mp4         # Person partially hidden
    │   └── background_motion.mp4         # Fan, curtains — no person
    └── scenario_03_low_light/
        ├── lowlight_no_equalise.mp4
        └── lowlight_with_equalise.mp4
```

---

### 4.3 Datasets — Where to Get Them

| Dataset | Source | What to Download |
|---|---|---|
| **VIRAT** | viratdata.org | Any outdoor parking/pathway clip (Ground Camera data) |
| **UCF Crime** | UCF (Kaggle mirrors available) | "Trespassing" or "Loitering" clips |
| **Custom** | Your webcam | Record in-room scenarios per the table below |

**Recommended custom recording scenarios:**

| Filename | What to record | Why it matters |
|---|---|---|
| `single_entry_normal.mp4` | Walk into ROI zone, stand, walk out | Baseline validation |
| `multi_person.mp4` | Two people, one in ROI, one outside | Tests per-box independence |
| `fast_walk.mp4` | Walk quickly through the zone | Tests temporal filter isn't too slow |
| `background_motion.mp4` | Empty room, fan or curtain moving | Must produce zero alerts |
| `lowlight_entry.mp4` | Same as single_entry but lights dimmed | Test histogram equalisation benefit |
| `partial_hide.mp4` | Enter zone while partially behind furniture | Edge case: partial bounding box |

---

### 4.4 How to Run Tests

```bash
# Run against a specific video file:
python main.py --source data/test_videos/scenario_01_normal_light/single_person_entry.mp4

# Run with webcam:
python main.py --source 0

# Run with histogram equalisation enabled:
python main.py --source data/test_videos/scenario_03_low_light/lowlight_no_equalise.mp4 --equalise
```

---

### 4.5 Iteration Loop for Tuning

1. Run the system against `background_motion.mp4` — count false alerts. Target: **0 alerts**.
2. If FP count > 0, raise `confidence_threshold` by 0.05. Re-run. Repeat.
3. Run against `single_entry_normal.mp4` — confirm alert fires within ~1 second of ROI entry.
4. If alert is too slow, lower `frame_threshold` from 3 → 2. Re-run.
5. Document final threshold values in the evaluation report.

---

## 5. CORE LOGIC FLOW

> This is the exact order of operations inside `main.py`'s frame loop. Follow it precisely.

```
START
  │
  ▼
[STARTUP — once only]
  Load config.yaml
  Instantiate VideoInput(source)
  Instantiate DetectionEngine(model_path)
  Instantiate ROIEngine(config.rois)
  Instantiate FilterManager(config.filters)
  Instantiate AlertSystem(config.alerts)
  │
  ▼
╔══════════════════════════════╗
║       MAIN FRAME LOOP        ║
╠══════════════════════════════╣
║                              ║
║  success, frame = read_frame()
║  if not success → BREAK      ║
║         │                    ║
║         ▼                    ║
║  [PREPROCESSING]             ║
║  rgb_frame = bgr_to_rgb(     ║
║    resize_frame(frame))      ║
║         │                    ║
║         ▼                    ║
║  [DETECTION ENGINE]          ║
║  detections = detect(rgb)    ║
║  → list of (x1,y1,x2,y2,conf)║
║         │                    ║
║         ▼                    ║
║  [ROI INTERSECTION]          ║
║  intrusions = get_intrusions(║
║    detections, rois)         ║
║  → dict {roi_name: [boxes]}  ║
║         │                    ║
║         ▼                    ║
║  [FILTER LAYER]              ║
║  for each roi in intrusions: ║
║    confirmed = filter_manager║
║      .process(roi, boxes,    ║
║               conf, time)    ║
║         │                    ║
║         ▼                    ║
║  [ALERT SYSTEM]              ║
║  if confirmed:               ║
║    alert.trigger(frame, roi) ║
║         │                    ║
║         ▼                    ║
║  [VISUALISATION]             ║
║  annotated = draw_all(frame, ║
║    detections, rois,         ║
║    intrusions, fps)          ║
║  show(annotated)             ║
║         │                    ║
║  if 'q' pressed → BREAK      ║
╚══════════════════════════════╝
  │
  ▼
[CLEANUP]
  video_input.release()
  cv2.destroyAllWindows()
  print("Session ended.")
  │
  ▼
END
```

**How data flows between modules:**

- `VideoInput` → raw BGR frame → `Preprocessing` → RGB resized frame → `DetectionEngine`
- `DetectionEngine` → list of `(x1, y1, x2, y2, conf)` → `ROIEngine`
- `ROIEngine` → dict of `{roi_name: [intruding_boxes]}` → `FilterManager`
- `FilterManager` → `confirmed: bool` → `AlertSystem` (only on `True`)
- `AlertSystem` → writes PNG + CSV row (side effects only; returns nothing)
- All of `{frame, detections, intrusions}` → `Visualization` → annotated frame on screen

Note: `Visualization` reads from the original BGR frame (not the RGB copy passed to the detector). Always keep both references in the loop.

---

## 6. KEY ALGORITHMS

> These are the three core algorithms that make COGNIVIEW work. Implement them exactly as described.

---

### 6.1 ROI Intersection Logic (AABB Overlap)

**When to call:** Once per detected person, once per defined ROI.

**Inputs:**
- Person bounding box: `(px1, py1, px2, py2)` — pixel coordinates of top-left and bottom-right corners
- ROI rectangle: `(rx1, ry1, rx2, ry2)` — pixel coordinates of the restricted zone

**The formula:**
```python
def check_intersection(px1, py1, px2, py2, rx1, ry1, rx2, ry2) -> bool:
    """
    Returns True if the person box overlaps with the ROI rectangle.
    AABB overlap: two rectangles do NOT overlap only when one is
    completely to the left, right, above, or below the other.
    We invert that condition.
    """
    no_overlap = (px2 < rx1) or (px1 > rx2) or (py2 < ry1) or (py1 > ry2)
    return not no_overlap
```

**Why this works:** If none of the four "completely outside" conditions are true, the boxes must overlap. No floating point, no external libraries — just four comparisons.

**Centre-point variant (optional, stricter):**
```python
def check_centre_inside(px1, py1, px2, py2, rx1, ry1, rx2, ry2) -> bool:
    """
    Only triggers if the person's centre point is inside the ROI.
    More conservative — requires person to be 'mostly' inside the zone.
    """
    cx = (px1 + px2) / 2
    cy = (py1 + py2) / 2
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2
```

Start with `check_intersection`. Switch to `check_centre_inside` if you get too many false positives from people standing near the boundary.

---

### 6.2 Temporal Filtering Logic

**When to call:** After ROI intersection is confirmed, before triggering any alert.

**State:** `intrusion_counters: dict[str, int]` — one counter per ROI name, initialised to 0.

```python
class TemporalFilter:
    def __init__(self, frame_threshold: int):
        self.threshold = frame_threshold     # e.g. 3
        self.counters = {}                   # {roi_name: int}

    def update(self, roi_name: str, intruding: bool) -> bool:
        """
        Returns True only when the counter reaches the threshold.
        Call this once per frame, per ROI.
        """
        if roi_name not in self.counters:
            self.counters[roi_name] = 0

        if intruding:
            self.counters[roi_name] += 1
        else:
            self.counters[roi_name] = 0      # Reset on any non-intruding frame

        return self.counters[roi_name] >= self.threshold
```

**Behaviour example with `threshold=3`:**
```
Frame 1: intruding=True  → counter=1 → returns False
Frame 2: intruding=True  → counter=2 → returns False
Frame 3: intruding=True  → counter=3 → returns True  ← ALERT FIRES
Frame 4: intruding=True  → counter=4 → returns True  (cooldown will suppress)
Frame 5: intruding=False → counter=0 → returns False (reset)
Frame 6: intruding=True  → counter=1 → returns False (must reach 3 again)
```

---

### 6.3 Confidence Filtering

**When to call:** Before the temporal filter — if confidence is too low, don't even increment the counter.

```python
def passes_confidence(confidence: float, threshold: float) -> bool:
    """
    Returns True only if the detection score meets the minimum threshold.
    Default threshold: 0.5 (configurable in config.yaml)
    """
    return confidence >= threshold
```

**Cooldown filter — preventing alert storms:**
```python
import time

class CooldownFilter:
    def __init__(self, cooldown_seconds: float):
        self.cooldown = cooldown_seconds     # e.g. 10.0
        self.last_alert_times = {}           # {roi_name: float (unix timestamp)}

    def can_alert(self, roi_name: str) -> bool:
        """
        Returns True if enough time has passed since the last alert for this ROI.
        """
        now = time.time()
        last = self.last_alert_times.get(roi_name, 0)
        return (now - last) >= self.cooldown

    def record_alert(self, roi_name: str):
        """Call this immediately after an alert fires."""
        self.last_alert_times[roi_name] = time.time()
```

**Complete filter chain (combine all three):**
```python
def process_candidate(roi_name, confidence, intruding, conf_filter, temporal_filter, cooldown_filter):
    if not conf_filter.passes_confidence(confidence):
        temporal_filter.update(roi_name, False)   # Don't count low-confidence frames
        return False

    sustained = temporal_filter.update(roi_name, intruding)

    if sustained and cooldown_filter.can_alert(roi_name):
        cooldown_filter.record_alert(roi_name)
        return True

    return False
```

---

## 7. CONFIGURATION DESIGN

> All tunable values live in `config.yaml`. No magic numbers in code. If you need to change a threshold, change the config — not the source.

---

### Complete `config.yaml`

```yaml
# ─────────────────────────────────────────────
# COGNIVIEW Configuration File
# Edit this file to tune system behaviour.
# Do NOT hardcode these values in Python files.
# ─────────────────────────────────────────────

# Video source: 0 for webcam, or file path string
source: 0
# source: "data/test_videos/scenario_01/single_entry.mp4"

# Display resolution (width x height, pixels)
resolution:
  width: 640
  height: 480

# Enable histogram equalisation for low-light scenes
preprocessing:
  equalise_histogram: false

# YOLOv8 model settings
model:
  path: "yolov8n.pt"       # Downloads automatically if not found
  device: "cpu"             # "cpu" or "0" for GPU

# Filtering thresholds
filters:
  confidence_threshold: 0.50    # Drop detections below this score (0.0–1.0)
  frame_threshold: 3            # Frames of consecutive detection before alert fires
  cooldown_seconds: 10.0        # Minimum seconds between alerts per ROI

# Regions of Interest (one or more)
# Coordinates are pixel values at the configured resolution
rois:
  - name: "Zone_A"
    x1: 150
    y1: 100
    x2: 490
    y2: 380
  # - name: "Zone_B"          # Uncomment to add a second ROI
  #   x1: 50
  #   y1: 50
  #   x2: 200
  #   y2: 200

# Alert output settings
alerts:
  output_dir: "alerts/"
  log_file: "logs/event_log.csv"
  console_print: true
```

---

### Why Externalise Configuration?

**Testability:** You can switch from webcam to a video file without touching any Python code. Run `python main.py` on Monday with the webcam, on Tuesday with a VIRAT clip — same code, different config.

**Tuning speed:** Iterating on `confidence_threshold` or `frame_threshold` is a 10-second text edit, not a code change with potential for bugs.

**Reproducibility:** Pin the exact config used for each test run by copying `config.yaml` into your test log folder. Anyone can reproduce your exact test results.

**Multiple environments:** Use `config_webcam.yaml` and `config_virat.yaml` for different scenarios. Pass the config path as a CLI argument.

**The rule:** If a value appears more than once in your code, it belongs in `config.yaml`. If a value might ever need to change during testing, it belongs in `config.yaml`.

---

## 8. TESTING STRATEGY

---

### 8.1 Test Scenarios

#### SCENARIO A — Normal Detection (No Intrusion)
**Setup:** Run `background_motion.mp4` (fan, curtains, no person present) for 5 minutes.  
**Pass condition:** Zero alerts fire. Zero PNG files created. Event log stays empty.  
**What it validates:** Confidence filter is working; motion alone doesn't trigger alerts.

#### SCENARIO B — Baseline Intrusion
**Setup:** Run `single_person_entry.mp4` — one person walks into the defined ROI zone.  
**Pass condition:** Alert fires within 3 frames of person entering zone. One PNG saved. One CSV row written.  
**What it validates:** End-to-end pipeline works. Temporal filter fires at correct threshold.

#### SCENARIO C — Temporal Filter Validation
**Setup:** Walk partially into the ROI zone for 1–2 frames then step back. Repeat 5 times.  
**Pass condition:** No alert fires for any of the partial entries.  
**What it validates:** Temporal filter correctly requires N consecutive frames and resets on exit.

#### SCENARIO D — Cooldown Filter Validation
**Setup:** Stand inside the ROI zone for 30 seconds continuously.  
**Pass condition:** Only one alert fires in the first ~0.3 seconds. No additional alerts during the 30-second stand.  
**What it validates:** Cooldown filter prevents alert storm from sustained presence.

#### SCENARIO E — Multiple Persons
**Setup:** Two people in frame simultaneously — one inside ROI, one outside.  
**Pass condition:** Alert fires for the person inside the ROI. Person outside generates green box only.  
**What it validates:** Per-box independence; system doesn't alert based on any person in frame.

#### SCENARIO F — Low-Light Condition
**Setup:** Run `lowlight_entry.mp4` first with `equalise_histogram: false`, then with `true`.  
**Pass condition:** Detection recall visibly improves with equalisation on.  
**What it validates:** Preprocessing enhancement works and is configurable.

#### SCENARIO G — Edge Cases
**Setup:** Run UCF Crime clip or `fast_walk.mp4` (person running through ROI).  
**Observe:** Does the alert fire? Does the system crash? Does FPS degrade?  
**What it validates:** System robustness under non-ideal conditions. Document results even if imperfect.

---

### 8.2 How to Measure False Positives

```
False Positive Rate per session =
    Number of alerts where no person was in ROI
    ─────────────────────────────────────────────
    Total number of alerts fired

Target: < 2 false alerts per 10-minute session
```

**Method:**
1. Run the system for 10 minutes on a scenario video.
2. Open the `/alerts/` folder.
3. Manually inspect each saved PNG image.
4. Count images where no person appears inside the ROI box — these are false positives.
5. Record the ratio. If > 2 FPs per 10 minutes, raise `confidence_threshold` by 0.05 and repeat.

---

### 8.3 How to Measure Alert Accuracy

```
Alert Precision =
    Alerts where a real person IS in the ROI
    ─────────────────────────────────────────
    Total alerts fired

Target: ≥ 90% precision
```

This is measured the same way as FP — by manual inspection of saved alert images. The PNG files are your ground truth.

---

### 8.4 Performance Measurement

```python
# Add this to main.py frame loop:
import cv2

prev_tick = cv2.getTickCount()

# ... [frame processing happens here] ...

curr_tick = cv2.getTickCount()
elapsed_ms = (curr_tick - prev_tick) / cv2.getTickFrequency() * 1000
fps = 1000 / elapsed_ms

prev_tick = curr_tick
```

**Target:** `elapsed_ms < 100` → FPS ≥ 10 on CPU with YOLOv8n.

If FPS drops below 10:
1. Reduce resolution to 416×416 in config.
2. Ensure model is only loaded once (not per frame).
3. Profile: use `print(elapsed_ms)` to identify which module is slowest.

---

## 9. DELIVERABLES MAPPING

> Everything you build maps to a concrete, shareable output. This table is your definition of done.

| Blueprint Module | PRD Deliverable | File/Location | Done When |
|---|---|---|---|
| Full system (Phases 1–4) | **Working Python Prototype** | `cogniview/` directory | All 5 phases pass their acceptance tests |
| All source files | **GitHub Repository** | `github.com/yourname/cogniview` | README, requirements.txt, `.gitignore`, and all `.py` files pushed |
| `config.yaml` | **Configuration File** | `cogniview/config.yaml` | All parameters documented with comments; no hardcoded values in code |
| Phase 5 test runs | **Sample Test Videos** | `data/test_videos/` | At least 3 scenarios: normal detection, confirmed intrusion, false-positive stress test |
| Alert System output | **Alert Image Samples** | `alerts/alert_*.png` | At least 5 sample PNG files from real test runs committed to repo |
| Alert System output | **Event Log Sample** | `logs/event_log.csv` | Sample CSV with ≥ 10 rows showing real test run data |
| Phase 5 documentation | **Evaluation Report** | `EVALUATION.md` or `report.pdf` | Covers: test scenarios run, FP rate, alert precision, latency, known limitations |
| Optional recording | **Demo Video** | `demo.mp4` | Screen recording of live system running — attach to GitHub Releases |

---

### Final Checklist Before Submission

```
[ ] python main.py runs from a fresh virtual environment in < 5 minutes setup
[ ] All config values are in config.yaml — no hardcoded numbers in .py files
[ ] YOLOv8n downloads automatically on first run (or is bundled)
[ ] Webcam mode and video file mode both work
[ ] /alerts/ directory auto-created on first alert
[ ] event_log.csv auto-created on first alert
[ ] 'q' key exits cleanly — no hanging processes
[ ] System runs stably for 10+ minutes without crashing or memory growth
[ ] README explains: install, run, config, expected outputs
[ ] requirements.txt has pinned versions (pip freeze > requirements.txt)
[ ] .gitignore excludes: /alerts/*.png, /logs/*.csv, /data/test_videos/, *.pt
[ ] Evaluation report documents at least 3 scenarios with observed results
```

---

*COGNIVIEW Development Blueprint · v1.0 · Generated from PRD v1.0*  
*This document is the authoritative execution guide for the prototype build.*
