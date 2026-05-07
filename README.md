# COGNIVIEW
**Real-Time AI-Powered Surveillance System**

COGNIVIEW is an intelligent, real-time video surveillance prototype built with Python, OpenCV, and Ultralytics YOLOv8. It identifies persons in live webcam feeds or recorded videos, flags when they enter user-defined restricted zones (ROIs), and filters false alarms using temporal, confidence, and cooldown metrics. Confirmed intrusions are automatically logged and saved as timestamped, annotated images.

---

## 🚀 Features
- **Real-Time Detection:** Uses YOLOv8n (nano) for high-speed person detection.
- **ROI Intrusion Logic:** Define restricted zones (Regions of Interest) where people shouldn't be.
- **Smart Filtering:** Mitigates false positives with confidence scoring, continuous frame tracking (temporal filtering), and alert cooldown windows.
- **Automated Logging:** Saves annotated PNG frames and detailed CSV logs upon intrusion detection.
- **Live Output:** Displays a real-time HUD with FPS, object counts, bounding boxes, and alert status.

## 🛠️ Installation

**Requirements:** Python 3.9+

1. **Clone or Extract the Project:**
   Ensure you are in the `cogniview` directory.

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will download `ultralytics` (and PyTorch) which may take several minutes depending on your connection.*

## 🎥 Usage

Run the main script to start the system:
```bash
python main.py
```

**Custom Arguments:**
- `--source`: Specify the video source (e.g., `0` for webcam, or `"data/test_videos/video.mp4"` for a file).
- `--config`: Specify a custom configuration file (default is `config.yaml`).
- `--equalise`: Enable histogram equalisation for low-light enhancement.

**Example:**
```bash
python main.py --source 0
```

*Press `q` at any time to exit the live window safely.*

## ⚙️ Configuration

Tune the system via `config.yaml`. No hardcoded thresholds are present in the Python code.

- **`source`**: The input video feed (integer or file path).
- **`resolution`**: The required video processing resolution.
- **`preprocessing.equalise_histogram`**: Contrast enhancement.
- **`model.path`**: Path to the YOLOv8 weights (auto-downloads if missing).
- **`filters`**: Tune detection rules here:
  - `confidence_threshold`: Minimum confidence score (0.0 - 1.0).
  - `frame_threshold`: How many consecutive frames an object must stay in an ROI to trigger an alert.
  - `cooldown_seconds`: Minimum time between alerts.
- **`rois`**: Define restricted zones via pixel coordinates (`x1, y1, x2, y2`).

## 📂 Project Structure

- `main.py`: The orchestrator script running the core pipeline loop.
- `video_input.py`: Manages video sources via OpenCV.
- `preprocessing.py`: Handles resizing, color conversion, and image enhancement.
- `detection.py`: Manages the YOLOv8 model and logic.
- `intrusion.py`: Computes Region of Interest (ROI) logic and bounding box overlaps.
- `filtering.py`: Applies temporal, cooldown, and confidence filtering.
- `alert.py`: Handles saving images and writing log records.
- `visualization.py`: Renders bounding boxes, ROIs, and the HUD to the live window.

## 🧪 Testing

We recommend testing the system against Kaggle's VIRAT or UCF Crime datasets. To avoid massive repo sizes, datasets are omitted. Place your `.mp4` test files in the `data/test_videos/` directory and update the `source` in `config.yaml` to run them.

---
*Created as a prototype development project.*
