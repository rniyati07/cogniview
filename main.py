"""
COGNIVIEW — Main Orchestrator
Wires all modules together and runs the real-time detection loop.

Usage:
    python main.py                          # Uses config.yaml defaults (webcam)
    python main.py --source 0               # Webcam explicitly
    python main.py --source path/to/video   # Video file
    python main.py --config custom.yaml     # Custom config file
"""

import argparse
import sys
import time

import cv2
import yaml

from video_input import VideoInput
from preprocessing import preprocess_frame
from detection import DetectionEngine
from intrusion import ROIEngine
from filtering import FilterManager
from alert import AlertSystem
from visualization import draw_detections, draw_hud, show_frame, draw_rois


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"[Main] Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"[Main] WARNING: Config file '{config_path}' not found. Using defaults.")
        return {
            "source": 0,
            "resolution": {"width": 640, "height": 480},
            "preprocessing": {"equalise_histogram": False},
            "model": {"path": "yolov8n.pt", "device": "cpu"},
        }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="COGNIVIEW — Real-Time AI-Powered Surveillance System"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source: webcam index (int) or file path (str). Overrides config."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--equalise",
        action="store_true",
        help="Enable histogram equalisation (overrides config)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying the video window (useful for benchmarks/CI)."
    )
    return parser.parse_args()


def run():
    """Main entry point: initialize modules and run the detection loop."""

    # ── Parse arguments and load config ──
    args = parse_args()
    config = load_config(args.config)

    # Determine video source (CLI overrides config)
    source = config.get("source", 0)
    if args.source is not None:
        # Try to interpret as integer (webcam index)
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    # Resolution settings
    res = config.get("resolution", {})
    frame_width = res.get("width", 640)
    frame_height = res.get("height", 480)

    # Preprocessing settings
    preproc = config.get("preprocessing", {})
    equalise = preproc.get("equalise_histogram", False)
    if args.equalise:
        equalise = True

    # Model settings
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("path", "yolov8n.pt")
    device = model_cfg.get("device", "cpu")

    # ── Initialize modules (ONCE — outside the loop) ──
    print("=" * 55)
    print("  COGNIVIEW — Real-Time AI-Powered Surveillance System")
    print("=" * 55)
    print(f"  Source      : {source}")
    print(f"  Resolution  : {frame_width}x{frame_height}")
    print(f"  Equalise    : {equalise}")
    print(f"  Model       : {model_path}")
    print(f"  Device      : {device}")
    print("=" * 55)

    # Module 1: Video Input
    video = VideoInput(source)
    try:
        video.open_source()
    except RuntimeError as e:
        print(f"[Main] ERROR: {e}")
        sys.exit(1)

    # Module 3: Detection Engine (loaded ONCE)
    detector = DetectionEngine(model_path=model_path, device=device)

    # Module 4: ROI Engine
    rois = config.get("rois", [])
    roi_engine = ROIEngine(rois)

    # Module 5: Filter Manager
    filter_manager = FilterManager(config)

    # Module 6: Alert System
    alert_system = AlertSystem(config)

    # ── FPS tracking ──
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    print("\n[Main] Starting detection loop. Press 'q' to quit.\n")

    # ── Main frame loop ──
    try:
        while True:
            # Step 1: Read frame
            success, raw_frame = video.read_frame()
            if not success:
                print("[Main] End of video stream or read failure.")
                break

            # Step 2: Preprocess
            bgr_frame, rgb_frame = preprocess_frame(
                raw_frame,
                width=frame_width,
                height=frame_height,
                equalise=equalise
            )

            # Step 3: Detect persons
            detections = detector.detect(rgb_frame)

            # Step 3.5: Check intrusions
            intrusions = roi_engine.get_intrusions(detections)
            
            # Gather all intruding detection IDs and ROIs
            intruding_ids = set()
            intruding_rois = set()
            is_intruding = False
            for r_name, box_indices in intrusions.items():
                    
                # Process through filters
                sustained, should_fire_alert = filter_manager.process(r_name, box_indices, detections)
                if sustained:
                    is_intruding = True
                    intruding_ids.update(box_indices)
                    intruding_rois.add(r_name)
                    
                if should_fire_alert:
                    # Find highest confidence among intruding boxes for this ROI
                    highest_conf = max([detections[i][4] for i in box_indices]) if box_indices else 0.0
                    
                    alert_frame = bgr_frame.copy()
                    alert_frame = draw_rois(alert_frame, rois, intruding_rois)
                    # We pass the currently known intruding_ids up to this point for the snapshot
                    alert_frame = draw_detections(alert_frame, detections, intruding_ids)
                    
                    alert_system.trigger(alert_frame, r_name, highest_conf, frame_count)

            # Step 4: Calculate FPS
            current_time = time.time()
            elapsed = current_time - prev_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            prev_time = current_time
            frame_count += 1

            # Step 5: Visualize (draw on a copy of BGR frame)
            display_frame = bgr_frame.copy()
            display_frame = draw_rois(display_frame, rois, intruding_rois)
            display_frame = draw_detections(display_frame, detections, intruding_ids)
            
            status_text = "⚠ INTRUSION DETECTED" if is_intruding else "CLEAR"
            display_frame = draw_hud(
                display_frame,
                fps=fps,
                status=status_text,
                detection_count=len(detections)
            )

            # Step 6: Show frame and check for quit
            if not args.headless:
                quit_pressed = show_frame(display_frame)
                if quit_pressed:
                    print("[Main] Quit signal received.")
                    break
            
            if frame_count % 100 == 0:
                print(f"[Main] Processed {frame_count} frames... (FPS: {fps:.1f})")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    # ── Cleanup ──
    video.release()
    cv2.destroyAllWindows()
    print(f"[Main] Session ended. Total frames processed: {frame_count}")


if __name__ == "__main__":
    run()
