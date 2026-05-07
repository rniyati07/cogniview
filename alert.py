"""
COGNIVIEW — Module 6: Alert System
Persists confirmed intrusion events as image files and structured log entries.
"""

import os
import csv
from datetime import datetime, timezone
import cv2

class AlertSystem:
    def __init__(self, config):
        """
        Initialize the AlertSystem with config.
        Creates output directories and log file if they don't exist.
        """
        alerts_cfg = config.get("alerts", {})
        self.output_dir = alerts_cfg.get("output_dir", "alerts/")
        self.log_file = alerts_cfg.get("log_file", "logs/event_log.csv")
        self.console_print = alerts_cfg.get("console_print", True)
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Initialize log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp (UTC)", "Frame Number", "ROI Name", "Highest Confidence", "Image Filename"])

    def trigger(self, frame, roi_name: str, highest_confidence: float, frame_num: int):
        """
        Orchestrates all three alert outputs.
        """
        timestamp = datetime.now(timezone.utc)
        time_str_iso = timestamp.isoformat()
        time_str_file = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # 1. Save annotated image
        filename = f"alert_{time_str_file}_{roi_name}.png"
        filepath = os.path.join(self.output_dir, filename)
        self.save_alert_image(frame, filepath)
        
        # 2. Write log entry
        self.write_log_entry(time_str_iso, frame_num, roi_name, highest_confidence, filename)
        
        # 3. Print to console
        if self.console_print:
            self.print_console_alert(roi_name, highest_confidence, time_str_iso)

    def save_alert_image(self, frame, filepath: str):
        """Save the annotated frame to disk."""
        cv2.imwrite(filepath, frame)

    def write_log_entry(self, timestamp: str, frame_num: int, roi_name: str, confidence: float, filename: str):
        """Append a structured row to the event log."""
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, frame_num, roi_name, f"{confidence:.4f}", filename])

    def print_console_alert(self, roi_name: str, confidence: float, timestamp: str):
        """Print a formatted alert to the console."""
        # Using ANSI colors if the terminal supports it
        color_red = "\033[91m"
        color_reset = "\033[0m"
        print(f"{color_red}[ALERT]{color_reset} Intrusion in {roi_name} at {timestamp} (Conf: {confidence:.2f})")
