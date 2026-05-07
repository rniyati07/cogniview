"""
COGNIVIEW — Module 3: Detection Engine
Loads YOLOv8 once at startup and runs per-frame inference.
Filters results to person class (class_id = 0) only.
"""

from ultralytics import YOLO


class DetectionEngine:
    """YOLOv8 inference wrapper. Model is loaded once at init."""

    # COCO class ID for 'person'
    PERSON_CLASS_ID = 0

    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        """
        Load the YOLOv8 model.

        Args:
            model_path: Path to YOLO weights file. Downloads automatically if not found.
            device: Inference device — "cpu" or "0" for GPU.
        """
        self.device = device
        print(f"[DetectionEngine] Loading model: {model_path} on {device}...")
        self.model = YOLO(model_path)
        print("[DetectionEngine] Model loaded successfully.")

    def detect(self, rgb_frame):
        """
        Run inference on a single RGB frame and return person detections.

        Args:
            rgb_frame: Preprocessed RGB frame (numpy array).

        Returns:
            List of tuples: [(x1, y1, x2, y2, confidence), ...]
            Coordinates are pixel values in the frame's resolution.
        """
        # Run inference with verbose=False to suppress per-frame YOLO logs
        results = self.model(rgb_frame, device=self.device, verbose=False)
        detections = self._filter_persons(results)
        return detections

    def _filter_persons(self, results):
        """
        Extract person-class detections from YOLO results.

        Args:
            results: Raw YOLO inference results.

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples for persons only.
        """
        person_detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                if class_id == self.PERSON_CLASS_ID:
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    person_detections.append((
                        int(x1), int(y1), int(x2), int(y2), confidence
                    ))

        return person_detections
