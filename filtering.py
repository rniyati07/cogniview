"""
COGNIVIEW — Module 5: Filtering Layer
Eliminates false alarms by requiring intrusions to be high-confidence
and temporally sustained before an alert fires.
"""

import time


class TemporalFilter:
    def __init__(self, frame_threshold: int):
        self.threshold = frame_threshold
        self.counters = {}

    def update(self, roi_name: str, intruding: bool) -> bool:
        """
        Returns True only when the counter reaches the threshold.
        """
        if roi_name not in self.counters:
            self.counters[roi_name] = 0

        if intruding:
            self.counters[roi_name] += 1
        else:
            self.counters[roi_name] = 0

        return self.counters[roi_name] >= self.threshold


class CooldownFilter:
    def __init__(self, cooldown_seconds: float):
        self.cooldown = cooldown_seconds
        self.last_alert_times = {}

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


class FilterManager:
    def __init__(self, config):
        """
        Initialize the FilterManager with config.
        """
        filters_cfg = config.get("filters", {})
        self.conf_threshold = filters_cfg.get("confidence_threshold", 0.5)
        self.temporal = TemporalFilter(filters_cfg.get("frame_threshold", 3))
        self.cooldown = CooldownFilter(filters_cfg.get("cooldown_seconds", 10.0))

    def process(self, roi_name: str, intruding_boxes: list, detections: list) -> bool:
        """
        Process a candidate intrusion through all filters.
        
        Args:
            roi_name: Name of the ROI.
            intruding_boxes: List of indices in detections that intersect this ROI.
            detections: The full list of detections (x1, y1, x2, y2, conf).
            
        Returns:
            tuple: (sustained: bool, should_fire_alert: bool)
                - sustained: True if the intrusion has met confidence and temporal thresholds (drives red box visual).
                - should_fire_alert: True if an alert should be logged/saved (drives the AlertSystem).
        """
        # 1. Confidence Filter
        highest_conf = 0.0
        for idx in intruding_boxes:
            conf = detections[idx][4]
            if conf > highest_conf:
                highest_conf = conf

        is_confident = highest_conf >= self.conf_threshold and len(intruding_boxes) > 0

        # 2. Temporal Filter
        sustained = self.temporal.update(roi_name, is_confident)

        # 3. Cooldown Filter
        should_fire_alert = False
        if sustained and self.cooldown.can_alert(roi_name):
            self.cooldown.record_alert(roi_name)
            should_fire_alert = True

        return sustained, should_fire_alert
