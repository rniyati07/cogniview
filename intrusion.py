"""
COGNIVIEW — Module 4: Intrusion Logic (ROI Engine)
Determines whether detected persons overlap with defined Restricted Zones.
"""

class ROIEngine:
    def __init__(self, rois_config):
        """
        Initialize the ROIEngine with a list of ROIs.
        
        Args:
            rois_config: List of dicts, e.g. [{"name": "Zone_A", "x1": 150, "y1": 100, "x2": 490, "y2": 380}]
        """
        self.rois = rois_config if rois_config else []
        print(f"[ROIEngine] Loaded {len(self.rois)} ROIs.")

    def check_intersection(self, px1, py1, px2, py2, rx1, ry1, rx2, ry2):
        """
        Returns True if the person box overlaps with the ROI rectangle.
        AABB overlap formula.
        """
        no_overlap = (px2 < rx1) or (px1 > rx2) or (py2 < ry1) or (py1 > ry2)
        return not no_overlap

    def get_intrusions(self, detections):
        """
        Check all detections against all ROIs.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence)
            
        Returns:
            dict: {roi_name: [list of intruding detection indices]}
        """
        intrusions = {roi["name"]: [] for roi in self.rois}
        
        for idx, (px1, py1, px2, py2, conf) in enumerate(detections):
            for roi in self.rois:
                if self.check_intersection(px1, py1, px2, py2, roi["x1"], roi["y1"], roi["x2"], roi["y2"]):
                    intrusions[roi["name"]].append(idx)
                    
        return intrusions
