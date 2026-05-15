"""
COGNIVIEW — Module 7: Visualization Layer
Stateless drawing functions for annotating frames with detections and HUD overlays.
All drawing is performed on a COPY of the frame — the original is never modified.
"""

import cv2


# Colour constants (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)

# Drawing parameters
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LABEL = 0.5
FONT_SCALE_HUD = 0.6
FONT_THICKNESS = 1


def draw_detections(frame, detections, intruding_ids=None):
    """
    Draw bounding boxes and confidence labels for detected persons.

    Args:
        frame: BGR frame to draw on (will be modified in-place — pass a copy).
        detections: List of (x1, y1, x2, y2, confidence) tuples.
        intruding_ids: Set of detection indices that are intruding (drawn in red).
                       If None, all boxes are drawn in green (Phase 1 behaviour).

    Returns:
        Annotated frame.
    """
    if intruding_ids is None:
        intruding_ids = set()

    for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
        # Choose colour based on intrusion status
        color = COLOR_RED if idx in intruding_ids else COLOR_GREEN

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw confidence label with background
        label = f"Person {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, FONT, FONT_SCALE_LABEL, FONT_THICKNESS)
        label_w, label_h = label_size

        # Label background rectangle
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 8),
            (x1 + label_w + 4, y1),
            color,
            cv2.FILLED
        )
        # Label text
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - 4),
            FONT, FONT_SCALE_LABEL, COLOR_WHITE, FONT_THICKNESS
        )

    return frame


def draw_rois(frame, rois, intruding_rois=None):
    """
    Draw ROI boundaries on the frame with transparent fills.
    
    Args:
        frame: BGR frame to draw on.
        rois: List of ROI configurations (dicts).
        intruding_rois: Set of ROI names currently experiencing intrusion.
        
    Returns:
        Annotated frame.
    """
    if intruding_rois is None:
        intruding_rois = set()

    overlay = frame.copy()

    for roi in rois:
        name = roi["name"]
        color = COLOR_RED if name in intruding_rois else COLOR_YELLOW
        
        x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
        
        # Draw transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
        
        # Draw solid boundary
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        
        # Draw ROI name
        cv2.putText(
            frame, name,
            (x1 + 5, y1 + 20),
            FONT, FONT_SCALE_HUD, color, FONT_THICKNESS
        )
        
    # Apply alpha blending for the fill
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
    return frame


def draw_hud(frame, fps, status="CLEAR", detection_count=0):
    """
    Overlay FPS counter, status text, and detection count on the frame.

    Args:
        frame: BGR frame to draw on.
        fps: Current frames per second.
        status: Status string ("CLEAR" or "⚠ INTRUSION DETECTED").
        detection_count: Number of persons detected in current frame.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # Semi-transparent HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), COLOR_BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS counter (top-left)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 25), FONT, FONT_SCALE_HUD, COLOR_GREEN, FONT_THICKNESS)

    # Detection count (centre)
    count_text = f"Persons: {detection_count}"
    cv2.putText(frame, count_text, (w // 2 - 50, 25), FONT, FONT_SCALE_HUD, COLOR_WHITE, FONT_THICKNESS)

    # Status (top-right)
    status_color = COLOR_GREEN if status == "CLEAR" else COLOR_RED
    status_size, _ = cv2.getTextSize(status, FONT, FONT_SCALE_HUD, FONT_THICKNESS)
    cv2.putText(
        frame, status,
        (w - status_size[0] - 10, 25),
        FONT, FONT_SCALE_HUD, status_color, FONT_THICKNESS
    )

    return frame


def show_frame(frame, window_name="COGNIVIEW"):
    """
    Display the annotated frame in an OpenCV window.

    Args:
        frame: Annotated BGR frame to display.
        window_name: Name for the display window.

    Returns:
        True if 'q' was pressed (signal to quit), False otherwise.
    """
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q')
