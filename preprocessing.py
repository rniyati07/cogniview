"""
COGNIVIEW — Module 2: Preprocessing
Stateless functions to normalize raw frames before detection.
Handles resize, colour conversion, and optional histogram equalisation.
"""

import cv2
import numpy as np


def resize_frame(frame, width=640, height=480):
    """
    Resize frame to the target resolution.

    Args:
        frame: Input BGR frame (numpy array).
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        Resized frame as numpy array.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(frame):
    """
    Convert BGR frame (OpenCV default) to RGB (YOLO expected input).

    Args:
        frame: Input BGR frame.

    Returns:
        RGB frame as numpy array.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def equalise_histogram(frame):
    """
    Apply CLAHE histogram equalisation to improve contrast in low-light frames.
    Operates on the luminance channel in LAB colour space to avoid colour distortion.

    Args:
        frame: Input BGR frame.

    Returns:
        Contrast-enhanced BGR frame.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def preprocess_frame(frame, width=640, height=480, equalise=False):
    """
    Full preprocessing pipeline: resize → optional equalisation → BGR to RGB.

    Args:
        frame: Raw BGR frame from VideoInput.
        width: Target width.
        height: Target height.
        equalise: Whether to apply histogram equalisation.

    Returns:
        tuple: (bgr_resized, rgb_frame)
            - bgr_resized: Resized BGR frame for visualization.
            - rgb_frame: Preprocessed RGB frame for detection.
    """
    resized = resize_frame(frame, width, height)

    if equalise:
        resized = equalise_histogram(resized)

    rgb = bgr_to_rgb(resized)
    return resized, rgb
