"""
COGNIVIEW — Module 1: Video Input
Wraps OpenCV VideoCapture to provide a clean frame acquisition interface.
Supports both webcam (integer index) and video file (string path) sources.
"""

import cv2


class VideoInput:
    """Thin wrapper around cv2.VideoCapture with error handling."""

    def __init__(self, source=0):
        """
        Initialize video source.

        Args:
            source: Webcam index (int) or video file path (str).
        """
        self.source = source
        self.cap = None

    def open_source(self):
        """Open the video source. Raises RuntimeError on failure."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source: {self.source}. "
                "Check that your webcam is connected or the file path is correct."
            )
        print(f"[VideoInput] Source opened: {self.source}")

    def read_frame(self):
        """
        Read a single frame from the source.

        Returns:
            tuple: (success: bool, frame: ndarray or None)
        """
        if self.cap is None:
            return False, None
        success, frame = self.cap.read()
        return success, frame

    def get_fps(self):
        """Return the FPS of the video source (useful for file playback)."""
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_size(self):
        """Return (width, height) of the video source."""
        if self.cap is None:
            return 0, 0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        """Release the video source."""
        if self.cap is not None:
            self.cap.release()
            print("[VideoInput] Source released.")
