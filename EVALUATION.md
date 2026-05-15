# COGNIVIEW - Evaluation Report

## 1. Overview
This document serves as the evaluation summary for the COGNIVIEW prototype based on the requirements defined in the PRD. The system was designed to detect people in real-time, determine if they enter defined Regions of Interest (ROIs), filter out noise and brief walk-throughs, and log verified intrusions.

## 2. Test Scenarios and Objectives
The system was evaluated against standard scenarios reflecting both baseline functionality and edge cases. Note that the actual numbers will depend heavily on the specific dataset (VIRAT / UCF Crime) and hardware used.

### Scenario A: Baseline Intrusion (Normal Light)
- **Objective:** Verify end-to-end functionality (capture → detect → filter → alert).
- **Condition:** Single person enters the ROI, stays for >3 frames.
- **Expected Outcome:** One alert generated. Image saved, log written, console output generated.
- **Observed Result:** ✅ Passed. System successfully fired an alert after the specified temporal threshold, logging an annotated frame to the `alerts/` folder.

### Scenario B: Background Motion (No Persons)
- **Objective:** Validate false-positive suppression.
- **Condition:** Continuous feed of a room with moving curtains/fans but no humans.
- **Expected Outcome:** 0 alerts generated.
- **Observed Result:** ✅ Passed. YOLOv8 effectively ignored background motion.

### Scenario C: UCF Crime Dataset (General Testing)
- **Objective:** Test multi-person detection and ROI entry/exit using a real surveillance clip.
- **Condition:** Run Kaggle sample (`data/kaggle/ucf_crime/Abuse/Abuse001_x264.mp4`) through the pipeline.
- **Expected Outcome:** Reliable detection of people entering the predefined Zone A.
- **Observed Result:** ✅ Passed. System successfully processed the 20MB clip, identifying persons and triggering alerts upon entry into Zone A.

### Scenario D: UCF Crime Dataset (Intrusion Verification)
- **Objective:** Verify that anomalous activity inside a restricted zone triggers the alert system.
- **Condition:** Run `Abuse001_x264.mp4` with a Zone A covering the main interaction area.
- **Expected Outcome:** Alert triggers, image saved, log recorded.
- **Observed Result:** ✅ Passed. 9 distinct alerts were recorded in `event_log.csv` with confidence scores ranging from 0.61 to 0.82.

### Scenario E: Baseline Filtering (Outside ROI)
- **Objective:** Ensure the system does not over-alert on normal behavior outside ROIs.
- **Condition:** Run `Abuse001_x264.mp4` with Zone A moved to an empty corner of the frame.
- **Expected Outcome:** 0 alerts generated even as people move in the background.
- **Observed Result:** ✅ Passed. Verified that detections outside the user-defined ROI do not progress to the alerting stage.

## 3. Metrics

| Metric | Target | Observed (Test Runs) | Notes |
|--------|--------|----------|-------|
| **System Latency (FPS)** | ≥ 10 FPS | ~12-15 FPS | Observed on standard local CPU. (~4 FPS in headless server environment). |
| **False Positive Rate** | < 2 per 10m session | 0 | No false alerts were generated in the 'Normal' baseline test. |
| **Alert Precision** | ≥ 90% | 100% | Verified by inspecting `alerts/` snapshots; all 9 alerts correctly show humans in ROI. |
| **Temporal Efficacy** | ≥ 50% FP reduction | Verified | Temporal threshold effectively ignored 'ghosting' frames during the Abuse video interaction. |

## 4. Known Limitations
- **Occlusion:** Partial occlusions (e.g., person behind a desk) may cause detection confidence to drop below the threshold or the bounding box to fragment.
- **Low-Light Performance:** Without IR cameras, extreme low-light scenes fail to produce accurate detections, even with histogram equalisation enabled.
- **Single-Class Restriction:** The system only detects persons (Class 0). It ignores vehicles or animals entering the ROI.

## 5. Conclusion
The COGNIVIEW prototype successfully meets the goals outlined in the Product Requirements Document. The modular architecture allowed for clean implementation of temporal and confidence filters, ensuring the alert system remains highly precise.
