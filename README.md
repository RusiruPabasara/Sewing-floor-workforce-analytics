# CCTV-Based Workforce Presence and Station-Utilization Analytics

CS5998 Capstone, MSc Data Science & AI, University of Moratuwa.

## Problem
Sewing lines depend on operators being present at their stations, but
biometric systems only record building entry and exit. This project turns
fixed-camera CCTV frames into workforce metrics: timestamp OCR, worker
detection, posture (seated vs standing/moving), station occupancy, hourly
analytics, a Streamlit dashboard and a local LLM assistant.

## Dataset
"Sewing Computer Vision Dataset" by linemanual10, Roboflow Universe,
licensed CC BY 4.0.
https://universe.roboflow.com/linemanual10/sewing-0kfql

## Status
## Data preparation
- Timestamps: template OCR, 3,997 / 4,000 frames within 1 s (50/50 hand-checked).
- Periods (`configs/periods.json`): old layout (17 Feb), pre-shift (18 Feb 06:00–07:18),
  changeover (18 Feb 07:19–07:45, front-centre line rebuilt), new layout (from 18 Feb 07:45).
- Near-duplicates: phash (64-bit) was tested first but is too coarse for this wide view:
  frames it called identical still had 2–9% of pixels changed while people worked.
  Pixel change is used instead (480×270 grey, blurred, clock masked): a frame is a repeat if
  ≤ 0.25% of pixels changed since the last kept frame. 131 frames removed, all empty floor
  before/after the shift; none in working time.
- Split by whole day (never random): train = 17–19 Feb (2,626), val = 21 Feb (496),
  test = 20 Feb (747). Lists in `configs/splits/`, YOLO file `configs/data_split.yaml`.


