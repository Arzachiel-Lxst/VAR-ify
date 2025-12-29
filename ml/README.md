---
title: VAR-ify
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# VAR-ify ML Service

Machine Learning service for Video Assistant Referee analysis.

## Features
- 🤚 Handball detection using MediaPipe pose estimation
- 🏃 Offside detection using YOLOv8 player tracking
- 📹 Generates highlight video with detected events

## Usage

Upload a football/soccer video (max 15 seconds) and select the analysis type:
- **Both**: Detect handball and offside
- **Handball**: Detect handball only
- **Offside**: Detect offside only

## Local Development

```bash
pip install -r requirements.txt
python app.py
```
