# VAR Backend

Video Assistant Referee (VAR) system for soccer highlight analysis. This system analyzes video clips to detect potential **offside** and **handball** violations.

## Features

- **Video Upload & Processing**: Upload soccer highlight clips for analysis
- **Offside Detection**: Multi-hypothesis offside analysis with confidence scoring
- **Handball Detection**: Pose-based handball detection following FIFA rules
- **Frame Quality Filtering**: Only analyzes eligible frames (good visibility, stable camera)
- **Scene Cut Detection**: Handles replays and camera transitions
- **Field Calibration**: Pixel-to-field coordinate mapping for accurate measurements

## Output Format

```json
{
  "clip_id": "match_001_clip_07",
  "events": [
    {
      "type": "offside",
      "decision": "PROBABLE",
      "confidence": 0.87,
      "reason": "Camera stable, field visible",
      "frame_index": 1245
    },
    {
      "type": "handball",
      "decision": "YES",
      "confidence": 0.91,
      "frame_index": 2310
    }
  ]
}
```

## Decision Types

| Decision | Description |
|----------|-------------|
| `YES` | Definitive violation detected |
| `NO` | Definitive no violation |
| `PROBABLE` | Likely violation but uncertainty exists |
| `NOT_DECIDABLE` | Cannot determine due to poor data quality |

## Tech Stack

- **Python 3.10+**
- **FastAPI** - Backend API framework
- **OpenCV** - Video/image processing
- **YOLOv8** - Object detection (players, ball, goal)
- **MediaPipe** - Pose estimation for handball
- **ByteTrack** - Multi-object tracking
- **Kalman Filter** - Ball trajectory prediction

## Project Structure

```
var-backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/
│   │   └── upload.py        # API endpoints
│   ├── core/
│   │   ├── config.py        # Configuration
│   │   ├── video_loader.py  # Video loading & frame extraction
│   │   ├── scene_detector.py # Scene cut detection
│   │   ├── frame_filter.py  # Frame eligibility scoring
│   │   └── camera_analyzer.py # Camera motion analysis
│   ├── ai/
│   │   ├── detector.py      # YOLO object detection
│   │   ├── tracker.py       # ByteTrack object tracking
│   │   ├── pose.py          # MediaPipe pose estimation
│   │   ├── field_calibration.py # Homography calibration
│   │   ├── ball_tracker.py  # Kalman filter ball tracking
│   │   └── decision_engine.py # Final decision logic
│   ├── services/
│   │   └── var_pipeline.py  # Main analysis pipeline
│   └── utils/
│       └── geometry.py      # Math utilities
├── models/                  # AI model weights
├── data/
│   ├── uploads/             # Uploaded videos
│   ├── frames/              # Extracted frames
│   └── results/             # Analysis results
├── requirements.txt
└── README.md
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Upload Video
```
POST /api/v1/upload
Content-Type: multipart/form-data

file: <video_file>
clip_id: (optional) custom identifier
```

### Analyze Video
```
POST /api/v1/analyze
Content-Type: application/json

{
  "clip_id": "your_clip_id",
  "analyze_offside": true,
  "analyze_handball": true
}
```

### Upload & Analyze (One Request)
```
POST /api/v1/analyze/file
Content-Type: multipart/form-data

file: <video_file>
analyze_offside: true
analyze_handball: true
```

### Get Result
```
GET /api/v1/result/{clip_id}
```

### List Clips
```
GET /api/v1/clips
```

### Delete Clip
```
DELETE /api/v1/clips/{clip_id}
```

## Pipeline Steps

1. **Video Ingestion** - Load video, extract frames at 30 FPS
2. **Scene Cut Detection** - Detect replays and camera transitions
3. **Content Classification** - Filter non-field-play frames
4. **Frame Eligibility Scoring** - Score based on grass, lines, players, stability
5. **Object Detection** - YOLOv8 detects players, ball, goal
6. **Object Tracking** - ByteTrack maintains consistent IDs
7. **Field Calibration** - Homography for pixel-to-field mapping
8. **Ball Tracking** - Kalman filter for trajectory and pass detection
9. **Offside Analysis** - Multi-hypothesis offside determination
10. **Handball Analysis** - Pose-based arm position analysis
11. **Decision Generation** - Final decisions with confidence scores

## Frame Eligibility Scoring

```
score = 0.30 * grass_visibility
      + 0.25 * field_lines_detected
      + 0.25 * players_detected
      + 0.20 * camera_stability

Eligible if score >= 0.70
```

## Expected Accuracy

| Analysis | Accuracy |
|----------|----------|
| Offside | ~80-88% |
| Handball | ~90%+ |

**Note**: This is designed for research/prototype use. Professional VAR systems use multiple synchronized cameras and EPTS data.

## Custom YOLO Model

For better accuracy, train a custom YOLO model with soccer-specific classes:
- player
- goalkeeper
- ball
- goal
- sideline
- penalty_area

Place the trained model at `models/yolo_soccer.pt`.

## License

MIT License
