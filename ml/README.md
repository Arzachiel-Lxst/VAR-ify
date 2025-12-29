# VAR-ify ML Service

Machine Learning service for Video Assistant Referee analysis.

## Features
- 🤚 Handball detection using MediaPipe pose estimation
- 🏃 Offside detection using YOLOv8 player tracking
- 📹 Generates highlight video with detected events

## Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload all files from this folder
4. The Space will automatically build and deploy

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

## API Usage

The Gradio interface provides:
- Video upload
- Analysis type selection (handball/offside/both)
- Results JSON output
- VAR highlight video output
