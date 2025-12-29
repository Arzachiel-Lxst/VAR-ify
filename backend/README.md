# VAR-ify Backend API

Lightweight FastAPI backend for Railway deployment.

## Features
- Video upload with auto-trim (max 15s)
- Serve result videos with CORS
- Connect to ML service on Hugging Face

## Deploy to Railway

1. Connect this repo to Railway
2. Set root directory to `/backend`
3. Add environment variable: `HF_SPACE_URL=https://your-space.hf.space`
4. Deploy!

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_SPACE_URL` | Hugging Face Space URL for ML processing |
| `PORT` | Server port (auto-set by Railway) |

## Local Development

```bash
pip install -r requirements.txt
python api_server.py
```

## API Endpoints

- `GET /` - API info
- `GET /api/health` - Health check
- `POST /api/upload` - Upload video
- `GET /api/uploads` - List uploaded videos
- `GET /results/{filename}` - Get result video
