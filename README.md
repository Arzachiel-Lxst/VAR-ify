# VAR-ify ⚽

Video Assistant Referee Analysis System using AI

## Architecture

```
VAR-ify/
├── frontend/    → Deploy to Vercel
├── backend/     → Deploy to Railway  
└── ml/          → Deploy to Hugging Face Spaces
```

## Services

| Service | Platform | Description |
|---------|----------|-------------|
| **Frontend** | Vercel | React UI for video upload |
| **Backend** | Railway | FastAPI for file handling |
| **ML** | Hugging Face | YOLOv8 + MediaPipe analysis |

## Quick Deploy

### 1. ML Service (Hugging Face)
1. Create new Space → Select Gradio SDK
2. Upload `ml/` folder contents
3. Note your Space URL

### 2. Backend (Railway)
1. New Project → Deploy from GitHub
2. Set root directory: `backend`
3. Add env: `HF_SPACE_URL=https://your-space.hf.space`

### 3. Frontend (Vercel)
1. Import from GitHub
2. Set root directory: `frontend`
3. Add env: `VITE_API_URL=https://your-backend.railway.app`

---

# VAR-ify ⚽

**Video Assistant Referee Analysis System**

Sistem analisis video sepak bola berbasis AI untuk mendeteksi pelanggaran Handball dan Offside secara otomatis.

## 🎯 Fitur

- **🖐️ Handball Detection** - Deteksi sentuhan tangan dengan bola menggunakan pose estimation
- **🚩 Offside Detection** - Deteksi posisi offside pemain dengan perspective correction
- **⏱️ Auto-trim** - Video lebih dari 15 detik otomatis dipotong
- **📥 Download** - Download hasil video VAR analysis
- **📊 History** - Simpan riwayat analisis ke database

## 🛠️ Tech Stack

### Backend
- Python 3.11
- FastAPI
- YOLOv8 (Player Detection)
- MediaPipe (Pose Estimation)
- OpenCV (Video Processing)
- SQLAlchemy + PostgreSQL/SQLite

### Frontend
- React 18
- Vite
- TailwindCSS
- Lucide Icons

## 🚀 Quick Start

### Development (Local)

**Backend:**
```bash
cd var-backend
pip install -r requirements.txt
python api_server.py
```

**Frontend:**
```bash
cd var-frontend
npm install
npm run dev
```

Buka http://localhost:3000

### Production (Docker)

```bash
# Build dan run semua services
docker-compose up -d

# Atau build dulu
docker-compose build
docker-compose up -d
```

Buka http://localhost

## 📁 Project Structure

```
VAR-ify/
├── docker-compose.yml
├── README.md
├── var-backend/
│   ├── Dockerfile
│   ├── api_server.py      # FastAPI server
│   ├── run_var.py         # VAR analysis main
│   ├── database.py        # Database models
│   ├── requirements.txt
│   ├── app/
│   │   └── var/
│   │       ├── handball_detector.py
│   │       └── offside_detector.py
│   ├── data/
│   │   ├── uploads/       # Uploaded videos
│   │   └── results/       # Analysis results
│   └── models/            # ML models
└── var-frontend/
    ├── Dockerfile
    ├── nginx.conf
    ├── package.json
    └── src/
        ├── App.jsx        # Main React component
        ├── main.jsx
        └── index.css
```

## 🔧 Environment Variables

### Backend (.env)
```
DATABASE_URL=sqlite:///./data/varify.db
REDIS_URL=redis://localhost:6379
```

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| POST | `/api/upload` | Upload video |
| POST | `/api/analyze` | Analyze video |
| GET | `/api/results/{filename}` | Download result video |
| GET | `/api/history` | Get analysis history |
| GET | `/api/analysis/{id}` | Get specific analysis |
| GET | `/api/health` | Health check |

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| frontend | 80 | React app (nginx) |
| backend | 8000 | FastAPI server |
| db | 5432 | PostgreSQL |
| redis | 6379 | Redis cache |

## 📄 License

MIT License

---

Made with ❤️ for football analysis
