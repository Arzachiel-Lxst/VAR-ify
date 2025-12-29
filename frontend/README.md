# VAR Frontend

React frontend untuk VAR Analysis System.

## Setup

```bash
cd var-frontend
npm install
npm run dev
```

Frontend akan jalan di `http://localhost:3000`

## Backend

Pastikan backend API jalan di port 8000:

```bash
cd var-backend
python api_server.py
```

## Fitur

- Upload video (MP4, MOV, AVI)
- Analisis otomatis Handball & Offside
- Tampilkan hasil dengan confidence score
- Video hasil VAR dengan highlight pelanggaran

## Tech Stack

- React 18
- Vite
- TailwindCSS
- Lucide Icons
- Axios
