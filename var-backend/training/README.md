# Custom YOLO Training untuk Soccer VAR

## Overview

Training custom YOLOv8 model untuk mendeteksi:
- **ball** - Bola sepak
- **player** - Pemain
- **goalkeeper** - Kiper
- **referee** - Wasit

## Struktur Folder

```
training/
├── datasets/
│   └── soccer/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── configs/
│   └── soccer.yaml
├── runs/
│   └── (hasil training)
├── download_dataset.py
├── train_yolo.py
└── README.md
```

## Langkah Training

### 1. Persiapan Dataset

```bash
# Download dataset dari Roboflow (gratis)
python download_dataset.py
```

### 2. Training

```bash
# Training dengan GPU (recommended)
python train_yolo.py --epochs 100 --device 0

# Training dengan CPU (slow)
python train_yolo.py --epochs 50 --device cpu
```

### 3. Gunakan Model

Setelah training selesai, model tersimpan di:
```
training/runs/detect/soccer_yolo/weights/best.pt
```

Copy ke folder models:
```bash
copy training\runs\detect\soccer_yolo\weights\best.pt models\yolo_soccer.pt
```

## Dataset Options

1. **Roboflow Universe** (Recommended)
   - https://universe.roboflow.com/search?q=soccer%20ball
   - Format: YOLOv8
   - Gratis untuk research

2. **SoccerNet**
   - https://www.soccer-net.org/
   - Dataset profesional

3. **Manual Labeling**
   - Gunakan tool: LabelImg, CVAT, Roboflow
   - Label frame dari video sendiri

## Tips Training

1. **Minimal 500-1000 images** per class
2. **Augmentasi** sudah include di YOLOv8
3. **Epochs**: 100-300 untuk hasil optimal
4. **Batch size**: 16 (GPU), 4-8 (CPU)
5. **Image size**: 640x640 (default)

## Hardware Requirements

- **GPU**: NVIDIA dengan CUDA (GTX 1060+)
- **RAM**: 16GB+
- **Storage**: 10GB+ untuk dataset

Training dengan CPU bisa tapi sangat lambat (10x lebih lama).
