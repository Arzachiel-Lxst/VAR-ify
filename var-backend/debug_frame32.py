import cv2
import numpy as np
from app.var.offside_detector import OffsideDetector

video_path = 'data/uploads/WTF.. bagaimana bisa Offside___ coba jelaskan.mp4'
cap = cv2.VideoCapture(video_path)

detector = OffsideDetector()

# Check frame 208
cap.set(cv2.CAP_PROP_POS_FRAMES, 208)
ret, frame = cap.read()
height, width = frame.shape[:2]

players = detector.player_detector.detect(frame)

def is_maroon(p):
    h, s, v = p['hsv']
    return (h < 25 or h > 155) and s > 40

maroon = [p for p in players if is_maroon(p)]
yellow = [p for p in players if not is_maroon(p)]

print(f'Frame 208: {len(maroon)} maroon, {len(yellow)} yellow')
print('Maroon players (sorted by X - leftmost first):')
for p in sorted(maroon, key=lambda x: x['feet'][0]):
    x, y = p['feet']
    bbox = p['bbox']
    print(f'  X={x}, Y={y}, bbox={bbox}')

print('\\nYellow defenders (sorted by Y):')
for p in sorted(yellow, key=lambda x: x['feet'][1]):
    x, y = p['feet']
    print(f'  X={x}, Y={y}')

cap.release()
