import cv2
import numpy as np

def estimate_offside_line(frame, attack_direction):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=int(w * 0.3), maxLineGap=10)
    if lines is None:
        return -1
    ys = []
    for l in lines[:50]:
        x1, y1, x2, y2 = l[0]
        if abs(y2 - y1) < 8 and abs(x2 - x1) > int(w * 0.25):
            ys.append((y1 + y2) // 2)
    if not ys:
        return -1
    ys.sort()
    if attack_direction == 1:
        return int(ys[0])
    return int(ys[-1])
